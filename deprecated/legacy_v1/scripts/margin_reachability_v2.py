"""Margin-reachability TCT (v2): softmin-vs-0.5 bit margin, no weight cap, matched N=1024.

Formalizes the winning pilot configuration (pilot_margin_reachability_v2.py /
v5 / v6 grid search): for a coalition C and candidate target c_t, weights are

    a* = argmax_{a in simplex} softmin_beta( sign_j * ((C^T a)_j - 0.5) )
         s.t. 1/sum(a_i^2) >= keff_frac * K   (K_eff soft-but-enforced floor)

with beta=8, keff_frac=0.6, no per-weight cap (full simplex a_i in [0,1]).
Unlike the current tct() in framing.py (Euclidean argmin ||C^T a - c_t||^2 with
a hard 0.5 per-weight cap and separate nearest-hull target selection), this
method both selects and weights each candidate target using the same
reachability score R(a*), computed purely from coalition/target bit payloads
-- no detector calls in the optimization, matching framing.py's payload-aware
but detector-agnostic design.

Target selection: every non-coalition identity in the active N=1024 registry
is scored by R(a*); the top N_CAND=10 by that score become this trial's
targets (self-ranked, not the Euclidean nearest-hull set framing.py uses).

Checkpointing/output layout mirrors framing.py: atomic partial CSV writes,
resumable by trial, written to results/evaluation/ (not published to data/).

Usage:
    python scripts/margin_reachability_v2.py --model audioseal --K 5 --n_trials 300
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
    int_to_bits, full_registry_size,
)
from watermarks import detect_many, detect_wavmark_many, get_wavmark  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry_size_control import active_registry  # noqa: E402
from framing import restricted_top1_and_margin  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
N_CAND = 10
REGISTRY_SIZE = 1024
BETA = 8.0
KEFF_FRAC = 0.6
FIELDS = [
    "model", "K", "spk", "local_t", "gi", "N_registry",
    "target", "method", "target_top1", "target_margin",
    "reachability_score", "weights", "effective_K", "solver_success",
]


def write_rows_atomic(path, rows):
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_completed_trials(partial_csv, expected_per_trial=N_CAND):
    if not partial_csv.exists():
        return [], set()
    with partial_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_trial = {}
    for row in rows:
        by_trial.setdefault(int(row["gi"]), []).append(row)
    completed = {gi for gi, trial_rows in by_trial.items() if len(trial_rows) == expected_per_trial}
    rows = [row for row in rows if int(row["gi"]) in completed]
    return rows, completed


def softmin_bit_margin(C, c_t, beta, keff_min):
    """a* = argmax_{a in simplex, K_eff(a) >= keff_min} softmin_beta(bit margins vs 0.5).

    Returns (a, R, success). All margin terms are linear in a, so the
    objective is concave; SLSQP with an analytic gradient converges reliably
    (confirmed stable to warm-start choice in the v6 pilot).
    """
    K = C.shape[0]
    sign = 2.0 * c_t.astype(np.float64) - 1.0
    M = C.astype(np.float64) * sign[None, :]
    offset = -0.5 * sign

    def margins(a):
        return M.T @ a + offset

    def neg_obj(a):
        m = margins(a)
        return (1.0 / beta) * logsumexp(-beta * m)

    def neg_obj_grad(a):
        m = margins(a)
        w = np.exp(-beta * (m - m.max())); w = w / w.sum()
        return -(M @ w)

    cons = [
        {"type": "eq", "fun": lambda a: np.sum(a) - 1.0, "jac": lambda a: np.ones(K)},
        {"type": "ineq", "fun": lambda a: 1.0 / keff_min - np.sum(a ** 2),
         "jac": lambda a: -2.0 * a},
    ]
    bounds = [(0.0, 1.0)] * K
    a0 = np.full(K, 1.0 / K)
    res = minimize(neg_obj, a0, jac=neg_obj_grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-14})
    a = np.clip(res.x, 0, 1)
    s = a.sum()
    a = a / s if s > 1e-8 else np.ones(K) / K
    m = margins(a)
    R = -(1.0 / beta) * logsumexp(-beta * m)
    keff = 1.0 / float(np.sum(a ** 2))
    success = bool(res.success and keff + 1e-6 >= keff_min)
    return a, R, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--keff_frac", type=float, default=KEFF_FRAC)
    args = ap.parse_args()

    model, K = args.model, args.K
    d = NBITS[model]
    reg_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    keff_min = args.keff_frac * K

    trial_idx = speaker_trial_index(n_total=args.n_trials)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"margin_reachability_v2_{model}_K{K}.csv"
    partial_csv = RESULTS / f"margin_reachability_v2_{model}_K{K}.partial.csv"
    rows, completed = load_completed_trials(partial_csv)
    if completed:
        print(f"  resuming {model} K={K}: {len(completed)}/{len(trial_idx)} trials from {partial_csv.name}",
              flush=True)
    t_start = time.time()

    for gi, (spk, local_t) in enumerate(trial_idx):
        if gi in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]
        C = np.array([int_to_bits(ci, d) for ci in coll_ints])

        if reg_size <= REGISTRY_SIZE:
            active_ids = np.arange(reg_size, dtype=np.int64)
        else:
            active_ids = active_registry(reg_size, coll_ints, REGISTRY_SIZE, spk, K, local_t)
        cand_ids = active_ids[~np.isin(active_ids, np.asarray(coll_ints, dtype=np.int64))]

        scored = []
        for q_t in cand_ids.tolist():
            c_t = int_to_bits(q_t, d)
            a, R, ok = softmin_bit_margin(C, c_t, args.beta, keff_min)
            scored.append((R, q_t, a, ok))
        scored.sort(key=lambda r: r[0], reverse=True)
        top = scored[:N_CAND]

        outputs = [sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
                   for _, _, a, _ in top]

        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for (R, q_t, a, ok), (scores, _, _) in zip(top, decoded):
            top1_int, margin = restricted_top1_and_margin(scores, active_ids, q_t)
            keff = 1.0 / float(np.sum(np.asarray(a) ** 2))
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
                "N_registry": min(reg_size, REGISTRY_SIZE),
                "target": q_t, "method": "margin_reachability_v2",
                "target_top1": int(top1_int == q_t),
                "target_margin": f"{margin:.8f}",
                "reachability_score": f"{R:.8f}",
                "weights": json.dumps(np.asarray(a).tolist()),
                "effective_K": f"{keff:.8f}",
                "solver_success": int(ok),
            })

        if (gi + 1) % 10 == 0:
            write_rows_atomic(partial_csv, rows)
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)} checkpointed ({time.time()-t_start:.0f}s)",
                  flush=True)

    write_rows_atomic(partial_csv, rows)
    write_rows_atomic(out_csv, rows)

    single = np.mean([int(r["target_top1"]) for r in rows]) * 100
    by_trial = {}
    for r in rows:
        by_trial.setdefault((r["spk"], r["local_t"]), []).append(int(r["target_top1"]))
    any_hit = np.mean([1 if max(v) == 1 else 0 for v in by_trial.values()]) * 100
    mean_keff = np.mean([float(r["effective_K"]) for r in rows])
    print(f"\n=== {model} K={K} margin_reachability_v2 (n={len(trial_idx)}) ===")
    print(f"  single-candidate hit rate={single:.1f}%  any-of-{N_CAND} hit rate={any_hit:.1f}%  "
          f"mean K_eff={mean_keff:.2f}")


if __name__ == "__main__":
    main()

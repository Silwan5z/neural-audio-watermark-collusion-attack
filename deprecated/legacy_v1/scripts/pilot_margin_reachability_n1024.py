"""Pilot: bit-margin maximin reachability vs current nearest-hull TCT, both in matched N=1024.

Current framing.py pipeline (kept as baseline here):
  - target selection: rank candidates by Euclidean distance from conv(C) (convex_dist_batch_exact)
  - weights: argmin ||C^T a - c_t||^2, a in simplex with per-weight cap 0.5

New pipeline (this pilot):
  - For each candidate target c_t, treat each bit j as a side-of-0.5 decision:
    margin_j(a) = sign_j * (x_j(a) - 0.5), sign_j = +1 if c_t bit is 1 else -1,
    where x(a) = C^T a is the mixed codeword point (a in the full simplex,
    no 0.5 cap). This distinguishes a stable 0.51 (small positive margin) from
    a stable 0.60 the same way a real bit decision boundary would, instead of
    collapsing everything into one Euclidean distance.
  - reachability score R(t, C) = max_a min_j margin_j(a), i.e. the best
    worst-bit margin achievable by any mixture -- a maximin, solved as a
    linear-objective SLSQP problem (all margin constraints are linear in a).
  - to keep the mixture a genuine multi-replica collusion (not a stand-in for
    a single copy), a soft floor is enforced on the continuous effective
    participant count K_eff = 1 / sum(a_i^2) via a convex quadratic
    constraint sum(a_i^2) <= 1/K_eff_min, K_eff_min = keff_frac * K.
  - target ranking uses R(t, C) descending (most reachable first); the
    corresponding optimal a* is reused directly to mix the audio, no second
    solve.

Both variants are scored inside the same independently-sampled N=1024 active
registry so the comparison isolates the algorithm change from any
registry-size confound.

Usage: python scripts/pilot_margin_reachability_n1024.py --model audioseal --K 5 --n_trials 30
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
    int_to_bits, full_registry_size,
)
from watermarks import detect_many, detect_wavmark_many, get_wavmark  # noqa: E402
from registry_size_control import active_registry  # noqa: E402
from framing import tct, convex_dist_batch_exact, restricted_top1_and_margin, N_CAND  # noqa: E402

REGISTRY_SIZE = 1024
KEFF_FRAC = 0.6  # soft floor: K_eff >= KEFF_FRAC * K


def margin_reachability(C, c_t, keff_min):
    """Solve a* = argmax_{a in simplex} min_j sign_j * (C^T a - 0.5)_j,
    subject to a soft K_eff floor sum(a_i^2) <= 1/keff_min.

    Returns (a*, R) where R is the achieved worst-bit margin (higher = more
    reachable). All margin constraints are linear in a; the only nonlinear
    piece is the convex quadratic K_eff constraint, so SLSQP converges reliably.
    """
    K = C.shape[0]
    sign = 2.0 * c_t.astype(np.float64) - 1.0  # {-1,+1} per bit

    def neg_t(z):
        return -z[-1]

    def margin_cons(z):
        a, t = z[:-1], z[-1]
        x = C.T @ a
        return sign * (x - 0.5) - t

    def keff_cons(z):
        a = z[:-1]
        return 1.0 / keff_min - np.sum(a ** 2)

    cons = [
        {"type": "eq", "fun": lambda z: np.sum(z[:-1]) - 1.0},
        {"type": "ineq", "fun": margin_cons},
        {"type": "ineq", "fun": keff_cons},
    ]
    bounds = [(0.0, 1.0)] * K + [(-0.5, 0.5)]
    z0 = np.concatenate([np.full(K, 1.0 / K), [0.0]])
    res = minimize(neg_t, z0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-14})
    a = np.clip(res.x[:-1], 0, 1)
    s = a.sum()
    a = a / s if s > 1e-8 else np.ones(K) / K
    t = float(sign @ (C.T @ a - 0.5) if not np.all(np.isfinite(res.x)) else res.x[-1])
    return a, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="audioseal")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=30)
    args = ap.parse_args()

    model, K = args.model, args.K
    d = NBITS[model]
    reg_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)
    keff_min = KEFF_FRAC * K

    methods = ["current_tct_cap0.5", "margin_maximin_nocap"]
    per_hits = {m: [] for m in methods}
    per_any = {m: {} for m in methods}
    per_keff = {m: [] for m in methods}
    t_start = time.time()

    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]
        C = np.array([int_to_bits(ci, d) for ci in coll_ints])

        active_ids = active_registry(reg_size, coll_ints, REGISTRY_SIZE, spk, K, local_t)
        cand_ids = active_ids[~np.isin(active_ids, np.asarray(coll_ints, dtype=np.int64))]

        # --- current pipeline: nearest-hull target selection + capped TCT ---
        cur_dist = convex_dist_batch_exact(C, registry_bits[cand_ids])
        cur_cands = cand_ids[np.argsort(cur_dist)[:N_CAND]].tolist()
        cur_weights = {q_t: tct(C, int_to_bits(q_t, d), 0.5) for q_t in cur_cands}

        # --- new pipeline: margin-maximin reachability over the whole pool ---
        scored = []
        for q_t in cand_ids.tolist():
            c_t = int_to_bits(q_t, d)
            a, R = margin_reachability(C, c_t, keff_min)
            scored.append((R, q_t, a))
        scored.sort(key=lambda r: r[0], reverse=True)
        new_cands = [(q_t, a) for _, q_t, a in scored[:N_CAND]]

        outputs, tags = [], []
        for q_t in cur_cands:
            a = cur_weights[q_t]
            outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
            tags.append((q_t, "current_tct_cap0.5", a))
        for q_t, a in new_cands:
            outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
            tags.append((q_t, "margin_maximin_nocap", a))

        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for (q_t, name, a), (scores, _, _) in zip(tags, decoded):
            top1_int, margin = restricted_top1_and_margin(scores, active_ids, q_t)
            hit = int(top1_int == q_t)
            per_hits[name].append(hit)
            per_any[name].setdefault((spk, local_t), []).append(hit)
            keff = 1.0 / float(np.sum(np.asarray(a) ** 2))
            per_keff[name].append(keff)

        if (gi + 1) % 5 == 0:
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)} trials "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    print(f"\n=== pilot: {model} K={K}, N_registry={REGISTRY_SIZE}, "
          f"n_trials={args.n_trials}, N_CAND={N_CAND}, K_eff_min={keff_min:.2f} ===")
    for name in methods:
        single = np.mean(per_hits[name]) * 100
        any_hit = np.mean([1 if max(v) == 1 else 0 for v in per_any[name].values()]) * 100
        mean_keff = np.mean(per_keff[name])
        print(f"  {name:22s}: single-candidate hit rate={single:5.1f}%  "
              f"any-of-{N_CAND} hit rate={any_hit:5.1f}%  mean K_eff={mean_keff:.2f}")


if __name__ == "__main__":
    main()

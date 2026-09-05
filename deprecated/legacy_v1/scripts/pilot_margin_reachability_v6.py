"""Pilot v6: fine local search around v2's best point (beta=8, keff_frac=0.6).

v5 confirmed beta=8 beats {5,6,10} and keff_frac=0.6 beats {0 (no floor), 1.0
(uniform)} on a coarse grid, and that neither an entropy bonus nor dropping
the floor entirely improves on the hard K_eff constraint. This pilot narrows
in with two small, orthogonal changes, still purely bit-space, no detector
calls in the objective:

  1. Fine grid right around the coarse optimum: beta in {7,8,9}, keff_frac in
     {0.5, 0.6, 0.7} (9 combos) -- checks whether 8/0.6 is actually the local
     optimum or just the best of the coarse grid.
  2. Warm start: SLSQP is initialized from uniform (1/K) in every prior
     version. Try initializing instead from the current capped-TCT solution
     (tct(C, c_t, 0.5)), which is already a reasonable Euclidean-optimal
     point -- a better starting point can matter for a non-convex-ish
     landscape (the K_eff constraint boundary is curved) even if the global
     optimum is unchanged, since SLSQP is a local solver.

Usage: python scripts/pilot_margin_reachability_v6.py --model audioseal --K 5 --n_trials 40 --device cuda:1
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

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


def softmin_bit_margin(C, c_t, beta, keff_min, a0):
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
    res = minimize(neg_obj, a0, jac=neg_obj_grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-14})
    a = np.clip(res.x, 0, 1)
    s = a.sum()
    a = a / s if s > 1e-8 else np.ones(K) / K
    m = margins(a)
    R = -(1.0 / beta) * logsumexp(-beta * m)
    return a, R


FINE_GRID = [
    (f"b{beta:g}_k{keff:g}", beta, keff)
    for beta in (7.0, 8.0, 9.0)
    for keff in (0.5, 0.6, 0.7)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="audioseal")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device:
        os.environ["WATERMARK_DEVICE"] = args.device

    model, K = args.model, args.K
    d = NBITS[model]
    reg_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    methods = (["current_tct_cap0.5"] + [g[0] for g in FINE_GRID]
               + ["b8_k0.6_warmstart_tct"])
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

        cur_dist = convex_dist_batch_exact(C, registry_bits[cand_ids])
        cur_cands = cand_ids[np.argsort(cur_dist)[:N_CAND]].tolist()
        cur_weights = {q_t: tct(C, int_to_bits(q_t, d), 0.5) for q_t in cur_cands}

        outputs, tags = [], []
        for q_t in cur_cands:
            a = cur_weights[q_t]
            outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
            tags.append((q_t, "current_tct_cap0.5", a))

        uniform_a0 = np.full(K, 1.0 / K)
        for label, beta, keff_frac in FINE_GRID:
            keff_min = keff_frac * K
            scored = []
            for q_t in cand_ids.tolist():
                c_t = int_to_bits(q_t, d)
                a, R = softmin_bit_margin(C, c_t, beta, keff_min, uniform_a0)
                scored.append((R, q_t, a))
            scored.sort(key=lambda r: r[0], reverse=True)
            for _, q_t, a in scored[:N_CAND]:
                outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
                tags.append((q_t, label, a))

        # warm-start variant: beta=8, keff_frac=0.6, init from capped-TCT solution
        scored = []
        for q_t in cand_ids.tolist():
            c_t = int_to_bits(q_t, d)
            a0 = tct(C, c_t, 0.5)
            a, R = softmin_bit_margin(C, c_t, 8.0, 0.6 * K, a0)
            scored.append((R, q_t, a))
        scored.sort(key=lambda r: r[0], reverse=True)
        for _, q_t, a in scored[:N_CAND]:
            outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
            tags.append((q_t, "b8_k0.6_warmstart_tct", a))

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

    print(f"\n=== pilot v6: {model} K={K}, N_registry={REGISTRY_SIZE}, "
          f"n_trials={args.n_trials}, N_CAND={N_CAND} ===")
    for name in methods:
        single = np.mean(per_hits[name]) * 100
        any_hit = np.mean([1 if max(v) == 1 else 0 for v in per_any[name].values()]) * 100
        mean_keff = np.mean(per_keff[name])
        print(f"  {name:26s}: single={single:5.1f}%  any-of-{N_CAND}={any_hit:5.1f}%  "
              f"mean_K_eff={mean_keff:.2f}")


if __name__ == "__main__":
    main()

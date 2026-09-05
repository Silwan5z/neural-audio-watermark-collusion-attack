"""Pilot v5: tuning + regularization variants built directly on v2 (soft-min-vs-0.5).

v2's objective is R(a) = softmin_beta( sign_j * (C^T a - 0.5)_j ), maximized over
the full simplex with a hard K_eff floor (1/sum(a_i^2) >= keff_frac*K), which
beat both the original capped TCT and two competitor-aware variants (v3, v4).
This pilot stays purely in bit space (no detector calls to build the objective)
and asks three orthogonal questions about v2 itself, rather than adding new
competitor terms (already tried and failed twice):

  1. Is beta=8 actually the best temperature, or just the best of a coarse
     {4,8,16} grid? Try a tighter grid around it.
  2. Is the hard K_eff floor even necessary for this objective? The softmin
     itself already penalizes any bit sitting too close to a single replica's
     value, so it may already discourage collapse without an explicit floor.
     Test keff_frac in {0 (no constraint), 0.6 (v2's setting), 1.0 (force
     near-uniform)}.
  3. Does swapping the hard quadratic K_eff constraint for a smooth entropy
     bonus (encourages spread without a hard boundary that SLSQP must sit on)
     change anything? Variant: R(a) = softmin_beta(bit margins) + gamma *
     entropy(a), unconstrained except the simplex itself.

Usage: python scripts/pilot_margin_reachability_v5.py --model audioseal --K 5 --n_trials 40 --device cuda:1
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


def softmin_bit_margin(C, c_t, beta, keff_min=None, entropy_gamma=0.0):
    """a* = argmax softmin_beta(bit margins vs 0.5) [+ entropy_gamma * H(a)],
    over the simplex, with an optional hard K_eff floor (keff_min=None means
    no floor at all -- the constraint list is simply omitted).
    """
    K = C.shape[0]
    sign = 2.0 * c_t.astype(np.float64) - 1.0
    M = C.astype(np.float64) * sign[None, :]
    offset = -0.5 * sign

    def margins(a):
        return M.T @ a + offset

    def neg_obj(a):
        m = margins(a)
        val = -(1.0 / beta) * logsumexp(-beta * m)
        if entropy_gamma > 0:
            ap = np.clip(a, 1e-12, 1.0)
            val += entropy_gamma * float(-np.sum(ap * np.log(ap)))
        return -val

    def neg_obj_grad(a):
        m = margins(a)
        w = np.exp(-beta * (m - m.max())); w = w / w.sum()
        grad = M @ w
        if entropy_gamma > 0:
            ap = np.clip(a, 1e-12, 1.0)
            grad += entropy_gamma * (-(np.log(ap) + 1.0))
        return -grad

    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0, "jac": lambda a: np.ones(K)}]
    if keff_min is not None:
        cons.append({"type": "ineq",
                     "fun": lambda a: 1.0 / keff_min - np.sum(a ** 2),
                     "jac": lambda a: -2.0 * a})
    bounds = [(0.0, 1.0)] * K
    a0 = np.full(K, 1.0 / K)
    res = minimize(neg_obj, a0, jac=neg_obj_grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-14})
    a = np.clip(res.x, 0, 1)
    s = a.sum()
    a = a / s if s > 1e-8 else np.ones(K) / K
    m = margins(a)
    R = -(1.0 / beta) * logsumexp(-beta * m)  # report the pure softmin score (no entropy term) for comparability
    return a, R


VARIANTS = [
    # (label, beta, keff_frac_or_None, entropy_gamma)
    ("v2_ref_b8_k0.6",     8.0,  0.6, 0.0),
    ("beta5_k0.6",         5.0,  0.6, 0.0),
    ("beta6_k0.6",         6.0,  0.6, 0.0),
    ("beta10_k0.6",       10.0,  0.6, 0.0),
    ("beta8_k0_nofloor",   8.0,  None, 0.0),
    ("beta8_k1.0_uniform", 8.0,  1.0, 0.0),
    ("beta8_entropy0.05",  8.0,  None, 0.05),
    ("beta8_entropy0.15",  8.0,  None, 0.15),
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

    methods = ["current_tct_cap0.5"] + [v[0] for v in VARIANTS]
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
        cur_cands = cur_dist_ids = cand_ids[np.argsort(cur_dist)[:N_CAND]].tolist()
        cur_weights = {q_t: tct(C, int_to_bits(q_t, d), 0.5) for q_t in cur_cands}

        outputs, tags = [], []
        for q_t in cur_cands:
            a = cur_weights[q_t]
            outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
            tags.append((q_t, "current_tct_cap0.5", a))

        for label, beta, keff_frac, gamma in VARIANTS:
            keff_min = keff_frac * K if keff_frac is not None else None
            scored = []
            for q_t in cand_ids.tolist():
                c_t = int_to_bits(q_t, d)
                a, R = softmin_bit_margin(C, c_t, beta, keff_min, gamma)
                scored.append((R, q_t, a))
            scored.sort(key=lambda r: r[0], reverse=True)
            for _, q_t, a in scored[:N_CAND]:
                outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
                tags.append((q_t, label, a))

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

    print(f"\n=== pilot v5: {model} K={K}, N_registry={REGISTRY_SIZE}, "
          f"n_trials={args.n_trials}, N_CAND={N_CAND} ===")
    for name in methods:
        single = np.mean(per_hits[name]) * 100
        any_hit = np.mean([1 if max(v) == 1 else 0 for v in per_any[name].values()]) * 100
        mean_keff = np.mean(per_keff[name])
        print(f"  {name:24s}: single={single:5.1f}%  any-of-{N_CAND}={any_hit:5.1f}%  "
              f"mean_K_eff={mean_keff:.2f}")


if __name__ == "__main__":
    main()

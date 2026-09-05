"""Pilot v2: soft-min margin reachability, no cap, tunable temperature.

The v1 margin_maximin pilot (pilot_margin_reachability_n1024.py) used a hard
worst-bit min as the reachability objective. That objective only cares about
the single weakest bit and is blind to how the other d-1 bits are doing, but
the actual detector score for soft-bit models (audioseal/voicemark/wmcodec/
timbrewm) is a SUM over bits (log-likelihood), and even wavmark's hard-vote
score is (up to sign) a sum of per-bit agreement. A pure worst-bit min is a
poor proxy for a sum-based score: it can spend weight budget "rescuing" one
stubborn bit while abandoning margin on bits that were already comfortably
correct, even though sacrificing those doesn't help the sum-based competition
against the nearest rival codeword.

Fix: replace the hard min with a smooth soft-min (negative log-sum-exp),
controlled by a temperature beta:

  R_soft(a) = -(1/beta) * log( sum_j exp(-beta * margin_j(a)) )

As beta -> infinity this recovers the v1 hard maximin. As beta -> 0 it
approaches the mean margin (closer to a sum-based score, i.e. closer to what
the detector actually competes on), while still weighting weak bits more than
strong ones (unlike a plain sum, a bit sitting exactly on the 0.5 boundary
still gets extra downweighting relative to one deep in target territory).

margin_j(a) is linear in a (see below), so R_soft is smooth and concave in a,
with a closed-form gradient (a softmax reweighting of the per-bit direction
vectors) -- this lets SLSQP use exact Jacobians instead of finite differences,
which is both faster and more numerically stable than the v1 epigraph LP.

No weight cap (full simplex, alpha_i >= 0, sum alpha_i = 1); a continuous
K_eff soft floor still guards against collapsing onto a single replica.

Usage: python scripts/pilot_margin_reachability_v2.py --model audioseal --K 5 --n_trials 30 --beta 8
"""
from __future__ import annotations
import argparse
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
KEFF_FRAC = 0.6  # soft floor: K_eff >= KEFF_FRAC * K


def soft_margin_reachability(C, c_t, keff_min, beta):
    """a* = argmax_{a in simplex, K_eff(a) >= keff_min} R_soft(a).

    margin_j(a) = sign_j * ((C^T a)_j - 0.5) = (M^T a)_j - 0.5*sign_j,
    where M = C * sign[None, :] (sign in {-1,+1} per bit).  Linear in a, so
    R_soft(a) = -(1/beta) * logsumexp(-beta * margin(a)) is smooth & concave;
    its gradient is a softmax-weighted sum of M's rows (per-bit direction
    vectors), giving SLSQP an exact Jacobian instead of finite differences.
    """
    K = C.shape[0]
    sign = 2.0 * c_t.astype(np.float64) - 1.0
    M = C.astype(np.float64) * sign[None, :]  # [K, d]
    offset = -0.5 * sign  # [d]

    def margin(a):
        return M.T @ a + offset  # [d]

    def neg_R(a):
        m = margin(a)
        return (1.0 / beta) * logsumexp(-beta * m)

    def neg_R_grad(a):
        m = margin(a)
        w = np.exp(-beta * (m - m.max()))
        w = w / w.sum()
        return -(M @ w)

    def keff_cons(a):
        return 1.0 / keff_min - np.sum(a ** 2)

    def keff_cons_grad(a):
        return -2.0 * a

    cons = [
        {"type": "eq", "fun": lambda a: np.sum(a) - 1.0, "jac": lambda a: np.ones(K)},
        {"type": "ineq", "fun": keff_cons, "jac": keff_cons_grad},
    ]
    bounds = [(0.0, 1.0)] * K
    a0 = np.full(K, 1.0 / K)
    res = minimize(neg_R, a0, jac=neg_R_grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-14})
    a = np.clip(res.x, 0, 1)
    s = a.sum()
    a = a / s if s > 1e-8 else np.ones(K) / K
    R = -neg_R(a)
    return a, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="audioseal")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--beta", type=float, default=8.0)
    args = ap.parse_args()

    model, K, beta = args.model, args.K, args.beta
    d = NBITS[model]
    reg_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)
    keff_min = KEFF_FRAC * K

    methods = ["current_tct_cap0.5", f"softmin_nocap_beta{beta:g}"]
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

        scored = []
        for q_t in cand_ids.tolist():
            c_t = int_to_bits(q_t, d)
            a, R = soft_margin_reachability(C, c_t, keff_min, beta)
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
            tags.append((q_t, f"softmin_nocap_beta{beta:g}", a))

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

    print(f"\n=== pilot v2: {model} K={K}, N_registry={REGISTRY_SIZE}, "
          f"n_trials={args.n_trials}, N_CAND={N_CAND}, beta={beta}, K_eff_min={keff_min:.2f} ===")
    for name in methods:
        single = np.mean(per_hits[name]) * 100
        any_hit = np.mean([1 if max(v) == 1 else 0 for v in per_any[name].values()]) * 100
        mean_keff = np.mean(per_keff[name])
        print(f"  {name:24s}: single-candidate hit rate={single:5.1f}%  "
              f"any-of-{N_CAND} hit rate={any_hit:5.1f}%  mean K_eff={mean_keff:.2f}")


if __name__ == "__main__":
    main()

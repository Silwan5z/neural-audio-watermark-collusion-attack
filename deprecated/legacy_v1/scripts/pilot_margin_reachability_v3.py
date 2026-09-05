"""Pilot v3: competitor-aware margin reachability, no cap.

v2 measured each bit against a fixed decision line (0.5) and used a soft-min
across bits. That is still blind to who the real rival is: a target can look
very reachable bit-by-bit yet still lose top-1 because one specific rival
codeword in the registry happens to agree with the mixture on almost every
bit. Conversely a target with one "weak" bit near 0.5 may still win easily if
no rival is close on the other bits.

v3 replaces the fixed 0.5 boundary with the actual competing identities in the
active registry. For a mixture a, define a linear per-identity score

    score_i(a) = sign_i . (C^T a - 0.5),   sign_i = 2*bits_i - 1 in {-1,+1}^d

which is (up to scale) exactly the sum-of-bit-margins score a soft-bit
detector or a hard-vote Hamming detector actually competes on. Reachability is

    R(t, a) = score_t(a) - softmax_beta( {score_i(a) : i in registry, i != t} )

i.e. the target's own score minus a smooth approximation of the best rival's
score. Maximizing R(a) over a (full simplex, no cap, K_eff soft floor) both
selects and directly reachability-ranks targets: R > 0 under the smooth
approximation means the target is winning its actual competition, not just
close to a fixed line.

score_t(a) and softmax_beta(...) are both linear/concave in a (softmax of
linear functions is convex, so its negative is concave), so the objective
stays a concave maximization over a convex feasible set (simplex intersected
with the K_eff quadratic ball) -- a single well-posed SLSQP problem per
candidate target, with an exact analytic gradient (a softmax-reweighted sum of
rival direction vectors), no detector calls.

Usage: python scripts/pilot_margin_reachability_v3.py --model audioseal --K 5 --n_trials 30 --beta 4
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


def competitor_margin_reachability(C, target_idx, V_all, offset_all, keff_min, beta):
    """a* = argmax_{a in simplex, K_eff(a) >= keff_min} score_target(a) - softmax_beta(rival scores).

    V_all: [K, n_active] = C @ Sign_all.T   (Sign_all = 2*active_bits - 1)
    offset_all: [n_active] = 0.5 * sum(Sign_all, axis=1)
    score_all(a) = V_all.T @ a - offset_all   (all active identities, incl. coalition)
    target_idx: column index of the target within V_all/offset_all.
    """
    K = C.shape[0]
    n_active = V_all.shape[1]
    rival_mask = np.ones(n_active, dtype=bool)
    rival_mask[target_idx] = False
    V_riv = V_all[:, rival_mask]        # [K, n_active-1]
    off_riv = offset_all[rival_mask]    # [n_active-1]
    v_t = V_all[:, target_idx]          # [K]
    off_t = offset_all[target_idx]

    def score_t(a):
        return float(v_t @ a - off_t)

    def rival_scores(a):
        return V_riv.T @ a - off_riv    # [n_active-1]

    def neg_R(a):
        rs = rival_scores(a)
        return -(score_t(a) - (1.0 / beta) * logsumexp(beta * rs))

    def neg_R_grad(a):
        rs = rival_scores(a)
        w = np.exp(beta * (rs - rs.max()))
        w = w / w.sum()
        return -(v_t - V_riv @ w)

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
    ap.add_argument("--beta", type=float, default=4.0)
    args = ap.parse_args()

    model, K, beta = args.model, args.K, args.beta
    d = NBITS[model]
    reg_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)
    keff_min = KEFF_FRAC * K

    methods = ["current_tct_cap0.5", f"competitor_nocap_beta{beta:g}"]
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

        # baseline pipeline (unchanged)
        cur_dist = convex_dist_batch_exact(C, registry_bits[cand_ids])
        cur_cands = cand_ids[np.argsort(cur_dist)[:N_CAND]].tolist()
        cur_weights = {q_t: tct(C, int_to_bits(q_t, d), 0.5) for q_t in cur_cands}

        # new pipeline: competitor-aware margin over the whole active registry
        active_bits = registry_bits[active_ids]              # [n_active, d]
        sign_all = 2.0 * active_bits.astype(np.float64) - 1.0
        V_all = C.astype(np.float64) @ sign_all.T             # [K, n_active]
        offset_all = 0.5 * sign_all.sum(axis=1)               # [n_active]
        id_to_col = {int(v): i for i, v in enumerate(active_ids)}

        scored = []
        for q_t in cand_ids.tolist():
            col = id_to_col[q_t]
            a, R = competitor_margin_reachability(C, col, V_all, offset_all, keff_min, beta)
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
            tags.append((q_t, f"competitor_nocap_beta{beta:g}", a))

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

    print(f"\n=== pilot v3: {model} K={K}, N_registry={REGISTRY_SIZE}, "
          f"n_trials={args.n_trials}, N_CAND={N_CAND}, beta={beta}, K_eff_min={keff_min:.2f} ===")
    for name in methods:
        single = np.mean(per_hits[name]) * 100
        any_hit = np.mean([1 if max(v) == 1 else 0 for v in per_any[name].values()]) * 100
        mean_keff = np.mean(per_keff[name])
        print(f"  {name:26s}: single-candidate hit rate={single:5.1f}%  "
              f"any-of-{N_CAND} hit rate={any_hit:5.1f}%  mean K_eff={mean_keff:.2f}")


if __name__ == "__main__":
    main()

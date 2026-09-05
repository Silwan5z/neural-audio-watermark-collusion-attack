"""Pilot: TCT weight-cap ablation, both variants scored in the same matched N=1024 registry.

Compares the current TCT (per-weight cap 0.5, i.e. no single replica may
exceed half the mixture) against an uncapped TCT (cap 1.0, full simplex,
alpha_i >= 0, sum alpha_i = 1) on identical coalitions/targets. Both are
evaluated inside the same independently-sampled N=1024 active registry so the
comparison isolates the cap's effect from any registry-size confound.

This does not change any published script; it only reuses framing.py's tct()
with a different cap argument.

Usage: python scripts/pilot_cap_ablation_n1024.py --model audioseal --K 5 --n_trials 30
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
    int_to_bits, full_registry_size,
)
from watermarks import detect, detect_many, detect_wavmark_many, get_wavmark  # noqa: E402
from registry_size_control import active_registry  # noqa: E402
from framing import tct, convex_dist_batch_exact, restricted_top1_and_margin, N_CAND  # noqa: E402

REGISTRY_SIZE = 1024
CAPS = [("tct_cap0.5", 0.5), ("tct_cap1.0", 1.0)]


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

    per_cap_hits = {name: [] for name, _ in CAPS}
    per_cap_any = {name: {} for name, _ in CAPS}
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

        # Opportunistic target selection: nearest-to-hull candidates within
        # the matched N=1024 pool (small enough to score exactly, no subsampling).
        cand_dist = convex_dist_batch_exact(C, registry_bits[cand_ids])
        cands = cand_ids[np.argsort(cand_dist)[:N_CAND]].tolist()

        outputs, tags = [], []
        for q_t in cands:
            c_t = int_to_bits(q_t, d)
            for name, cap in CAPS:
                a = tct(C, c_t, cap)
                outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))
                tags.append((q_t, name))

        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for (q_t, name), (scores, _, _) in zip(tags, decoded):
            top1_int, margin = restricted_top1_and_margin(scores, active_ids, q_t)
            hit = int(top1_int == q_t)
            per_cap_hits[name].append(hit)
            per_cap_any[name].setdefault((spk, local_t), []).append(hit)

        if (gi + 1) % 5 == 0:
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)} trials "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    print(f"\n=== pilot: {model} K={K}, N_registry={REGISTRY_SIZE}, "
          f"n_trials={args.n_trials}, N_CAND={N_CAND} ===")
    for name, _ in CAPS:
        single = np.mean(per_cap_hits[name]) * 100
        any_hit = np.mean([1 if max(v) == 1 else 0 for v in per_cap_any[name].values()]) * 100
        print(f"  {name:14s}: single-candidate hit rate={single:5.1f}%  "
              f"any-of-{N_CAND} hit rate={any_hit:5.1f}%")


if __name__ == "__main__":
    main()

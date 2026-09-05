"""Detailed WavMark arbitrary-target TCT control.

Uses exactly the arbitrary target sampling and TCT construction from
``framing.py`` while additionally exporting detector margin and convex-hull
distance for every target.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (CAP, NBITS, coalition_seed, full_registry_bits,  # noqa: E402
                      full_registry_size, get_or_embed, int_to_bits,
                      sample_coalition, speaker_trial_index)
from watermarks import detect_wavmark_many, get_wavmark  # noqa: E402
from framing import N_CAND, convex_dist_batch_exact, tct  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
FIELDS = [
    "model", "trial_id", "spk", "local_t", "K", "target_id", "target_hit",
    "target_margin", "d_hull",
]


def write_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    completed = {trial for trial, rr in grouped.items() if len(rr) == N_CAND}
    return [r for r in rows if int(r["trial_id"]) in completed], completed


def detector_margin(scores: np.ndarray, target: int) -> float:
    target_score = float(scores[target])
    if target == 0:
        other_max = float(np.max(scores[1:]))
    elif target == len(scores) - 1:
        other_max = float(np.max(scores[:-1]))
    else:
        other_max = float(max(np.max(scores[:target]), np.max(scores[target + 1:])))
    return target_score - other_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--trial_start", type=int, default=0,
                        help="Inclusive global trial id (for disjoint GPU shards).")
    parser.add_argument("--trial_end", type=int, default=None,
                        help="Exclusive global trial id; defaults to n_trials.")
    parser.add_argument("--output_tag", default="",
                        help="Optional suffix for a shard's independent output files.")
    args = parser.parse_args()

    trial_end = args.n_trials if args.trial_end is None else args.trial_end
    if not 0 <= args.trial_start < trial_end <= args.n_trials:
        parser.error("require 0 <= trial_start < trial_end <= n_trials")
    if args.output_tag and not args.output_tag.replace("_", "").isalnum():
        parser.error("output_tag may contain only letters, digits, and underscores")

    model = "wavmark"
    d = NBITS[model]
    registry_bits = full_registry_bits(model)
    reg_size = full_registry_size(model)
    trials = speaker_trial_index(n_total=args.n_trials)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f".{args.output_tag}" if args.output_tag else ""
    out = RESULTS / f"tamper_arbitrary_detail_wavmark_K{args.K}{tag}.csv"
    partial = RESULTS / f"tamper_arbitrary_detail_wavmark_K{args.K}{tag}.partial.csv"
    rows, completed = load_completed(partial)
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id < args.trial_start or trial_id >= trial_end:
            continue
        if trial_id in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(rng, model, args.K)
        wavs = [get_or_embed(model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        C = np.asarray([int_to_bits(identity, d) for identity in coalition])

        coll_set = set(coalition)
        candidates = [identity for identity in range(reg_size) if identity not in coll_set]
        target_rng = np.random.default_rng(coalition_seed(spk, args.K, local_t) + 999)
        targets = target_rng.choice(candidates, size=N_CAND, replace=False).tolist()
        distances = convex_dist_batch_exact(C, registry_bits[targets])

        outputs = []
        for target in targets:
            weights = tct(C, int_to_bits(target, d), CAP)
            outputs.append(sum(weights[i] * wavs[i] for i in range(args.K)).astype(np.float32))
        decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)

        for target, distance, (scores, _, _) in zip(targets, distances, decoded):
            top1 = int(np.argsort(scores, kind="stable")[::-1][0])
            rows.append({
                "model": model, "trial_id": trial_id, "spk": spk,
                "local_t": local_t, "K": args.K, "target_id": target,
                "target_hit": int(top1 == target),
                "target_margin": f"{detector_margin(scores, target):.8f}",
                "d_hull": f"{float(distance):.8f}",
            })
        if (trial_id + 1) % 10 == 0 or trial_id + 1 == trial_end:
            write_atomic(partial, rows)
            done_in_shard = sum(args.trial_start <= t < trial_end for t in completed)
            done_in_shard += sum(
                1 for t in {int(r["trial_id"]) for r in rows}
                if args.trial_start <= t < trial_end and t not in completed
            )
            print(f"wavmark K={args.K}{tag}: shard {done_in_shard}/"
                  f"{trial_end - args.trial_start}, global trial {trial_id + 1}/{len(trials)} "
                  f"({time.time() - start:.0f}s)", flush=True)

    write_atomic(partial, rows)
    write_atomic(out, rows)
    print(f"completed {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

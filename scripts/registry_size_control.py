"""Matched registry-size control for mean/FWP attribution attacks.

The detector is evaluated once against the native full codebook.  Candidate
registries are then deterministic, independently sampled subsets that always
contain every coalition identity.  Restricting the already-computed score
vector is exactly equivalent to decoding against each subset separately while
avoiding repeated neural-network forwards.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, coalition_seed, full_registry_bits, full_registry_size,
    get_or_embed, sample_coalition, speaker_trial_index,
)
from watermarks import detect_many  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
FIELDS = [
    "model", "K", "N_registry", "trial_id", "spk", "local_t", "method",
    "colluder_ids", "top1_identity", "top1_is_colluder", "max_colluder_score",
    "max_noncolluder_score", "attribution_margin", "best_colluder_rank", "ASR",
    "R3_escape", "R5_escape",
]
SWEEP_16BIT = [256, 1024, 4096, 16384, 65536]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path, expected_per_trial: int) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    completed = {trial for trial, rr in grouped.items() if len(rr) == expected_per_trial}
    return [r for r in rows if int(r["trial_id"]) in completed], completed


def fwp(wavs: list[np.ndarray]) -> np.ndarray:
    K = len(wavs)
    best_pair, best_dist = (0, 1), -np.inf
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.mean((wavs[i] - wavs[j]) ** 2))
            if dist > best_dist:
                best_pair, best_dist = (i, j), dist
    weights = np.zeros(K, dtype=np.float64)
    weights[list(best_pair)] = 0.5
    return weights


def independent_registry_seed(spk: str, K: int, local_t: int, size: int) -> int:
    material = f"registry-control|{spk}|K={K}|t={local_t}|N={size}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def active_registry(full_size: int, coalition: list[int], size: int,
                    spk: str, K: int, local_t: int) -> np.ndarray:
    if size < len(coalition):
        raise ValueError(f"registry size {size} is smaller than coalition K={len(coalition)}")
    if size == full_size:
        return np.arange(full_size, dtype=np.int64)
    coalition_arr = np.asarray(coalition, dtype=np.int64)
    pool = np.setdiff1d(np.arange(full_size, dtype=np.int64), coalition_arr,
                        assume_unique=False)
    rng = np.random.default_rng(independent_registry_seed(spk, K, local_t, size))
    sampled = rng.choice(pool, size=size - len(coalition), replace=False)
    return np.sort(np.concatenate([coalition_arr, sampled]))


def restricted_metrics(scores: np.ndarray, active: np.ndarray,
                       coalition: list[int]) -> dict[str, int | float]:
    active_scores = np.asarray(scores)[active]
    order = np.argsort(active_scores, kind="stable")[::-1]
    ranked_ids = active[order]
    coll_set = set(coalition)
    top1 = int(ranked_ids[0])
    coll_scores = np.asarray(scores)[np.asarray(coalition, dtype=np.int64)]
    non_mask = ~np.isin(active, np.asarray(coalition, dtype=np.int64))
    non_scores = active_scores[non_mask]
    max_coll = float(np.max(coll_scores))
    max_non = float(np.max(non_scores))
    best_rank = 1 + int(np.sum(active_scores > max_coll))
    return {
        "top1_identity": top1,
        "top1_is_colluder": int(top1 in coll_set),
        "max_colluder_score": max_coll,
        "max_noncolluder_score": max_non,
        "attribution_margin": max_non - max_coll,
        "best_colluder_rank": best_rank,
        "ASR": int(top1 not in coll_set),
        "R3_escape": int(not any(int(v) in coll_set for v in ranked_ids[:3])),
        "R5_escape": int(not any(int(v) in coll_set for v in ranked_ids[:5])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"])
    parser.add_argument("--K", type=int, required=True, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--matched_only", action="store_true",
                        help="Evaluate only N=1024, including for 16-bit models.")
    args = parser.parse_args()

    full_size = full_registry_size(args.model)
    sizes = [1024] if args.matched_only or NBITS[args.model] == 10 else SWEEP_16BIT
    if any(size > full_size for size in sizes):
        raise ValueError(f"requested registry sizes {sizes} exceed native size {full_size}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"registry_control_{args.model}_K{args.K}.csv"
    partial = RESULTS / f"registry_control_{args.model}_K{args.K}.partial.csv"
    rows, completed = load_completed(partial, expected_per_trial=2 * len(sizes))
    trials = speaker_trial_index(n_total=args.n_trials)
    registry_bits = full_registry_bits(args.model)
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id in completed:
            continue
        coalition_rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(coalition_rng, args.model, args.K)
        wavs = [get_or_embed(args.model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        method_weights = [("mean", np.full(args.K, 1 / args.K)), ("fwp", fwp(wavs))]
        outputs = [sum(weights[i] * wavs[i] for i in range(args.K)).astype(np.float32)
                   for _, weights in method_weights]
        decoded = detect_many(args.model, outputs, registry_bits)

        registries = {
            size: active_registry(full_size, coalition, size, spk, args.K, local_t)
            for size in sizes
        }
        for (method, _), (scores, _, _) in zip(method_weights, decoded):
            for size in sizes:
                metrics = restricted_metrics(scores, registries[size], coalition)
                rows.append({
                    "model": args.model, "K": args.K, "N_registry": size,
                    "trial_id": trial_id, "spk": spk, "local_t": local_t,
                    "method": method, "colluder_ids": json.dumps(coalition),
                    **metrics,
                })
        if (trial_id + 1) % 10 == 0:
            write_rows_atomic(partial, rows)
            print(f"{args.model} K={args.K}: {trial_id + 1}/{len(trials)} "
                  f"({time.time() - start:.0f}s)", flush=True)

    write_rows_atomic(partial, rows)
    write_rows_atomic(out, rows)
    print(f"completed {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate uniform averaging with valid source copies."""
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

from registry import (  # noqa: E402
    NBITS, get_or_embed, full_registry_bits,
    trial_schedule, coalition_seed, source_record,
)
from watermarks import detect, detect_many, pesq_wb, stoi, si_sdr  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "average"

FIELDS = [
    "model", "k", "trial_id", "speaker", "clip_index", "source_path",
    "valid_copy_count", "payloads_tested", "mixing", "escaped",
    "closest_member_bits", "closest_member_bit_accuracy", "pesq", "stoi",
    "si_sdr",
]


def atomic_csv(path: Path, rows: list[dict]) -> None:
    """Write a restart-safe checkpoint without exposing a partial CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: str, k: int) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row["model"] != model or int(row["k"]) != k:
            raise ValueError(f"checkpoint model/k mismatch: {path}")
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    complete = {
        trial_id for trial_id, group in grouped.items() if len(group) == 1
    }
    kept = [row for row in rows if int(row["trial_id"]) in complete]
    kept.sort(key=lambda row: int(row["trial_id"]))
    return kept, complete


def metrics_of(model, waveform, coalition, registry_bits, payload_length,
               decoded=None):
    """Return the escape outcome and closest-member bit agreement."""
    scores, _, hard = (decoded if decoded is not None else
                       detect(model, waveform.astype(np.float32), registry_bits))
    rank = np.argsort(scores)[::-1]
    coalition_set = set(coalition)

    top1_ints = _rows_to_ints(rank[:1], registry_bits)
    escaped = int(len(set(top1_ints) & coalition_set) == 0)

    if hard is None:
        acc_near = None
    else:
        coalition_bits = np.array([
            _int_to_bits_row(payload, payload_length) for payload in coalition
        ])
        acc_near = int((coalition_bits == hard[None, :]).sum(axis=1).max())

    return escaped, acc_near


def _int_to_bits_row(v, d):
    return np.array([(v >> i) & 1 for i in range(d)], dtype=np.int8)


def _rows_to_ints(row_idx, registry_bits):
    d = registry_bits.shape[1]
    weights = 2 ** np.arange(d)
    return (registry_bits[row_idx] @ weights).tolist()


def valid_coalition(model, speaker, clip_slot, k, rng, registry_bits,
                    max_attempts=1000):
    """Draw k distinct payloads whose individual copies decode exactly."""
    selected, wavs, attempts = [], [], 0
    limit = 2 ** NBITS[model]
    while len(selected) < k:
        needed = k - len(selected)
        candidates = []
        while len(candidates) < needed:
            payload = int(rng.integers(0, limit))
            if payload not in selected and payload not in candidates:
                candidates.append(payload)
        candidate_wavs = [get_or_embed(model, speaker, p, clip_slot) for p in candidates]
        decoded = detect_many(model, candidate_wavs, registry_bits)
        attempts += len(candidates)
        for payload, wav, (_, _, hard) in zip(candidates, candidate_wavs, decoded):
            if hard is None:
                continue
            got = int(sum(int(v) << i for i, v in enumerate(hard)))
            if got == payload:
                selected.append(payload); wavs.append(wav)
        if attempts >= max_attempts and len(selected) < k:
            raise RuntimeError(
                f"{model} {speaker} clip_slot={clip_slot}: only {len(selected)}/{k} "
                f"valid payloads after {attempts} attempts")
    return selected, wavs, attempts



def main():
    ap = argparse.ArgumentParser(
        description="Evaluate uniform averaging with valid source copies.")
    ap.add_argument("--model", choices=tuple(NBITS), required=True)
    ap.add_argument("--k", type=int, choices=(2, 3, 5), required=True)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--output-dir", type=Path, default=RESULTS)
    ap.add_argument("--shard-id", type=int, default=0,
                    help="zero-based trial shard (default: 0)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of disjoint trial shards (default: 1)")
    args = ap.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require 0 <= shard-id < num-shards")

    model = args.model
    k = args.k
    d = NBITS[model]

    registry_bits = full_registry_bits(model)
    trial_idx = trial_schedule(n_total=args.n_trials)

    output_dir = args.output_dir / f"k{k}"
    stem = model
    shard_suffix = ("" if args.num_shards == 1
                    else f".shard{args.shard_id}of{args.num_shards}")
    out_csv = output_dir / f"{stem}{shard_suffix}.csv"
    partial_csv = output_dir / f"{stem}{shard_suffix}.partial.csv"
    rows, complete = load_checkpoint(partial_csv, model, k)
    # When a previously single-worker run exists, preserve it as the immutable
    # base and distribute only its missing global trial indices across shards.
    base_complete: set[int] = set()
    if args.num_shards > 1:
        base_partial = output_dir / f"{stem}.partial.csv"
        _, base_complete = load_checkpoint(base_partial, model, k)
    assigned = [
        (gi, item) for gi, item in enumerate(trial_idx)
        if gi not in base_complete and gi % args.num_shards == args.shard_id
    ]
    assigned_ids = {gi for gi, _ in assigned}
    if not complete.issubset(assigned_ids):
        raise ValueError(f"shard checkpoint contains unassigned trials: {partial_csv}")
    if complete:
        print(f"resume {model} k={k} shard={args.shard_id}/{args.num_shards}: "
              f"{len(complete)}/{len(assigned)} trials", flush=True)
    t_start = time.time()
    for gi, (speaker, clip_slot) in assigned:
        if gi in complete:
            continue
        rng = np.random.default_rng(coalition_seed(speaker, k, clip_slot))
        source = source_record(speaker, clip_slot)
        coll_ints, wavs, payloads_tested = valid_coalition(
            model, speaker, clip_slot, k, rng, registry_bits)
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]

        average_weights = np.ones(k) / k

        reference = wavs[0]

        output = sum(
            average_weights[index] * wavs[index] for index in range(k)
        ).astype(np.float32)
        decoded = detect_many(model, [output], registry_bits)[0]
        escaped, agreement = metrics_of(
            model, output, coll_ints, registry_bits, d, decoded)
        rows.append({
            "model": model, "k": k, "trial_id": gi, "speaker": speaker,
            "clip_index": source["clip_index"],
            "source_path": source["path"],
            "valid_copy_count": k, "payloads_tested": payloads_tested,
            "mixing": "uniform", "escaped": escaped,
            "closest_member_bits": "" if agreement is None else agreement,
            "closest_member_bit_accuracy": (
                "" if agreement is None else f"{agreement/d:.4f}"),
            "pesq": f"{pesq_wb(reference, output):.4f}",
            "stoi": f"{stoi(reference, output):.4f}",
            "si_sdr": f"{si_sdr(reference, output):.2f}",
        })
        complete.add(gi)
        if len(complete) % 10 == 0:
            rows.sort(key=lambda row: int(row["trial_id"]))
            atomic_csv(partial_csv, rows)
            elapsed = time.time() - t_start
            print(f"  {model} k={k} shard={args.shard_id}/{args.num_shards}: "
                  f"{len(complete)}/{len(assigned)}  ({elapsed:.0f}s)", flush=True)

    rows.sort(key=lambda row: int(row["trial_id"]))
    atomic_csv(partial_csv, rows)
    atomic_csv(out_csv, rows)

    print(f"\n=== {model} k={k} shard={args.shard_id}/{args.num_shards} "
          f"summary (n={len(assigned)}) ===")
    escapes = [int(row["escaped"]) for row in rows]
    agreements = [float(row["closest_member_bit_accuracy"]) for row in rows
                  if row["closest_member_bit_accuracy"] != ""]
    print(
        f"  tracing_failure_pct={100.0 * np.mean(escapes):.1f} "
        f"closest_member_bit_accuracy="
        f"{np.mean(agreements) if agreements else float('nan'):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge and validate the final Payload Match and Target-Bit Margin shards."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
METHODS = ("payload_match", "bit_margin")
KS = (5, 8)


def merge_one(source: Path, output: Path, model: str, method: str, k: int,
              num_shards: int) -> None:
    pattern = f"{model}.shard*of{num_shards}.csv"
    shard_dir = source / f"k{k}" / method / "shards"
    paths = [p for p in sorted(shard_dir.glob(pattern))
             if not p.name.endswith(".partial.csv")]
    if len(paths) != num_shards:
        raise RuntimeError(
            f"{model} k={k} {method}: expected {num_shards} shards, "
            f"got {len(paths)}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise RuntimeError(f"schema mismatch: {path}")
            rows.extend(reader)

    rows.sort(key=lambda row: (int(row["trial_id"]), int(row["target_rank"])))
    trials = {int(row["trial_id"]) for row in rows}
    if len(rows) != 3000 or trials != set(range(300)):
        raise RuntimeError(
            f"{model} k={k} {method}: expected 300 trials/3000 rows, "
            f"got {len(trials)} trials/{len(rows)} rows")
    for trial in range(300):
        ranks = {int(row["target_rank"]) for row in rows
                 if int(row["trial_id"]) == trial}
        if ranks != set(range(1, 11)):
            raise RuntimeError(f"{model} k={k} {method}: incomplete trial {trial}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    hits = sum(int(row["target_hit"]) for row in rows)
    print(f"{output}: rows={len(rows)} mean_hits_per_10={hits / 300:.6f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=root / "results" / "targeted")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "targeted")
    parser.add_argument("--num-shards", type=int, default=7)
    args = parser.parse_args()

    for model in MODELS:
        for k in KS:
            for method in METHODS:
                merge_one(
                    args.input_dir,
                    args.output_dir / f"k{k}" / method / f"{model}.csv",
                    model, method, k, args.num_shards)


if __name__ == "__main__":
    main()

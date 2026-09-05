#!/usr/bin/env python3
"""Merge restart-safe attack shards into the canonical 300-trial CSV."""
from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path


FIELDS = [
    "model", "K", "spk", "local_t", "clip_index", "source_path",
    "source_exact_count", "source_payload_attempts", "gi", "method",
    "ASR", "R3_escape", "R5_escape", "ACC_near", "ACC_near_norm",
    "AggResid", "PESQ", "STOI", "SI_SDR",
]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if list(rows[0]) != FIELDS if rows else False:
        raise ValueError(f"unexpected columns: {path}")
    return rows


def atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stem = f"attack_{args.model}_K{args.K}"
    sources = [args.output_dir / f"{stem}.partial.csv"]
    sources += [
        args.output_dir / f"{stem}.shard{i}of{args.num_shards}.csv"
        for i in range(args.num_shards)
    ]
    merged: dict[tuple[int, str], dict] = {}
    for source in sources:
        for row in read_rows(source):
            if row["model"] != args.model or int(row["K"]) != args.K:
                raise ValueError(f"model/K mismatch in {source}")
            key = (int(row["gi"]), row["method"])
            if key in merged and merged[key] != row:
                raise ValueError(f"conflicting duplicate {key} in {source}")
            merged[key] = row

    expected = {(gi, "mean") for gi in range(args.n_trials)}
    if set(merged) != expected:
        missing = sorted(expected - set(merged))
        extra = sorted(set(merged) - expected)
        raise ValueError(f"incomplete merge: missing={missing[:10]} extra={extra[:10]}")
    rows = [merged[key] for key in sorted(merged)]
    if any(int(row["source_exact_count"]) != args.K for row in rows):
        raise ValueError("source-correct validation failed")
    if len({row["source_path"] for row in rows}) != args.n_trials:
        raise ValueError("expected one distinct source path per trial")
    clips = {value: 0 for value in (1, 2, 3)}
    for row in rows:
        clips[int(row["clip_index"])] += 1
    if clips != {1: 100, 2: 100, 3: 100}:
        raise ValueError(f"unexpected clip distribution: {clips}")

    atomic_write(args.output_dir / f"{stem}.partial.csv", rows)
    atomic_write(args.output_dir / f"{stem}.csv", rows)
    print({"model": args.model, "K": args.K, "trials": len(rows),
           "unique_sources": len({row['source_path'] for row in rows}),
           "clip_counts": clips})


if __name__ == "__main__":
    main()

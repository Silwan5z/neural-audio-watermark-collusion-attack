#!/usr/bin/env python3
"""Merge uniform-averaging shards and require 300 complete trials."""
from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
KS = (2, 3, 5)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge_one(directory: Path, model: str, k: int, n_trials: int) -> Path:
    output = directory / f"k{k}" / f"{model}.csv"
    candidates = []
    if output.exists():
        candidates.append(output)
    base_partial = output.with_suffix(".partial.csv")
    if base_partial.exists():
        candidates.append(base_partial)
    candidates.extend(sorted(output.parent.glob(f"{model}.shard*of*.csv")))
    candidates = [path for path in candidates
                  if not path.name.endswith(".partial.csv")
                  or path == base_partial]
    if not candidates:
        raise FileNotFoundError(f"no uniform-average records for {model} k={k}")

    fields: list[str] | None = None
    by_trial: dict[int, dict[str, str]] = {}
    for path in candidates:
        observed_fields, rows = read_rows(path)
        if fields is None:
            fields = observed_fields
        elif observed_fields != fields:
            raise RuntimeError(f"schema mismatch: {path}")
        for row in rows:
            if row["model"] != model or int(row["k"]) != k:
                raise RuntimeError(f"model/k mismatch: {path}")
            trial = int(row["trial_id"])
            if trial in by_trial and by_trial[trial] != row:
                raise RuntimeError(
                    f"conflicting duplicate for {model} k={k} trial={trial}")
            by_trial[trial] = row

    expected = set(range(n_trials))
    actual = set(by_trial)
    if actual != expected:
        raise RuntimeError(
            f"{model} k={k}: expected trials 0--{n_trials - 1}; "
            f"missing={sorted(expected - actual)[:10]}, "
            f"extra={sorted(actual - expected)[:10]}")
    assert fields is not None
    rows = [by_trial[trial] for trial in range(n_trials)]
    atomic_csv(output, fields, rows)
    print(f"{output}: {len(rows)} complete trials")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "results" / "average")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--k", type=int, choices=KS)
    parser.add_argument("--n-trials", type=int, default=300)
    args = parser.parse_args()
    models = (args.model,) if args.model else MODELS
    ks = (args.k,) if args.k else KS
    for model in models:
        for k in ks:
            merge_one(args.input_dir, model, k, args.n_trials)


if __name__ == "__main__":
    main()

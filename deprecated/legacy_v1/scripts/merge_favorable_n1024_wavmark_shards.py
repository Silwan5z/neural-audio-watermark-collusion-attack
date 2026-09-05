#!/usr/bin/env python3
"""Merge historical WavMark checkpoints with isolated favorable-N1024 shards."""
from __future__ import annotations

import argparse
import csv
import os
import uuid
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"
EXPECTED_PER_TRIAL = 20


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, choices=[2, 3, 5, 8])
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()

    stem = f"tamper_N1024_wavmark_K{args.K}"
    base = RESULTS / f"{stem}.partial.csv"
    paths = [base] + [RESULTS / f"{stem}.{tag}.csv" for tag in args.tags]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing shard inputs: " + ", ".join(missing))

    by_key: dict[tuple[int, str, int], dict[str, str]] = {}
    for path in paths:
        for row in read(path):
            key = (int(row["gi"]), row["method"], int(row["target"]))
            previous = by_key.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate {key} in {path}")
            by_key[key] = row

    rows = sorted(by_key.values(), key=lambda row: (
        int(row["gi"]), int(row["target"]), 0 if row["method"] == "mean" else 1))
    counts = Counter(int(row["gi"]) for row in rows)
    if set(counts) != set(range(300)):
        absent = sorted(set(range(300)) - set(counts))
        raise ValueError(f"missing trials: {absent}")
    bad = {trial: count for trial, count in counts.items() if count != EXPECTED_PER_TRIAL}
    if bad:
        raise ValueError(f"incomplete trials: {bad}")
    if len(rows) != 6000:
        raise ValueError(f"expected 6000 rows, got {len(rows)}")

    final = RESULTS / f"{stem}.csv"
    write_atomic(base, rows)
    write_atomic(final, rows)
    print(f"PASS K={args.K}: trials=300 rows=6000 -> {final}")


if __name__ == "__main__":
    main()

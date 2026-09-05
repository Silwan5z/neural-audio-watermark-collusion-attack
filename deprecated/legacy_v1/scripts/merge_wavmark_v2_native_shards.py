"""Merge resumable WavMark native-v2 shards into canonical result files."""
from __future__ import annotations

import csv
import os
import uuid
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"
EXPECTED_TRIALS = set(range(300))
TARGETS_PER_TRIAL = 10


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def merge_k(k: int) -> None:
    stem = f"margin_reachability_v2_native_wavmark_K{k}"
    prefix = RESULTS / f"{stem}.partial.csv"
    shard_paths = sorted(
        path for path in RESULTS.glob(f"{stem}.shard_*_*.csv")
        if not path.name.endswith(".partial.csv")
    )
    if not prefix.exists():
        raise FileNotFoundError(f"missing prefix checkpoint: {prefix}")
    if not shard_paths:
        raise FileNotFoundError(f"no completed shards found for K={k}")

    merged: dict[tuple[int, int], dict[str, str]] = {}
    sources = [prefix, *shard_paths]
    for path in sources:
        for row in read_rows(path):
            key = (int(row["gi"]), int(row["target"]))
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate {key} in {path}")
            merged[key] = row

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for (gi, _), row in merged.items():
        grouped[gi].append(row)
    if set(grouped) != EXPECTED_TRIALS:
        missing = sorted(EXPECTED_TRIALS - set(grouped))
        extra = sorted(set(grouped) - EXPECTED_TRIALS)
        raise ValueError(f"K={k}: incomplete trial coverage; missing={missing}, extra={extra}")
    bad = {gi: len(rows) for gi, rows in grouped.items()
           if len(rows) != TARGETS_PER_TRIAL}
    if bad:
        raise ValueError(f"K={k}: expected 10 unique targets per trial; bad={bad}")

    rows = [row for gi in range(300)
            for row in sorted(grouped[gi], key=lambda item: int(item["target"]))]
    output = RESULTS / f"{stem}.csv"
    write_atomic(output, rows)
    print(f"merged K={k}: {len(rows)} rows from {len(sources)} files -> {output}")


def main() -> None:
    for k in (2, 3, 5, 8):
        merge_k(k)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge disjoint trial shards with schema, range, and duplicate checks."""
from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected_trials", type=int, default=300)
    ap.add_argument("--rows_per_trial", type=int, required=True)
    ap.add_argument("--trial_field", default="trial_id")
    ap.add_argument("inputs", nargs="+", type=Path)
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            current = reader.fieldnames
            if not current:
                raise ValueError(f"missing CSV header: {path}")
            if fields is None:
                fields = current
            elif current != fields:
                raise ValueError(f"schema mismatch: {path}: {current} != {fields}")
            rows.extend(reader)

    assert fields is not None
    counts: dict[int, int] = {}
    for row in rows:
        trial = int(row[args.trial_field])
        counts[trial] = counts.get(trial, 0) + 1
    expected_ids = set(range(args.expected_trials))
    actual_ids = set(counts)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"trial coverage mismatch: missing={missing[:20]} extra={extra[:20]}")
    bad = {trial: count for trial, count in counts.items()
           if count != args.rows_per_trial}
    if bad:
        raise ValueError(f"rows-per-trial mismatch: {dict(list(sorted(bad.items()))[:20])}")

    sort_fields = [name for name in
                   [args.trial_field, "N_registry", "method", "selector", "target"]
                   if name in fields]
    rows.sort(key=lambda row: tuple(
        int(row[name]) if row[name].lstrip("-").isdigit() else row[name]
        for name in sort_fields))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, args.output)
    print(f"merged {len(args.inputs)} shards -> {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()

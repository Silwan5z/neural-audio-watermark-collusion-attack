#!/usr/bin/env python3
"""Merge seven disjoint one-bit range shards into paper-ready full CSVs."""
from __future__ import annotations

import csv
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_onebit_k5_pair_analysis import FIELDNAMES, schedule  # noqa: E402

OUT = ROOT / "results" / "one_bit_k5_20260828"


def write(path: Path, rows: list[dict]):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    expected = schedule()
    for model in ("audioseal", "voicemark"):
        rows = []
        for shard in range(7):
            path = OUT / f"onebit_k5_{model}_shard{shard}of7.partial.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
        rows.sort(key=lambda r: int(r["trial_id"]))
        ids = [int(r["trial_id"]) for r in rows]
        if ids != list(range(300)):
            raise ValueError(f"{model}: expected exact trial IDs 0..299, got {len(ids)} rows")
        for row, plan in zip(rows, expected):
            for key in ("trial_id", "spk", "local_t", "coalition_member_index",
                        "base_payload", "flipped_payload", "flipped_bit"):
                if str(row[key]) != str(plan[key]):
                    raise ValueError(f"{model}: schedule mismatch trial={plan['trial_id']} key={key}")
        write(OUT / f"onebit_k5_{model}_full.csv", rows)
        print(f"merged {model}: {len(rows)} rows")


if __name__ == "__main__":
    main()

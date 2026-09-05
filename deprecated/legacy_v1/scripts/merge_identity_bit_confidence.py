#!/usr/bin/env python3
"""Merge and validate identity-bit-confidence outputs."""
from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path


FIELDS = (
    "model", "K", "trial_id", "condition", "identity", "nbits",
    "min_bit_confidence", "weakest_bit", "source_path", "target_rank",
    "confidence_kind",
)
EXPECTED_MRC = {
    "audioseal": 294,
    "wavmark": 300,
    "timbrewm": 300,
    "voicemark": 19,
    "wmcodec": 34,
}


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()

    for model, expected_mrc in EXPECTED_MRC.items():
        sharded = sorted(args.input_dir.glob(
            f"identity_bit_confidence_k8_{model}_shard*of7.csv"))
        if sharded:
            rows = [row for path in sharded for row in read_rows(path)]
        else:
            final = args.input_dir / f"identity_bit_confidence_k8_{model}.csv"
            partial = args.input_dir / f"identity_bit_confidence_k8_{model}.partial.csv"
            source = final if final.exists() else partial
            rows = read_rows(source)

        keyed: dict[tuple[int, str], dict] = {}
        for row in rows:
            key = (int(row["trial_id"]), row["condition"])
            if key in keyed and keyed[key] != row:
                raise RuntimeError(f"conflicting duplicate for {model}: {key}")
            keyed[key] = row
        rows = sorted(keyed.values(), key=lambda row: (int(row["trial_id"]), row["condition"]))

        counts = {
            condition: sum(row["condition"] == condition for row in rows)
            for condition in ("benign_copy", "uniform_collusion", "successful_mrc")
        }
        expected = {
            "benign_copy": 300,
            "uniform_collusion": 300,
            "successful_mrc": expected_mrc,
        }
        if counts != expected:
            raise RuntimeError(f"invalid coverage for {model}: {counts}, expected {expected}")
        trials = {
            condition: len({int(row["trial_id"]) for row in rows if row["condition"] == condition})
            for condition in counts
        }
        if trials != counts:
            raise RuntimeError(f"duplicate trial-condition rows for {model}: {trials}")
        final = args.input_dir / f"identity_bit_confidence_k8_{model}.csv"
        atomic_csv(final, rows)
        print(f"PASS model={model} rows={len(rows)} counts={counts} output={final}")


if __name__ == "__main__":
    main()

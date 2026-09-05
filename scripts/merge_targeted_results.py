#!/usr/bin/env python3
"""Merge the final K=5/K=8 PM and MRC shards used by the paper."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
METHODS = ("pm", "mrc")
KS = (5, 8)


def merge_one(source: Path, output: Path, model: str, method: str, k: int) -> None:
    pattern = f"mrc_pm_native_{method}_{model}_K{k}_shard*of7.csv"
    paths = [p for p in sorted((source / "shards").glob(pattern))
             if not p.name.endswith(".partial.csv")]
    if len(paths) != 7:
        raise RuntimeError(f"{model} K={k} {method}: expected 7 shards, got {len(paths)}")

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
            f"{model} K={k} {method}: expected 300 trials/3000 rows, "
            f"got {len(trials)} trials/{len(rows)} rows")
    for trial in range(300):
        ranks = {int(row["target_rank"]) for row in rows
                 if int(row["trial_id"]) == trial}
        if ranks != set(range(1, 11)):
            raise RuntimeError(f"{model} K={k} {method}: incomplete trial {trial}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    hits = sum(int(row["target_top1"]) for row in rows)
    print(f"{output}: rows={len(rows)} mean_hits_per_10={hits / 300:.6f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=root / "results" / "mrc_pm_native_shared4_fulltop10_300clips_20260904")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "targeted")
    args = parser.parse_args()

    for model in MODELS:
        for k in KS:
            for method in METHODS:
                merge_one(
                    args.input_dir,
                    args.output_dir / f"{model}_k{k}_{method}.csv",
                    model, method, k)


if __name__ == "__main__":
    main()

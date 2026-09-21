#!/usr/bin/env python3
"""Recompute Target-Bit Margin paper results from released attempts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "k", "method", "trials", "targets_per_trial",
    "valid_copy_count", "mean_hits_out_of_10", "mean_pesq", "mean_stoi",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data" / "targeted")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "summary" /
                        "targeted_hits.csv")
    args = parser.parse_args()
    summary = []
    for model in MODELS:
        for k in (5, 8):
            path = (args.input_dir / f"k{k}" / "target_bit_margin" /
                    f"{model}.csv")
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 3000:
                raise RuntimeError(f"{path}: expected 3000 target attempts")
            trials = {int(row["trial_id"]) for row in rows}
            if trials != set(range(300)):
                raise RuntimeError(f"{path}: incomplete trial IDs")
            summary.append({
                "model": model,
                "k": k,
                "method": "Target-Bit Margin",
                "trials": len(trials),
                "targets_per_trial": 10,
                "valid_copy_count": k,
                "mean_hits_out_of_10": f"{sum(int(row['target_hit']) for row in rows)/300:.6f}",
                "mean_pesq": f"{np.mean([float(row['pesq']) for row in rows]):.6f}",
                "mean_stoi": f"{np.mean([float(row['stoi']) for row in rows]):.6f}",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"{args.output}: {len(summary)} rows")


if __name__ == "__main__":
    main()

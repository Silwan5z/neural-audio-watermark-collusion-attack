#!/usr/bin/env python3
"""Recompute the uniform-averaging paper summary from trial records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "k", "trials", "speakers", "clips", "valid_copy_count",
    "tracing_failure_pct", "mean_pesq", "mean_stoi",
)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data" / "average")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "summary" /
                        "average_results.csv")
    args = parser.parse_args()
    summary = []
    for model in MODELS:
        for k in (2, 3, 5, 8):
            if k == 8:
                rows = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(
                            (args.input_dir / "k8" / model).glob("trial_*.json"))]
            else:
                rows = csv_rows(args.input_dir / f"k{k}" / f"{model}.csv")
            if len(rows) != 300:
                raise RuntimeError(f"{model} k={k}: expected 300 trials")
            if {int(row["trial_id"]) for row in rows} != set(range(300)):
                raise RuntimeError(f"{model} k={k}: incomplete trial IDs")
            summary.append({
                "model": model,
                "k": k,
                "trials": len(rows),
                "speakers": len({row["speaker"] for row in rows}),
                "clips": len({row["source_path"] for row in rows}),
                "valid_copy_count": k,
                "tracing_failure_pct": f"{100*np.mean([int(row['escaped']) for row in rows]):.6f}",
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

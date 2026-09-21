#!/usr/bin/env python3
"""Merge five model-specific confidence-screen summaries for the paper."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "k", "speakers", "trials", "single_acceptance_pct",
    "average_rejection_pct", "target_success_before_pct",
    "target_success_after_pct",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "results" / "confidence_screening")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "summary" /
                        "confidence_screening.csv")
    args = parser.parse_args()
    rows = []
    for model in MODELS:
        path = args.input_dir / f"{model}_summary.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
        if len(source) != 1 or source[0]["model"] != model:
            raise RuntimeError(f"unexpected confidence summary: {path}")
        row = source[0]
        rows.append({
            "model": model,
            "k": 8,
            "speakers": int(row["n_speakers"]),
            "trials": int(row["n_trials"]),
            "single_acceptance_pct": f"{float(row['single_acceptance_pct']):.1f}",
            "average_rejection_pct": f"{float(row['average_rejection_pct']):.1f}",
            "target_success_before_pct": f"{float(row['target_success_before_pct']):.1f}",
            "target_success_after_pct": f"{float(row['target_success_after_pct']):.1f}",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()

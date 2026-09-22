#!/usr/bin/env python3
"""Export the sufficient statistics and fold thresholds behind Table 4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from screen_confidence import MODELS, load_rows, passes


ROOT = Path(__file__).resolve().parents[1]
RECORD_FIELDS = (
    "model", "k", "trial_id", "speaker", "fold", "condition", "identity",
    "target_rank", "minimum_confidence", "mean_confidence",
    "log_confidence_variance", "source_path", "confidence_source",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "results" / "confidence_full")
    parser.add_argument("--screening-dir", type=Path,
                        default=ROOT / "results" / "confidence_screening")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "supplementary" /
                        "confidence_screening")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "data" / "summary" /
                        "confidence_screening.csv")
    args = parser.parse_args()

    public_records: list[dict] = []
    public_folds: list[dict] = []
    manifests = []
    summary = {row["model"]: row for row in read_csv(args.summary)}
    for model in MODELS:
        fold_path = args.screening_dir / f"{model}_folds.csv"
        manifest_path = args.screening_dir / f"{model}_manifest.json"
        folds = read_csv(fold_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        speaker_fold = {
            speaker: int(record["fold"])
            for record in manifest["fold_records"]
            for speaker in record["test_speakers"]
        }
        if len(speaker_fold) != 100:
            raise RuntimeError(f"{model}: expected 100 speakers in fold manifest")
        for row in load_rows(args.input_dir, model):
            public_records.append({
                field: (speaker_fold[row["speaker"]] if field == "fold"
                        else row[field])
                for field in RECORD_FIELDS
            })
        public_folds.extend(folds)
        manifests.append(manifest)

        thresholds = {
            int(row["fold"]): (
                float(row["threshold_minimum"]),
                float(row["threshold_mean"]),
                float(row["threshold_log_variance"]),
            ) for row in folds
        }
        records = [row for row in public_records if row["model"] == model]
        accepted = {
            condition: sum(
                passes(row, thresholds[int(row["fold"])])
                for row in records if row["condition"] == condition)
            for condition in ("single", "average", "targeted")
        }
        counts = {
            condition: sum(row["condition"] == condition for row in records)
            for condition in ("single", "average", "targeted")
        }
        expected = summary[model]
        observed = (
            round(100 * accepted["single"] / counts["single"], 1),
            round(100 * (1 - accepted["average"] / counts["average"]), 1),
            round(100 * counts["targeted"] / 3000, 1),
            round(100 * accepted["targeted"] / 3000, 1),
        )
        wanted = tuple(float(expected[name]) for name in (
            "single_acceptance_pct", "average_rejection_pct",
            "target_success_before_pct", "target_success_after_pct"))
        if observed != wanted:
            raise RuntimeError(
                f"{model}: exported records produce {observed}, expected {wanted}")

    public_records.sort(key=lambda row: (
        row["model"], int(row["trial_id"]), row["condition"],
        int(row["target_rank"] or 0)))
    public_folds.sort(key=lambda row: (row["model"], int(row["fold"])))
    write_csv(args.output_dir / "records.csv", RECORD_FIELDS, public_records)
    write_csv(args.output_dir / "fold_thresholds.csv",
              tuple(public_folds[0]), public_folds)
    release_manifest = {
        "purpose": "sufficient statistics, speaker folds, and thresholds for independently reproducing Table 4",
        "targeted_rows": "exact target hits before screening; non-hits remain in the fixed 3000-attempt denominator per model",
        "waveforms_released": False,
        "bit_vectors_released": False,
        "model_manifests": manifests,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(release_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output_dir}: {len(public_records)} records, "
          f"{len(public_folds)} fold thresholds")


if __name__ == "__main__":
    main()

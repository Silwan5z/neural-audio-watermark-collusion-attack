#!/usr/bin/env python3
"""End-to-end smoke test for the model-free confidence-screening stage."""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "model", "k", "trial_id", "speaker", "condition", "identity",
    "target_rank", "bit_count", "minimum_confidence", "mean_confidence",
    "log_confidence_variance", "bit_probabilities", "bit_confidences",
    "source_path", "confidence_source",
)


def record(trial: int, condition: str, minimum: float, mean: float,
           log_variance: float, target_rank: str = "") -> dict:
    return {
        "model": "timbrewm",
        "k": 8,
        "trial_id": trial,
        "speaker": f"speaker_{trial // 3:03d}",
        "condition": condition,
        "identity": trial,
        "target_rank": target_rank,
        "bit_count": 10,
        "minimum_confidence": minimum,
        "mean_confidence": mean,
        "log_confidence_variance": log_variance,
        "bit_probabilities": "[]",
        "bit_confidences": "[]",
        "source_path": f"clip_{trial:03d}.wav",
        "confidence_source": "synthetic test",
    }


class ConfidenceScreeningTest(unittest.TestCase):
    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            rows = []
            for trial in range(300):
                # Repeating Single levels give every speaker fold a
                # non-degenerate calibration distribution.
                offset = (trial % 40) / 1000.0
                rows.append(record(trial, "single", 0.80 + offset,
                                   0.90 + offset, -7.0 + offset))
                rows.append(record(trial, "average", 0.52, 0.65, -4.0))
                rows.append(record(trial, "targeted", 0.60, 0.72, -5.0, "1"))
            with (input_dir / "timbrewm.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            command = [
                sys.executable, str(ROOT / "scripts" / "screen_confidence.py"),
                "--model", "timbrewm", "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

            with (output_dir / "timbrewm_summary.csv").open(
                    newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            with (output_dir / "timbrewm_folds.csv").open(
                    newline="", encoding="utf-8") as handle:
                folds = list(csv.DictReader(handle))
            self.assertEqual(len(folds), 5)
            self.assertEqual(int(summary["n_trials"]), 300)
            self.assertEqual(int(summary["n_targeted_exact_hits"]), 300)
            self.assertAlmostEqual(
                float(summary["target_success_before_pct"]), 10.0)
            self.assertGreaterEqual(
                float(summary["average_rejection_pct"]), 99.0)
            self.assertLessEqual(
                float(summary["target_success_after_pct"]), 10.0)


if __name__ == "__main__":
    unittest.main()

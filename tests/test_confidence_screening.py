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
sys.path.insert(0, str(ROOT / "scripts"))

from screen_confidence import calibrate  # noqa: E402

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
    def test_calibration_does_not_force_joint_retention(self) -> None:
        rows = [record(index, "single", 0.9, 0.95, -8.0)
                for index in range(100)]
        for index in range(5):
            rows[index]["minimum_confidence"] = 0.1 + 0.1 * index
            rows[index + 5]["mean_confidence"] = 0.2 + 0.1 * index
            rows[index + 10]["log_confidence_variance"] = -1.0 + 0.1 * index
        _, _, marginal, joint = calibrate(rows, 0.95, 0.001)
        self.assertTrue(all(value >= 0.95 for value in marginal))
        self.assertLess(joint, 0.95)

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
            for fold in folds:
                for field in (
                        "train_minimum_acceptance_pct",
                        "train_mean_acceptance_pct",
                        "train_log_variance_acceptance_pct"):
                    self.assertGreaterEqual(float(fold[field]), 95.0)

    def test_each_threshold_is_calibrated_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            rows = []
            for trial in range(300):
                # Put low values for the three statistics on different trials.
                # Each marginal threshold must retain at least 95%, even though
                # the three-rule conjunction can retain less.
                minimum = 0.90 if trial % 20 else 0.40
                mean = 0.95 if trial % 20 != 1 else 0.50
                log_variance = -8.0 if trial % 20 != 2 else -1.0
                rows.append(record(
                    trial, "single", minimum, mean, log_variance))
                rows.append(record(trial, "average", 0.35, 0.45, -0.5))
            with (input_dir / "timbrewm.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "screen_confidence.py"),
                "--model", "timbrewm", "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
            ], check=True, capture_output=True, text=True)

            with (output_dir / "timbrewm_folds.csv").open(
                    newline="", encoding="utf-8") as handle:
                folds = list(csv.DictReader(handle))
            for fold in folds:
                for field in (
                        "train_minimum_acceptance_pct",
                        "train_mean_acceptance_pct",
                        "train_log_variance_acceptance_pct"):
                    self.assertGreaterEqual(float(fold[field]), 95.0)


if __name__ == "__main__":
    unittest.main()

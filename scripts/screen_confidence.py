#!/usr/bin/env python3
"""Calibrate and evaluate the three-statistic confidence screen."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FOLD_FIELDS = (
    "model", "fold", "n_train_speakers", "n_train_single",
    "n_test_speakers", "n_test_trials", "calibrated_z",
    "threshold_minimum", "threshold_mean", "threshold_log_variance",
    "train_single_acceptance_pct", "test_single_acceptance_pct",
    "test_average_rejection_pct", "test_targeted_hits_before",
    "test_targeted_hits_after",
)
SUMMARY_FIELDS = (
    "model", "n_speakers", "n_trials", "n_targeted_exact_hits",
    "single_acceptance_pct", "average_rejection_pct",
    "target_success_before_pct", "target_success_after_pct",
    "targeted_exact_hit_rejection_pct", "mean_train_single_acceptance_pct",
    "mean_calibrated_z",
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_rows(root: Path, model: str) -> list[dict]:
    merged = root / f"{model}.csv"
    paths = [merged] if merged.exists() else sorted(
        path for path in root.glob(f"{model}.shard*of*.csv")
        if not path.name.endswith(".partial.csv"))
    if not paths:
        raise FileNotFoundError(f"no full confidence records for {model} in {root}")
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    counts = Counter(row["condition"] for row in rows)
    if counts["single"] != 300 or counts["average"] != 300:
        raise RuntimeError(f"expected 300 Single and Average rows, got {counts}")
    if set(counts) - {"single", "average", "targeted"}:
        raise RuntimeError(f"unexpected confidence condition(s): {counts}")
    keys = [
        (int(row["trial_id"]), row["condition"], row["target_rank"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate trial/condition/target-rank confidence rows")
    required = {
        "minimum_confidence", "mean_confidence", "log_confidence_variance"
    }
    if rows and not required <= set(rows[0]):
        raise RuntimeError(f"full confidence columns missing: {sorted(required)}")
    values = np.asarray([
        [float(row[name]) for name in sorted(required)] for row in rows
    ])
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite confidence statistic")
    return rows


def passes(row: dict, thresholds: tuple[float, float, float]) -> bool:
    minimum, mean, log_variance = thresholds
    return (
        float(row["minimum_confidence"]) >= minimum
        and float(row["mean_confidence"]) >= mean
        and float(row["log_confidence_variance"]) <= log_variance
    )


def calibrate(train: list[dict], retention: float, z_step: float
              ) -> tuple[float, tuple[float, float, float], float]:
    minimum = np.asarray([float(row["minimum_confidence"]) for row in train])
    mean = np.asarray([float(row["mean_confidence"]) for row in train])
    log_variance = np.asarray([
        float(row["log_confidence_variance"]) for row in train])
    for z in np.arange(0.0, 6.0 + z_step / 2.0, z_step):
        tail = float(norm.cdf(-z))
        thresholds = (
            float(np.quantile(minimum, tail, method="linear")),
            float(mean.mean() - z * mean.std(ddof=0)),
            float(log_variance.mean() + z * log_variance.std(ddof=0)),
        )
        accepted = np.mean(
            (minimum >= thresholds[0])
            & (mean >= thresholds[1])
            & (log_variance <= thresholds[2]))
        if accepted + 1e-12 >= retention:
            return float(z), thresholds, float(accepted)
    raise RuntimeError("no z in [0, 6] reaches the requested Single retention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results" / "confidence_full")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "confidence_screening")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--retention", type=float, default=0.95)
    parser.add_argument("--z-step", type=float, default=0.001)
    parser.add_argument("--targets-per-trial", type=int, default=10)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not 0.0 < args.retention < 1.0:
        parser.error("--retention must be between 0 and 1")
    if args.z_step <= 0.0:
        parser.error("--z-step must be positive")
    if args.targets_per_trial < 1:
        parser.error("--targets-per-trial must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_dir, args.model)
    single = [row for row in rows if row["condition"] == "single"]
    average = [row for row in rows if row["condition"] == "average"]
    targeted = [row for row in rows if row["condition"] == "targeted"]
    speakers = sorted({row["speaker"] for row in single})
    if len(speakers) != 100:
        raise RuntimeError(f"expected 100 speakers, got {len(speakers)}")
    counts = Counter(row["speaker"] for row in single)
    if set(counts.values()) != {3}:
        raise RuntimeError("each speaker must contribute exactly three Single outputs")
    speaker_by_trial = {int(row["trial_id"]): row["speaker"] for row in single}
    for row in average + targeted:
        if row["speaker"] != speaker_by_trial[int(row["trial_id"])]:
            raise RuntimeError(
                f"speaker mismatch at trial {row['trial_id']} ({row['condition']})")

    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(speakers, dtype=object)[rng.permutation(len(speakers))]
    test_folds = [list(group) for group in np.array_split(shuffled, args.folds)]
    fold_rows: list[dict] = []
    decisions: dict[tuple[int, str, str], bool] = {}
    fold_manifest = []
    for fold_index, test_speakers in enumerate(test_folds, start=1):
        test_set = set(test_speakers)
        train_set = set(speakers) - test_set
        train = [row for row in single if row["speaker"] in train_set]
        test_single = [row for row in single if row["speaker"] in test_set]
        test_average = [row for row in average if row["speaker"] in test_set]
        test_targeted = [row for row in targeted if row["speaker"] in test_set]
        z, thresholds, train_acceptance = calibrate(
            train, args.retention, args.z_step)
        for row in test_single + test_average + test_targeted:
            key = (int(row["trial_id"]), row["condition"], row["target_rank"])
            decisions[key] = passes(row, thresholds)
        single_acceptance = np.mean([passes(row, thresholds) for row in test_single])
        average_rejection = 1.0 - np.mean([
            passes(row, thresholds) for row in test_average])
        target_after = sum(passes(row, thresholds) for row in test_targeted)
        fold_rows.append({
            "model": args.model,
            "fold": fold_index,
            "n_train_speakers": len(train_set),
            "n_train_single": len(train),
            "n_test_speakers": len(test_set),
            "n_test_trials": len(test_single),
            "calibrated_z": f"{z:.3f}",
            "threshold_minimum": f"{thresholds[0]:.16g}",
            "threshold_mean": f"{thresholds[1]:.16g}",
            "threshold_log_variance": f"{thresholds[2]:.16g}",
            "train_single_acceptance_pct": 100.0 * train_acceptance,
            "test_single_acceptance_pct": 100.0 * single_acceptance,
            "test_average_rejection_pct": 100.0 * average_rejection,
            "test_targeted_hits_before": len(test_targeted),
            "test_targeted_hits_after": target_after,
        })
        fold_manifest.append({
            "fold": fold_index,
            "test_speakers": test_speakers,
            "calibrated_z": z,
            "threshold_minimum": thresholds[0],
            "threshold_mean": thresholds[1],
            "threshold_log_variance": thresholds[2],
        })

    single_after = sum(decisions[(int(row["trial_id"]), "single", "")]
                       for row in single)
    average_after = sum(decisions[(int(row["trial_id"]), "average", "")]
                        for row in average)
    targeted_after = sum(decisions[(int(row["trial_id"]), "targeted", row["target_rank"])]
                         for row in targeted)
    denominator = len(single) * args.targets_per_trial
    summary = [{
        "model": args.model,
        "n_speakers": len(speakers),
        "n_trials": len(single),
        "n_targeted_exact_hits": len(targeted),
        "single_acceptance_pct": 100.0 * single_after / len(single),
        "average_rejection_pct": 100.0 * (1.0 - average_after / len(average)),
        "target_success_before_pct": 100.0 * len(targeted) / denominator,
        "target_success_after_pct": 100.0 * targeted_after / denominator,
        "targeted_exact_hit_rejection_pct": (
            math.nan if not targeted else 100.0 * (1.0 - targeted_after / len(targeted))),
        "mean_train_single_acceptance_pct": float(np.mean([
            float(row["train_single_acceptance_pct"]) for row in fold_rows])),
        "mean_calibrated_z": float(np.mean([
            float(row["calibrated_z"]) for row in fold_rows])),
    }]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / f"{args.model}_folds.csv", FOLD_FIELDS, fold_rows)
    atomic_csv(args.output_dir / f"{args.model}_summary.csv", SUMMARY_FIELDS, summary)
    manifest = {
        "model": args.model,
        "seed": args.seed,
        "folds": args.folds,
        "fold_assignment": "sorted speakers, seeded NumPy Generator permutation, contiguous folds",
        "speaker_grouping": "all three utterances from one speaker stay together",
        "threshold_data": "training Single only",
        "retention": args.retention,
        "z_step": args.z_step,
        "variance_transform": "log(population variance + 1e-12)",
        "mean_rule": "training mean minus z times population standard deviation",
        "log_variance_rule": "training mean plus z times population standard deviation",
        "minimum_rule": "linear empirical quantile at Gaussian lower-tail probability Phi(-z)",
        "acceptance_rule": "minimum and mean at or above thresholds; log variance at or below threshold",
        "attack_data_used_for_thresholds": False,
        "targets_per_trial": args.targets_per_trial,
        "fold_records": fold_manifest,
    }
    atomic_text(
        args.output_dir / f"{args.model}_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary[0], indent=2), flush=True)


if __name__ == "__main__":
    main()

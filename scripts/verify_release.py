#!/usr/bin/env python3
"""Verify every released record without running model inference."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import wave
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
SHARED_MODELS = ("audioseal", "wavmark", "voicemark", "wmcodec")
METHODS = ("target_bit_margin",)
METHOD_LABELS = {"Target-Bit Margin": "target_bit_margin"}
ACTIVE_DATA = (
    "average", "coalitions", "targets", "targeted", "one_bit", "confidence",
    "summary", "supplementary",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def int_to_bits(value: int, length: int) -> np.ndarray:
    return np.asarray([(value >> index) & 1 for index in range(length)],
                      dtype=np.int8)


def bits_to_int(bits) -> int:
    return int(sum(int(value) << index for index, value in enumerate(bits)))


def require_close(observed: float, expected: float, message: str,
                  tolerance: float = 1e-5) -> None:
    require(abs(observed - expected) <= tolerance,
            f"{message}: {observed} != {expected}")


def target_selection_score(coalition: list[int], target: int,
                           weights: np.ndarray, bit_count: int,
                           beta: float = 8.0) -> float:
    coalition_bits = np.stack([
        int_to_bits(payload, bit_count) for payload in coalition
    ]).astype(np.float64)
    target_bits = int_to_bits(target, bit_count).astype(np.float64)
    sign = 2.0 * target_bits - 1.0
    margins = (coalition_bits * sign[None, :]).T @ weights - 0.5 * sign
    values = -beta * margins
    maximum = float(values.max())
    logsumexp = maximum + float(np.log(np.exp(values - maximum).sum()))
    return -logsumexp / beta


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_average() -> None:
    summary = read_csv(DATA / "summary" / "average_results.csv")
    expected = {(row["model"], int(row["k"])): row for row in summary}
    require(len(expected) == 20, "average summary must contain 20 model/k rows")

    expected_header = [
        "model", "k", "trial_id", "speaker", "clip_index", "source_path",
        "valid_copy_count", "payloads_tested", "mixing", "escaped",
        "closest_member_bits", "closest_member_bit_accuracy", "pesq", "stoi",
        "si_sdr",
    ]
    for model in MODELS:
        for k in (2, 3, 5):
            path = DATA / "average" / f"k{k}" / f"{model}.csv"
            require(header(path) == expected_header, f"unexpected schema: {path}")
            rows = read_csv(path)
            require(len(rows) == 300, f"{path}: expected 300 rows")
            require({int(row["trial_id"]) for row in rows} == set(range(300)),
                    f"{path}: incomplete trial IDs")
            require(len({row["source_path"] for row in rows}) == 300,
                    f"{path}: source paths are not unique")
            require(len({row["speaker"] for row in rows}) == 100,
                    f"{path}: expected 100 speakers")
            require({row["mixing"] for row in rows} == {"uniform"},
                    f"{path}: mixing must be uniform")
            require({int(row["valid_copy_count"]) for row in rows} == {k},
                    f"{path}: invalid source copy")
            value = 100.0 * sum(int(row["escaped"]) for row in rows) / 300
            target = expected[(model, k)]
            require_close(value, float(target["tracing_failure_pct"]),
                          f"{path}: tracing failure")
            require_close(
                float(np.mean([float(row["pesq"]) for row in rows])),
                float(target["mean_pesq"]), f"{path}: mean PESQ")
            require_close(
                float(np.mean([float(row["stoi"]) for row in rows])),
                float(target["mean_stoi"]), f"{path}: mean STOI")

    required = {
        "condition", "model", "k", "trial_id", "speaker", "clip_index",
        "source_path", "sample_rate", "bit_count", "bit_order",
        "coalition_seed", "coalition_payloads", "coalition_bits", "weights",
        "mixing", "source_copies", "valid_copy_count", "decoded_payload",
        "decoded_bits", "bit_probabilities", "confidence_source",
        "watermark_score", "escaped", "ones_in_coalition",
        "strict_majority_bit_accuracy",
        "unanimous_bit_accuracy", "all_strict_majority_bits",
        "closest_member_bits", "closest_member_bit_accuracy", "pesq", "stoi",
    }
    for model in MODELS:
        paths = sorted((DATA / "average" / "k8" / model).glob("trial_*.json"))
        records = [read_json(path) for path in paths]
        require(len(records) == 300, f"{model} k=8: expected 300 records")
        require(all(required == set(row) for row in records),
                f"{model} k=8: unexpected schema")
        require({row["condition"] for row in records} == {"average"},
                f"{model} k=8: inconsistent condition")
        require({row["mixing"] for row in records} == {"uniform"},
                f"{model} k=8: mixing must be uniform")
        require(len({row["source_path"] for row in records}) == 300,
                f"{model} k=8: source paths are not unique")
        require(len({row["speaker"] for row in records}) == 100,
                f"{model} k=8: expected 100 speakers")
        require({int(row["valid_copy_count"]) for row in records} == {8},
                f"{model} k=8: invalid source copy")
        for row in records:
            bit_count = int(row["bit_count"])
            require(len(row["coalition_payloads"]) == 8,
                    f"{model} trial {row['trial_id']}: bad coalition length")
            require(len(row["decoded_bits"]) == bit_count,
                    f"{model} trial {row['trial_id']}: bad decoded bit length")
            require(len(row["bit_probabilities"]) == bit_count,
                    f"{model} trial {row['trial_id']}: bad confidence length")
            require(len(row["source_copies"]) == 8 and
                    all(int(copy["valid"]) == 1 for copy in row["source_copies"]),
                    f"{model} trial {row['trial_id']}: invalid source copy")
            require(all(set(copy) == {
                        "assigned_payload", "decoded_payload", "valid",
                        "watermark_score"} for copy in row["source_copies"]),
                        f"{model} trial {row['trial_id']}: bad source-copy schema")
            decoded_bits = np.asarray(row["decoded_bits"], dtype=np.int8)
            coalition = [int(value) for value in row["coalition_payloads"]]
            coalition_bits = np.asarray(row["coalition_bits"], dtype=np.int8)
            require(bits_to_int(decoded_bits) == int(row["decoded_payload"]),
                    f"{model} trial {row['trial_id']}: payload/bit mismatch")
            require(np.array_equal(
                coalition_bits,
                np.stack([int_to_bits(payload, bit_count)
                          for payload in coalition])),
                f"{model} trial {row['trial_id']}: coalition bit mismatch")
            ones = coalition_bits.sum(axis=0)
            require(np.array_equal(ones, np.asarray(row["ones_in_coalition"])),
                    f"{model} trial {row['trial_id']}: bit-support mismatch")
            closest = int((coalition_bits == decoded_bits[None, :]).sum(axis=1).max())
            require(closest == int(row["closest_member_bits"]),
                    f"{model} trial {row['trial_id']}: closest-member mismatch")
            require_close(
                closest / bit_count, float(row["closest_member_bit_accuracy"]),
                f"{model} trial {row['trial_id']}: closest-member accuracy")
            require(int(row["escaped"]) == int(int(row["decoded_payload"]) not in coalition),
                    f"{model} trial {row['trial_id']}: escape mismatch")
        value = 100.0 * sum(int(row["escaped"]) for row in records) / 300
        target = expected[(model, 8)]
        require_close(value, float(target["tracing_failure_pct"]),
                      f"{model} k=8: tracing failure")
        require_close(float(np.mean([float(row["pesq"]) for row in records])),
                      float(target["mean_pesq"]), f"{model} k=8: mean PESQ")
        require_close(float(np.mean([float(row["stoi"]) for row in records])),
                      float(target["mean_stoi"]), f"{model} k=8: mean STOI")
    print("PASS average: 20 model/k cells with 300 valid trials each")


def verify_coalitions() -> dict[tuple[int, int], list[int]]:
    coalitions: dict[tuple[int, int], list[int]] = {}
    for k in (5, 8):
        paths = sorted((DATA / "coalitions" / f"k{k}").glob("trial_*.json"))
        require(len(paths) == 300, f"k{k}: expected 300 coalition records")
        for trial, path in enumerate(paths):
            row = read_json(path)
            require(set(row) == {
                        "dataset", "selection_rule", "shared_models", "k",
                        "trial_id", "speaker", "clip_index", "source_path",
                        "coalition_payloads", "valid_copy_count",
                        "payloads_tested", "candidate_pool_size"},
                    f"unexpected coalition schema in {path}")
            payloads = [int(value) for value in row["coalition_payloads"]]
            require(row["dataset"] == "collusion_300", f"bad dataset in {path}")
            require(int(row["trial_id"]) == trial, f"bad trial ID in {path}")
            require(int(row["k"]) == k and len(payloads) == k,
                    f"bad coalition size in {path}")
            require(len(set(payloads)) == k, f"duplicate payload in {path}")
            require(set(row["shared_models"]) == set(SHARED_MODELS),
                    f"bad shared-model list in {path}")
            require(row["valid_copy_count"] ==
                    {model: k for model in SHARED_MODELS},
                    f"source validation mismatch in {path}")
            coalitions[(k, trial)] = payloads

        for method in METHODS:
            paths = sorted(
                (DATA / "targets" / method / f"k{k}").glob("trial_*.json"))
            require(len(paths) == 300,
                    f"{method} k{k}: expected 300 target records")
            for trial, path in enumerate(paths):
                row = read_json(path)
                require(set(row) == {
                            "dataset", "method", "k", "coalition_payloads",
                            "targets"},
                        f"unexpected target schema in {path}")
                coalition = coalitions[(k, trial)]
                require(row["dataset"] == "collusion_300", f"bad dataset in {path}")
                require(row["method"] == method, f"bad method in {path}")
                require(int(row["k"]) == k, f"bad k in {path}")
                require(row["coalition_payloads"] == coalition,
                        f"coalition mismatch in {path}")
                targets = row["targets"]
                require(len(targets) == 10, f"{path}: expected ten targets")
                require([int(item["rank"]) for item in targets] == list(range(1, 11)),
                        f"{path}: target ranks are not 1--10")
                payloads = [int(item["target_payload"]) for item in targets]
                require(len(set(payloads)) == 10, f"{path}: duplicate targets")
                require(not set(payloads) & set(coalition),
                        f"{path}: coalition member selected as target")
                for item in targets:
                    require(set(item) == {
                                "rank", "target_payload", "selection_score",
                                "weights", "weights_valid", "payload_distance"},
                            f"unexpected selected-target schema in {path}")
                    weights = np.asarray(item["weights"], dtype=float)
                    require(len(weights) == k and abs(weights.sum() - 1.0) < 1e-6,
                            f"{path}: invalid weights")
                    require(bool(item["weights_valid"]),
                            f"{path}: invalid weight optimization")
    print("PASS coalitions: shared coalitions and ten Target-Bit Margin targets")
    return coalitions


def verify_targeted(coalitions: dict[tuple[int, int], list[int]]) -> None:
    summary = read_csv(DATA / "summary" / "targeted_hits.csv")
    expected = {
        (row["model"], int(row["k"]), METHOD_LABELS[row["method"]]): row
        for row in summary
    }
    require(len(expected) == 10, "targeted summary must contain 10 rows")
    expected_header = [
        "dataset", "model", "k", "method", "trial_id", "speaker",
        "clip_index", "source_path", "coalition_payloads", "valid_copy_count",
        "payloads_tested", "target_rank", "target_payload", "selection_rule",
        "candidate_count", "selection_score", "payload_distance", "weights",
        "effective_members", "max_weight", "weights_valid", "decoded_payload",
        "target_hit", "target_margin",
        "hits_out_of_10", "escaped", "closest_member_bit_accuracy",
        "watermark_score", "decoded_bits", "target_bit_accuracy",
        "quality_reference", "pesq", "stoi", "si_sdr",
    ]
    rules = {"target_bit_margin": "largest_minimum_target_bit_margin"}
    for model in MODELS:
        for k in (5, 8):
            for method in METHODS:
                path = DATA / "targeted" / f"k{k}" / method / f"{model}.csv"
                require(header(path) == expected_header, f"unexpected schema: {path}")
                rows = read_csv(path)
                require(len(rows) == 3000, f"{path}: expected 3000 rows")
                counts = Counter(int(row["trial_id"]) for row in rows)
                require(counts == Counter({trial: 10 for trial in range(300)}),
                        f"{path}: expected ten rows per trial")
                require({row["model"] for row in rows} == {model},
                        f"{path}: model mismatch")
                require({int(row["k"]) for row in rows} == {k},
                        f"{path}: k mismatch")
                require({row["method"] for row in rows} == {method},
                        f"{path}: method mismatch")
                require({row["selection_rule"] for row in rows} == {rules[method]},
                        f"{path}: target rule mismatch")
                require({int(row["valid_copy_count"]) for row in rows} == {k},
                        f"{path}: invalid source copy")
                by_trial: dict[int, list[dict[str, str]]] = {}
                for row in rows:
                    by_trial.setdefault(int(row["trial_id"]), []).append(row)
                for trial, group in by_trial.items():
                    require({int(row["target_rank"]) for row in group}
                            == set(range(1, 11)),
                            f"{path}: target ranks missing for trial {trial}")
                    coalition = json.loads(group[0]["coalition_payloads"])
                    if model in SHARED_MODELS:
                        require(coalition == coalitions[(k, trial)],
                                f"{path}: shared coalition mismatch")
                        cached = read_json(
                            DATA / "targets" / method / f"k{k}" /
                            f"trial_{trial:03d}.json")
                        require(
                            [int(item["target_payload"]) for item in cached["targets"]]
                            == [int(row["target_payload"]) for row in group],
                            f"{path}: selected targets mismatch cache")
                    hits = sum(int(row["target_hit"]) for row in group)
                    require({int(row["hits_out_of_10"]) for row in group} == {hits},
                            f"{path}: per-trial hit count mismatch")
                    for row in group:
                        decoded = int(row["decoded_payload"])
                        target = int(row["target_payload"])
                        bit_count = 10 if model == "timbrewm" else 16
                        weights = np.asarray(json.loads(row["weights"]), dtype=float)
                        require(len(weights) == k and np.all(weights >= -1e-12),
                                f"{path}: invalid weight vector")
                        require_close(float(weights.sum()), 1.0,
                                      f"{path}: weights do not sum to one",
                                      tolerance=1e-6)
                        effective = 1.0 / float(np.sum(weights ** 2))
                        require(effective + 1e-6 >= 0.6 * k,
                                f"{path}: effective coalition constraint failed")
                        require_close(effective, float(row["effective_members"]),
                                      f"{path}: effective-members mismatch",
                                      tolerance=1e-6)
                        require_close(float(weights.max()), float(row["max_weight"]),
                                      f"{path}: max-weight mismatch",
                                      tolerance=1e-6)
                        selection_score = target_selection_score(
                            coalition, target, weights, bit_count)
                        require_close(selection_score, float(row["selection_score"]),
                                      f"{path}: selection-score mismatch",
                                      tolerance=2e-7)
                        decoded_bits = json.loads(row["decoded_bits"])
                        require(bits_to_int(decoded_bits) == decoded,
                                f"{path}: decoded payload/bit mismatch")
                        target_accuracy = np.mean(
                            np.asarray(decoded_bits) == int_to_bits(target, bit_count))
                        require_close(target_accuracy,
                                      float(row["target_bit_accuracy"]),
                                      f"{path}: target-bit accuracy mismatch",
                                      tolerance=1e-7)
                        require(int(row["target_hit"]) == int(decoded == target),
                                f"{path}: target-hit flag mismatch")
                        require(int(row["escaped"]) == int(decoded not in coalition),
                                f"{path}: escape flag mismatch")
                value = sum(int(row["target_hit"]) for row in rows) / 300.0
                target = expected[(model, k, method)]
                require_close(value, float(target["mean_hits_out_of_10"]),
                              f"{path}: mean hits")
                require_close(float(np.mean([float(row["pesq"]) for row in rows])),
                              float(target["mean_pesq"]), f"{path}: mean PESQ")
                require_close(float(np.mean([float(row["stoi"]) for row in rows])),
                              float(target["mean_stoi"]), f"{path}: mean STOI")
    print("PASS targeted: 10 files, 300 trials x ten selected targets")


def verify_one_bit() -> None:
    pair_header = [
        "model", "k", "trial_id", "speaker", "clip_index", "source_path",
        "coalition_payloads", "coalition_member_index", "base_payload",
        "flipped_payload", "flipped_bit", "pair_attempts", "sample_count",
        "waveform_mse", "mel_distance", "band_energy_0_1k",
        "band_energy_1_2k", "band_energy_2_4k", "band_energy_4_8k",
        "band_fraction_0_1k", "band_fraction_1_2k", "band_fraction_2_4k",
        "band_fraction_4_8k", "base_decoded_payload",
        "flipped_decoded_payload", "base_payload_correct",
        "flipped_payload_correct", "base_changed_bit_correct",
        "flipped_changed_bit_correct", "decoded_bit_changes",
        "base_changed_bit_score", "flipped_changed_bit_score",
        "changed_bit_score_change", "unchanged_bit_score_change",
        "base_bit_scores", "flipped_bit_scores",
    ]
    for model in ("audioseal", "voicemark"):
        path = DATA / "one_bit" / "pairs" / f"{model}.csv"
        require(header(path) == pair_header, f"unexpected schema: {path}")
        rows = read_csv(path)
        require(len(rows) == 300, f"{path}: expected 300 rows")
        require(len({row["speaker"] for row in rows}) == 100,
                f"{path}: expected 100 speakers")
        require(len({row["source_path"] for row in rows}) == 300,
                f"{path}: source paths are not unique")
        for row in rows:
            base = int(row["base_payload"])
            flipped = int(row["flipped_payload"])
            require((base ^ flipped).bit_count() == 1,
                    f"{path}: pair does not differ by one bit")
            require(int(row["base_payload_correct"]) == 1 and
                    int(row["flipped_payload_correct"]) == 1,
                    f"{path}: invalid source copy")
    paths = read_csv(DATA / "one_bit" / "paths.csv")
    require(header(DATA / "one_bit" / "paths.csv") == [
        "model", "trial_id", "base_payload", "flipped_payload",
        "weight_count", "direct_path", "has_other_payload",
        "unique_payload_count", "other_payload_count",
    ], "unexpected one-bit path summary schema")
    require(len(paths) == 600, "one-bit path summary must have 600 rows")
    direct = {
        model: sum(int(row["direct_path"]) for row in paths
                   if row["model"] == model)
        for model in ("audioseal", "voicemark")
    }
    require(direct == {"audioseal": 299, "voicemark": 113},
            f"unexpected one-bit path counts: {direct}")
    print("PASS one-bit paths: AudioSeal 299 direct; VoiceMark 113 direct")


def verify_confidence() -> None:
    expected_counts = {
        "audioseal": {"single": 300, "average": 300, "targeted": 294},
        "wavmark": {"single": 300, "average": 300, "targeted": 300},
        "timbrewm": {"single": 300, "average": 300, "targeted": 300},
        "voicemark": {"single": 300, "average": 300, "targeted": 19},
        "wmcodec": {"single": 300, "average": 300, "targeted": 34},
    }
    expected_header = [
        "model", "k", "trial_id", "condition", "decoded_payload",
        "bit_count", "minimum_confidence", "weakest_bit_index",
        "source_path", "target_rank", "confidence_source",
    ]
    for model, counts in expected_counts.items():
        path = DATA / "confidence" / f"{model}.csv"
        require(header(path) == expected_header, f"unexpected schema: {path}")
        rows = read_csv(path)
        require(Counter(row["condition"] for row in rows) == counts,
                f"{path}: condition counts mismatch")
        require(all(0.0 <= float(row["minimum_confidence"]) <= 1.0
                    for row in rows),
                f"{path}: confidence outside [0,1]")
    print("PASS confidence: Single, Average, and Targeted counts match")


def verify_summaries() -> None:
    expected = {
        "average_results.csv": (
            ["model", "k", "trials", "speakers", "clips",
             "valid_copy_count", "tracing_failure_pct", "mean_pesq",
             "mean_stoi"],
            20,
        ),
        "targeted_hits.csv": (
            ["model", "k", "method", "trials", "targets_per_trial",
             "valid_copy_count", "mean_hits_out_of_10", "mean_pesq",
             "mean_stoi"],
            10,
        ),
        "bit_composition.csv": (
            ["model", "ones_in_coalition", "samples", "decoded_one_rate",
             "speaker_bootstrap_low", "speaker_bootstrap_high", "speakers"],
            45,
        ),
        "confidence_summary.csv": (
            ["model", "condition", "samples", "p05", "p25", "median",
             "p75", "p95"],
            15,
        ),
        "confidence_screening.csv": (
            ["model", "k", "speakers", "trials", "single_acceptance_pct",
             "average_rejection_pct", "target_success_before_pct",
             "target_success_after_pct"],
            5,
        ),
        "ideal_tracing_failure.csv": (
            ["bit_count", "k", "mean_pct", "lower_pct", "upper_pct"],
            8,
        ),
        "one_bit_examples.csv": (
            ["model", "trial_id", "flipped_copy_weight", "bit_score",
             "decoded_payload", "base_payload", "flipped_payload"],
            40,
        ),
    }
    for name, (expected_header, expected_rows) in expected.items():
        path = DATA / "summary" / name
        require(header(path) == expected_header, f"unexpected schema: {path}")
        require(len(read_csv(path)) == expected_rows,
                f"{path}: expected {expected_rows} rows")
    screening = read_csv(DATA / "summary" / "confidence_screening.csv")
    targeted = {
        row["model"]: 10.0 * float(row["mean_hits_out_of_10"])
        for row in read_csv(DATA / "summary" / "targeted_hits.csv")
        if int(row["k"]) == 8 and row["method"] == "Target-Bit Margin"
    }
    require({row["model"] for row in screening} == set(MODELS),
            "confidence-screening summary must cover all five systems")
    for row in screening:
        require(int(row["k"]) == 8 and int(row["speakers"]) == 100
                and int(row["trials"]) == 300,
                f"bad confidence-screening scope for {row['model']}")
        require(abs(float(row["target_success_before_pct"])
                    - targeted[row["model"]]) < 0.051,
                f"confidence-screening target rate mismatch for {row['model']}")
    print("PASS summaries: seven documented tables have fixed schemas")


def verify_supplementary() -> None:
    base = DATA / "supplementary"
    expected = {
        "quality/table2_quality_means.csv": (
            ["model", "n", "pesq_avg", "stoi_avg", "visqol_avg",
             "si_sdr_avg", "snr_avg"],
            5,
        ),
        "quality/summary_by_system_k.csv": (
            ["model", "k", "n", "pesq", "stoi", "visqol", "si_sdr", "snr",
             "mean_si_sdr_reproduction_error",
             "max_si_sdr_reproduction_error"],
            20,
        ),
        "alignment/summary_cross_system.csv": (
            ["condition", "system_count", "tf_pct", "attribution_margin",
             "pesq", "stoi", "si_sdr", "snr", "tf_delta_pp_vs_aligned",
             "pesq_delta_vs_aligned", "stoi_delta_vs_aligned",
             "si_sdr_delta_vs_aligned", "snr_delta_vs_aligned"],
            4,
        ),
        "alignment/summary_direction_averaged.csv": (
            ["model", "condition", "n", "tf_pct", "attribution_margin",
             "pesq", "stoi", "si_sdr", "snr"],
            20,
        ),
        "alignment/summary_by_system_shift.csv": (
            ["model", "shift_ms", "n", "tf_pct", "attribution_margin",
             "pesq", "stoi", "si_sdr", "snr"],
            35,
        ),
        "codec/summary_by_system_codec.csv": (
            ["model", "codec", "n", "tf_pct", "attribution_margin",
             "pesq", "stoi", "si_sdr", "snr"],
            15,
        ),
        "codec/summary_cross_system.csv": (
            ["codec", "system_count", "tf_pct", "attribution_margin",
             "pesq", "stoi", "si_sdr", "snr",
             "tf_pct_delta_vs_none", "pesq_delta_vs_none",
             "stoi_delta_vs_none", "si_sdr_delta_vs_none",
             "snr_delta_vs_none"],
            3,
        ),
        "registry_occupancy/registry_occupancy_by_system_k.csv": (
            ["model", "k", "bit_count", "n_trials", "native_escape_pct",
             "ideal_native_escape_pct", "occupancy_pct", "registry_size",
             "coalition_trace_pct", "registered_nonmember_pct",
             "unassigned_pct", "escape_pct", "ideal_coalition_trace_pct",
             "ideal_registered_nonmember_pct", "ideal_unassigned_pct",
             "ideal_escape_pct"],
            80,
        ),
        "registry_occupancy/registry_occupancy_all_k_average.csv": (
            ["occupancy_pct", "cells", "coalition_trace_pct",
             "registered_nonmember_pct", "unassigned_pct", "escape_pct",
             "ideal_coalition_trace_pct", "ideal_registered_nonmember_pct",
             "ideal_unassigned_pct", "ideal_escape_pct"],
            4,
        ),
        "registry_occupancy/registry_occupancy_k2_average.csv": (
            ["occupancy_pct", "systems", "coalition_trace_pct",
             "registered_nonmember_pct", "unassigned_pct", "escape_pct",
             "ideal_coalition_trace_pct", "ideal_registered_nonmember_pct",
             "ideal_unassigned_pct", "ideal_escape_pct"],
            4,
        ),
    }
    for relative, (expected_header, expected_rows) in expected.items():
        path = base / relative
        require(header(path) == expected_header, f"unexpected schema: {path}")
        require(len(read_csv(path)) == expected_rows,
                f"{path}: expected {expected_rows} rows")

    quality = read_csv(base / "quality" / "all_trials.csv")
    require(len(quality) == 6000,
            "supplementary quality data must contain 6000 rows")
    require(Counter(row["model"] for row in quality)
            == Counter({model: 1200 for model in MODELS}),
            "supplementary quality model counts mismatch")
    require("visqol" in header(base / "quality" / "all_trials.csv"),
            "supplementary quality data must include ViSQOL")
    require(all(1.0 <= float(row["visqol"]) <= 5.0 for row in quality),
            "supplementary ViSQOL scores must be in [1, 5]")

    alignment = read_csv(base / "alignment" / "all_trials.csv")
    require(len(alignment) == 10500,
            "supplementary alignment data must contain 10500 rows")
    require(Counter(int(row["shift_ms"]) for row in alignment)
            == Counter({shift: 1500 for shift in (0, -10, 10, -20, 20, -50, 50)}),
            "supplementary alignment shift counts mismatch")

    codec = read_csv(base / "codec" / "all_trials.csv")
    require(len(codec) == 4500,
            "supplementary codec data must contain 4500 rows")
    require(Counter(row["model"] for row in codec)
            == Counter({model: 900 for model in MODELS}),
            "supplementary codec model counts mismatch")
    require(Counter(row["codec"] for row in codec)
            == Counter({condition: 1500 for condition in
                        ("none", "mp3_128k", "opus_64k")}),
            "supplementary codec condition counts mismatch")
    for model in MODELS:
        model_rows = [row for row in codec if row["model"] == model]
        require(len({row["source_path"] for row in model_rows}) == 300,
                f"supplementary codec {model}: expected 300 recordings")
        require({int(row["k"]) for row in model_rows} == {5},
                f"supplementary codec {model}: expected K=5")

    occupancy = read_csv(
        base / "registry_occupancy" /
        "registry_occupancy_by_system_k.csv")
    for row in occupancy:
        require(abs(float(row["registered_nonmember_pct"])
                    + float(row["unassigned_pct"])
                    - float(row["escape_pct"])) < 1e-8,
                "registry outcomes do not sum to escape rate")
    print("PASS supplementary: quality, alignment, codec, and registry analyses")


def read_demo_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        require(handle.getnchannels() == 1, f"demo must be mono: {path}")
        require(handle.getsampwidth() == 2, f"demo must be PCM-16: {path}")
        require(handle.getframerate() == 16000,
                f"demo must be 16 kHz: {path}")
        require(handle.getnframes() == 160000,
                f"demo must contain ten seconds: {path}")
        return np.frombuffer(handle.readframes(160000), dtype="<i2").astype(np.int32)


def verify_demos() -> None:
    demos = ROOT / "demos"
    source = read_demo_pcm16(demos / "source_reference.wav")
    require(len(source) == 160000, "invalid demo source")
    page = (demos / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    public_url = (
        "https://silwan5z.github.io/"
        "neural-audio-watermark-collusion-attack/index.html")
    require(readme.count(public_url) == 1
            and readme.count(
                "silwan5z.github.io/neural-audio-watermark-collusion-attack")
            == 1,
            "README must contain exactly one explicit public demo entry")

    models = {"audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"}
    expected_counts = {"coalition_size": 4, "offset": 7, "codec": 3}
    metadata = read_csv(demos / "conditions" / "metadata.csv")
    require(len(metadata) == 70,
            "complete demo must contain 70 model-condition examples")
    require({row["system"] for row in metadata} == models,
            "demo conditions must cover all five systems")

    alignment_rows = read_csv(
        DATA / "supplementary" / "alignment" / "all_trials.csv")
    codec_rows = read_csv(
        DATA / "supplementary" / "codec" / "all_trials.csv")
    decoded_by_model = {
        model: json.loads((
            demos / "decoded" / f"{model}.json").read_text(encoding="utf-8"))
        for model in models
    }
    for model in models:
        for family, count in expected_counts.items():
            require(sum(row["system"] == model and row["family"] == family
                        for row in metadata) == count,
                    f"incomplete {family} demo coverage: {model}")

    expected_audio: set[str] = {
        "source_reference.wav", "source_reference.mp3",
    }
    for row in metadata:
        model = row["system"]
        family = row["family"]
        condition = row["condition"]
        k = int(row["k"])
        coalition = [int(value) for value in
                     json.loads(row["coalition_payloads"])]
        decoded_record = decoded_by_model[model]
        require(coalition == decoded_record["coalitions"][str(k)],
                f"demo coalition mismatch: {model}/{family}/{condition}")

        if family == "coalition_size":
            decoded = int(decoded_record[family][str(k)]["decoded_payload"])
            if k < 8:
                evidence = next(
                    item for item in read_csv(
                        DATA / "average" / f"k{k}" / f"{model}.csv")
                    if int(item["trial_id"]) == 150)
            else:
                evidence = json.loads((
                    DATA / "average" / "k8" / model /
                    "trial_150.json").read_text(encoding="utf-8"))
                require(decoded == int(evidence["decoded_payload"]),
                        f"K=8 decoded payload mismatch: {model}")
        elif family == "offset":
            decoded = int(decoded_record[family][condition]["decoded_payload"])
            evidence = next(
                item for item in alignment_rows
                if item["model"] == model and int(item["trial_id"]) == 150
                and int(item["shift_ms"]) == int(condition))
        elif family == "codec":
            decoded = int(decoded_record[family][condition]["decoded_payload"])
            evidence = next(
                item for item in codec_rows
                if item["model"] == model and int(item["trial_id"]) == 150
                and item["codec"] == condition)
        else:
            raise RuntimeError(f"unknown demo family: {family}")
        require(int(decoded not in coalition) == int(evidence["escaped"]),
                f"demo outcome mismatch: {model}/{family}/{condition}")
        require_close(float(row["pesq"]), float(evidence["pesq"]),
                      f"demo PESQ mismatch: {model}/{family}/{condition}",
                      tolerance=1e-5)
        require_close(float(row["stoi"]), float(evidence["stoi"]),
                      f"demo STOI mismatch: {model}/{family}/{condition}",
                      tolerance=1e-5)
        if not (family == "coalition_size" and k == 8):
            require_close(float(row["si_sdr_db"]),
                          float(evidence["si_sdr"]),
                          f"demo SI-SDR mismatch: {model}/{family}/{condition}",
                          tolerance=1e-5)
        require(np.isfinite(float(row["si_sdr_db"])),
                f"demo SI-SDR is not finite: {model}/{family}/{condition}")

        wav_relative = row["audio_path"]
        mp3_relative = str(Path(wav_relative).with_suffix(".mp3"))
        expected_audio.update({wav_relative, mp3_relative})
        read_demo_pcm16(demos / wav_relative)
        mp3 = demos / mp3_relative
        require(mp3.is_file() and mp3.stat().st_size > 100000
                and mp3.read_bytes()[:3] == b"ID3",
                f"invalid browser MP3: {mp3_relative}")
        require((ROOT / row["evidence_path"]).is_file(),
                f"missing evidence for demo condition: {row}")

    observed_condition_audio = {
        path.relative_to(demos).as_posix()
        for path in (demos / "conditions").glob("*/*/*")
        if path.suffix in {".wav", ".mp3"}
    }
    require(observed_condition_audio == expected_audio - {
                "source_reference.wav", "source_reference.mp3"},
            "demo contains missing or unreferenced condition audio")

    bundle_text = (demos / "demo-data.js").read_text(encoding="utf-8")
    prefix = "window.DEMO_DATA = "
    require(bundle_text.startswith(prefix) and bundle_text.endswith(";\n"),
            "invalid demo browser data bundle")
    bundle = json.loads(bundle_text[len(prefix):-2])
    require(len(bundle["systems"]) == 5,
            "browser bundle must expose five systems")
    bundle_conditions = sum(
        len(system["coalitionSize"]) + len(system["offsets"])
        + len(system["codecs"]) for system in bundle["systems"])
    require(bundle_conditions == 70,
            "browser bundle must expose all 70 conditions")
    require(sum(len(system["copies"]) for system in bundle["systems"]) == 20,
            "browser bundle must expose four source/copy rows per system")
    bundle_audio = {
        f"{item['audio']}{suffix}"
        for system in bundle["systems"]
        for family in ("coalitionSize", "offsets", "codecs")
        for item in system[family]
        for suffix in (".wav", ".mp3")
    }
    require(bundle_audio == observed_condition_audio,
            "browser bundle does not match released demo audio")

    copy_metadata = {row["system"]: row for row in read_csv(
        demos / "metadata.csv")}
    expected_copy_audio: set[str] = set()
    for model in models:
        row = copy_metadata[model]
        for stem in (f"member_{row['payload_a']}",
                     f"member_{row['payload_b']}", "uniform_average"):
            for suffix in (".wav", ".mp3"):
                relative = f"{model}/{stem}{suffix}"
                expected_copy_audio.add(relative)
                path = demos / relative
                require(path.is_file() and path.stat().st_size > 100000,
                        f"missing watermarked-copy demo audio: {relative}")
                if suffix == ".wav":
                    read_demo_pcm16(path)
                else:
                    require(path.read_bytes()[:3] == b"ID3",
                            f"invalid watermarked-copy MP3: {relative}")
    bundle_copy_audio = {
        f"{item['audio']}{suffix}"
        for system in bundle["systems"]
        for item in system["copies"]
        if item["audio"] != "source_reference"
        for suffix in (".wav", ".mp3")
    }
    require(bundle_copy_audio == expected_copy_audio,
            "browser bundle does not match watermarked-copy audio")

    bundle_metrics = {
        (system["id"], family, item["label"]): item["metrics"]
        for system in bundle["systems"]
        for family in ("coalitionSize", "offsets", "codecs")
        for item in system[family]
    }
    require(len(bundle_metrics) == 70 and all(
                set(value) == {"pesq", "stoi", "siSdr"}
                for value in bundle_metrics.values()),
            "browser bundle must expose all three metrics for every condition")
    metadata_by_audio = {
        str(Path(row["audio_path"]).with_suffix("")): row for row in metadata
    }
    for system in bundle["systems"]:
        for family in ("coalitionSize", "offsets", "codecs"):
            for item in system[family]:
                row = metadata_by_audio[item["audio"]]
                require_close(float(item["metrics"]["pesq"]),
                              float(row["pesq"]),
                              f"browser PESQ mismatch: {item['audio']}")
                require_close(float(item["metrics"]["stoi"]),
                              float(row["stoi"]),
                              f"browser STOI mismatch: {item['audio']}")
                require_close(float(item["metrics"]["siSdr"]),
                              float(row["si_sdr_db"]),
                              f"browser SI-SDR mismatch: {item['audio']}")

    require(page.count('class="site-nav"') == 1,
            "demo must contain one primary navigation bar")
    require(page.count('class="tab-button"') == 5,
            "demo must expose five switchable primary views")
    require(page.count("data-view-panel=") == 5,
            "demo must contain five switchable view panels")
    require(page.count('<nav class="system-picker" data-system-picker=') == 4,
            "each comparison view must have a system selector")
    require('src="demo-data.js"' in page,
            "demo page does not load the complete data bundle")
    require(page.count(
                '<audio controls preload="none" src="source_reference.mp3" '
                'data-fallback="source_reference.wav" '
                'aria-label="Clean source reference">') == 1,
            "overview must contain one clean reference player")
    require('system[category[view]].map' in page,
            "comparison players must be rendered by the switchable view")
    require('controls preload="none" src="${escape(item.audio)}.mp3" '
            'data-fallback="${escape(item.audio)}.wav"' in page,
            "dynamic players must load MP3 on demand with WAV fallback")
    require('player.addEventListener("error"' in page
            and 'player.src=player.dataset.fallback' in page,
            "demo must automatically activate the WAV fallback")
    require('<span>PESQ</span>' in page and '<span>STOI</span>' in page
            and '<span>SI-SDR</span>' in page,
            "demo must label all three quality metrics")
    require("Suggested listening route" not in page,
            "demo must not contain the removed listening-route card")
    require(not any(term in page for term in (
                "tracing failure", "Tracing failure", "300-trial", "300 trials")),
            "demo page must not duplicate paper result statistics")
    print("PASS demos: 5 systems, 70 conditions, 20 source/copy rows, 172 audio files")


def json_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_keys(child)


def verify_names() -> None:
    banned_path_parts = {
        "main", "final", "selection_cache", "target_selection", "onebit",
        "mixture_path", "average_k8",
    }
    public_name = re.compile(r"^[a-z0-9][a-z0-9_.]*$")
    field_name = re.compile(r"^[a-z][a-z0-9_]*$")
    obsolete_fields = {
        "target_score", "target_rule", "targets_considered", "payload_count",
        "valid_weights", "decoder_margin", "score_source", "source_ones",
        "mix_weight", "lambda_index", "local_trial",
    }
    for directory in ACTIVE_DATA:
        for path in (DATA / directory).rglob("*"):
            relative = path.relative_to(DATA)
            parts = set(relative.parts)
            require(not parts & banned_path_parts,
                    f"obsolete path name: {path.relative_to(ROOT)}")
            require(all(public_name.fullmatch(part) for part in relative.parts),
                    f"nonstandard path name: {path.relative_to(ROOT)}")
            if path.suffix == ".csv":
                fields = header(path)
                require(all(field_name.fullmatch(name) for name in fields),
                        f"nonstandard CSV field in {path.relative_to(ROOT)}")
                require(not set(fields) & obsolete_fields,
                        f"obsolete CSV field in {path.relative_to(ROOT)}")
            elif path.suffix == ".json":
                keys = set(json_keys(read_json(path)))
                require(all(field_name.fullmatch(name) for name in keys),
                        f"nonstandard JSON key in {path.relative_to(ROOT)}")
                require(not keys & obsolete_fields,
                        f"obsolete JSON key in {path.relative_to(ROOT)}")

    pattern = re.compile(
        r"\bPM\b|\bMRC\b|successful_mrc|uniform_collusion|shared4|"
        r"cache_clip_indexed_v20|data/main|data/k8|data/average_k8|"
        r"source_correct|source_exact|exact_hits_out_of_10|exact_match|"
        r"nearest_member|identity_|onebit|mixture_path|target_selection"
    )
    paths = [ROOT / "README.md", ROOT / "data" / "README.md",
             ROOT / "dataset" / "README.md", ROOT / "scripts" / "README.md"]
    paths += sorted((ROOT / "scripts").rglob("*.py"))
    paths += sorted((ROOT / "src").rglob("*.py"))
    for path in paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        match = pattern.search(path.read_text(encoding="utf-8"))
        if match is not None:
            raise RuntimeError(f"obsolete name {match.group()!r} in {path}")
    print("PASS naming: paths, schemas, and interfaces use the public terminology")


def verify_manifest() -> None:
    rows = read_csv(DATA / "manifest.csv")
    listed = [row["path"] for row in rows]
    require(len(listed) == len(set(listed)), "duplicate paths in data/manifest.csv")
    current = {
        path.relative_to(ROOT).as_posix()
        for directory in ACTIVE_DATA
        for path in (DATA / directory).rglob("*")
        if path.is_file()
    }
    require(set(listed) == current,
            "data/manifest.csv does not match the released data tree")
    for row in rows:
        path = ROOT / row["path"]
        require(path.is_file(), f"missing manifest file: {path}")
        require(path.stat().st_size == int(row["bytes"]), f"size mismatch: {path}")
        require(sha256(path) == row["sha256"], f"checksum mismatch: {path}")
        if path.suffix == ".csv":
            with path.open("rb") as handle:
                count = max(0, sum(1 for _ in handle) - 1)
            require(count == int(row["rows"]), f"row-count mismatch: {path}")
    print(f"PASS manifest: {len(rows)} files and checksums")


def main() -> None:
    verify_average()
    coalitions = verify_coalitions()
    verify_targeted(coalitions)
    verify_one_bit()
    verify_confidence()
    verify_summaries()
    verify_supplementary()
    verify_demos()
    verify_names()
    verify_manifest()
    print("ALL RELEASE CHECKS PASSED")


if __name__ == "__main__":
    main()

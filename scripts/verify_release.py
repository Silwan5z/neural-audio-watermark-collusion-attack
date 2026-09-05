#!/usr/bin/env python3
"""Verify every released record without running model inference."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
SHARED_MODELS = ("audioseal", "wavmark", "voicemark", "wmcodec")
METHODS = ("payload_match", "bit_margin")
METHOD_LABELS = {"Payload Match": "payload_match", "Bit Margin": "bit_margin"}
ACTIVE_DATA = (
    "average", "coalitions", "targets", "targeted", "one_bit", "confidence",
    "summary",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
            target = float(expected[(model, k)]["tracing_failure_pct"])
            require(abs(value - target) < 1e-5,
                    f"{path}: TF {value} != summary {target}")

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
        value = 100.0 * sum(int(row["escaped"]) for row in records) / 300
        target = float(expected[(model, 8)]["tracing_failure_pct"])
        require(abs(value - target) < 1e-5,
                f"{model} k=8: TF {value} != summary {target}")
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
    print("PASS coalitions: shared coalitions and ten targets per method")
    return coalitions


def verify_targeted(coalitions: dict[tuple[int, int], list[int]]) -> None:
    summary = read_csv(DATA / "summary" / "targeted_hits.csv")
    expected = {
        (row["model"], int(row["k"]), METHOD_LABELS[row["method"]]):
        float(row["mean_hits_out_of_10"])
        for row in summary
    }
    require(len(expected) == 20, "targeted summary must contain 20 rows")
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
    rules = {
        "payload_match": "nearest_nonmember_payloads",
        "bit_margin": "largest_minimum_bit_margin",
    }
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
                        require(int(row["target_hit"]) == int(decoded == target),
                                f"{path}: target-hit flag mismatch")
                        require(int(row["escaped"]) == int(decoded not in coalition),
                                f"{path}: escape flag mismatch")
                value = sum(int(row["target_hit"]) for row in rows) / 300.0
                target = expected[(model, k, method)]
                require(abs(value - target) < 1e-5,
                        f"{path}: hits {value} != summary {target}")
    print("PASS targeted: 20 files, 300 trials x ten selected targets")


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
            20,
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
    print("PASS summaries: six documented tables have fixed schemas")


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
    verify_names()
    verify_manifest()
    print("ALL RELEASE CHECKS PASSED")


if __name__ == "__main__":
    main()

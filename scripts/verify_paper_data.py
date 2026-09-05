#!/usr/bin/env python3
"""Read-only integrity checks for every result used by the manuscript."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_main() -> None:
    expected = read_csv(DATA / "summary" / "main_results.csv")
    expected_by_key = {(row["model"], int(row["K"])): row for row in expected}
    for model in MODELS:
        for k in (2, 3, 5):
            rows = read_csv(DATA / "main" / f"attack_{model}_K{k}.csv")
            assert len(rows) == 300
            assert len({row["source_path"] for row in rows}) == 300
            assert len({row["spk"] for row in rows}) == 100
            assert {int(row["source_exact_count"]) for row in rows} == {k}
            tf = 100.0 * sum(int(row["ASR"]) for row in rows) / len(rows)
            target = float(expected_by_key[(model, k)]["tracing_failure_pct"])
            assert abs(tf - target) < 1e-5, (model, k, tf, target)

    for model in MODELS:
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((DATA / "k8" / "raw" / model).glob("trial_*.json"))]
        assert len(records) == 300
        assert len({row["source_path"] for row in records}) == 300
        assert len({row["speaker"] for row in records}) == 100
        assert {int(row["source_exact_count"]) for row in records} == {8}
        tf = 100.0 * sum(int(row["tracing_failure"]) for row in records) / len(records)
        target = float(expected_by_key[(model, 8)]["tracing_failure_pct"])
        assert abs(tf - target) < 1e-5, (model, 8, tf, target)
    print("PASS main results: 20 system/K cells, 300 source-correct trials each")


def verify_targeted() -> None:
    expected = read_csv(DATA / "summary" / "targeted_hits.csv")
    expected_by_key = {
        (row["model"], int(row["K"]), row["method"].lower()):
        float(row["mean_exact_hits_out_of_10"])
        for row in expected
    }
    for model in MODELS:
        for k in (5, 8):
            for method in ("pm", "mrc"):
                rows = read_csv(DATA / "targeted" / f"{model}_k{k}_{method}.csv")
                assert len(rows) == 3000
                counts = Counter(int(row["trial_id"]) for row in rows)
                assert counts == Counter({trial: 10 for trial in range(300)})
                hits = sum(int(row["target_top1"]) for row in rows) / 300.0
                target = expected_by_key[(model, k, method)]
                assert abs(hits - target) < 1e-5, (model, k, method, hits, target)
    print("PASS targeted results: 20 files, 300 trials × 10 selected targets")


def verify_one_bit() -> None:
    for model in ("audioseal", "voicemark"):
        rows = read_csv(DATA / "one_bit" / f"{model}_pairs.csv")
        assert len(rows) == 300
        assert all(int(row["base_payload_correct"]) == 1 for row in rows)
        assert all(int(row["flipped_payload_correct"]) == 1 for row in rows)
    summary = read_csv(DATA / "one_bit" / "path_summary.csv")
    assert len(summary) == 600
    direct = {
        model: sum(int(row["direct"]) for row in summary if row["model"] == model)
        for model in ("audioseal", "voicemark")
    }
    assert direct == {"audioseal": 299, "voicemark": 113}, direct
    print("PASS one-bit paths: AudioSeal 299 direct; VoiceMark 113 direct / 187 other")


def verify_confidence() -> None:
    expected = {
        "audioseal": {"benign_copy": 300, "uniform_collusion": 300, "successful_mrc": 294},
        "wavmark": {"benign_copy": 300, "uniform_collusion": 300, "successful_mrc": 300},
        "timbrewm": {"benign_copy": 300, "uniform_collusion": 300, "successful_mrc": 300},
        "voicemark": {"benign_copy": 300, "uniform_collusion": 300, "successful_mrc": 19},
        "wmcodec": {"benign_copy": 300, "uniform_collusion": 300, "successful_mrc": 34},
    }
    for model, target in expected.items():
        rows = read_csv(DATA / "confidence" / f"min_bit_confidence_k8_{model}.csv")
        assert Counter(row["condition"] for row in rows) == target
    print("PASS confidence data: all Fig. 4 condition counts match the manuscript")


def verify_manifest() -> None:
    rows = read_csv(DATA / "MANIFEST.csv")
    for row in rows:
        path = ROOT / row["path"]
        assert path.is_file(), path
        assert path.stat().st_size == int(row["bytes"]), path
        assert sha256(path) == row["sha256"], path
    print(f"PASS manifest: {len(rows)} files and checksums")


def main() -> None:
    verify_main()
    verify_targeted()
    verify_one_bit()
    verify_confidence()
    verify_manifest()
    print("ALL PAPER DATA CHECKS PASSED")


if __name__ == "__main__":
    main()

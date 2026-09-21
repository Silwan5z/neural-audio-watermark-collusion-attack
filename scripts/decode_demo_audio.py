#!/usr/bin/env python3
"""Decode every native-rate signal represented in the web demo."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_demo_audio import MODELS, native_conditions  # noqa: E402
from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE, bits_to_int, detect_many_native,
)
from registry import full_registry_bits  # noqa: E402


def decode(model: str, signals: list[np.ndarray]) -> list[int]:
    results = detect_many_native(
        model, signals, NATIVE_SAMPLE_RATE[model], full_registry_bits(model))
    payloads = []
    for index, (scores, _, hard) in enumerate(results):
        if hard is None or not np.isfinite(scores).all() or not np.any(scores):
            raise RuntimeError(f"{model}: unusable demo decode at index {index}")
        payload = bits_to_int(np.asarray(hard, dtype=np.int8))
        if payload != int(np.argmax(scores)):
            raise RuntimeError(
                f"{model}: hard bits and registry top-1 disagree at index {index}")
        payloads.append(payload)
    return payloads


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "demos" / "decoded")
    args = parser.parse_args()

    model = args.model
    coalitions, conditions = native_conditions(model)
    labels: list[tuple[str, str | int]] = []
    signals: list[np.ndarray] = []
    for family in ("coalition_size", "offset", "codec"):
        for condition, waveform in conditions[family].items():
            labels.append((family, condition))
            signals.append(waveform)
    decoded = decode(model, signals)

    record: dict = {
        "model": model,
        "trial_id": 150,
        "coalitions": {str(k): values for k, values in coalitions.items()},
        "coalition_size": {}, "offset": {}, "codec": {},
    }
    for (family, condition), payload in zip(labels, decoded):
        record[family][str(condition)] = {"decoded_payload": int(payload)}
    record["cross_system"] = {
        "coalition_payloads": coalitions[2],
        "decoded_payload": record["coalition_size"]["2"]["decoded_payload"],
    }

    output = args.output_dir / f"{model}.json"
    atomic_json(output, record)
    print(json.dumps({
        "model": model, "conditions": len(signals), "output": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()

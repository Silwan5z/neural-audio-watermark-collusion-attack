#!/usr/bin/env python3
"""Decode the exact native signals exported by the interactive demo."""
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

from export_demo_audio import (  # noqa: E402
    AUDIOSEAL_COALITIONS, CODECS, EXAMPLES, OFFSETS_MS,
    load_audioseal_members,
)
from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE, bits_to_int, detect_many_native, get_or_embed_native,
)
from registry import full_registry_bits  # noqa: E402
from run_alignment_stress_test import shift_fixed_length  # noqa: E402
from run_codec_stress_test import independently_code  # noqa: E402


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
    parser.add_argument("--model", required=True, choices=tuple(EXAMPLES))
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "demos" / "decoded")
    args = parser.parse_args()

    model = args.model
    _, payload_a, payload_b, _ = EXAMPLES[model]
    members = [
        get_or_embed_native(model, "english:103", payload, 0)[0]
        for payload in (payload_a, payload_b)
    ]
    n = min(map(len, members))
    mixture = np.mean(
        np.stack([member[:n] for member in members]), axis=0,
        dtype=np.float64).astype(np.float32)
    record: dict = {
        "model": model,
        "trial_id": 150,
        "cross_system": {
            "coalition_payloads": [payload_a, payload_b],
            "decoded_payload": decode(model, [mixture])[0],
        },
    }

    if model == "audioseal":
        size_labels = []
        size_signals = []
        for k, coalition in AUDIOSEAL_COALITIONS.items():
            inputs = load_audioseal_members(args.cache_root, coalition)
            size_labels.append(str(k))
            size_signals.append(np.mean(
                np.stack(inputs), axis=0,
                dtype=np.float64).astype(np.float32))
        size_decoded = decode(model, size_signals)
        record["coalition_size"] = {
            label: {
                "coalition_payloads": AUDIOSEAL_COALITIONS[int(label)],
                "decoded_payload": payload,
            }
            for label, payload in zip(size_labels, size_decoded)
        }

        k5_members = load_audioseal_members(
            args.cache_root, AUDIOSEAL_COALITIONS[5])
        offset_labels = []
        offset_signals = []
        for offset_ms in OFFSETS_MS:
            inputs = list(k5_members)
            if offset_ms:
                inputs[0] = shift_fixed_length(
                    inputs[0], int(round(16000 * offset_ms / 1000)))
            offset_labels.append(str(offset_ms))
            offset_signals.append(np.mean(
                np.stack(inputs), axis=0,
                dtype=np.float64).astype(np.float32))
        record["offset"] = {
            label: {"decoded_payload": payload}
            for label, payload in zip(offset_labels, decode(model, offset_signals))
        }

        codec_signals = []
        for codec in CODECS:
            transformed = independently_code(k5_members, 16000, codec)
            codec_signals.append(np.mean(
                np.stack(transformed), axis=0,
                dtype=np.float64).astype(np.float32))
        record["codec"] = {
            label: {"decoded_payload": payload}
            for label, payload in zip(CODECS, decode(model, codec_signals))
        }

    output = args.output_dir / f"{model}.json"
    atomic_json(output, record)
    print(json.dumps({"output": str(output), **record}, sort_keys=True))


if __name__ == "__main__":
    main()

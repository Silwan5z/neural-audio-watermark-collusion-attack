#!/usr/bin/env python3
"""Current-protocol temporal-misalignment test for K=5 averaging."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE,
    detect_many_native,
    get_or_embed_native,
)
from registry import full_registry_bits, source_record, trial_schedule  # noqa: E402
from watermarks import pesq_wb, resample_to, si_sdr, stoi  # noqa: E402

K = 5
SHIFTS_MS = (0, -10, 10, -20, 20, -50, 50)
FIELDS = [
    "model", "k", "trial_id", "speaker", "clip_index", "source_path",
    "coalition_payloads", "shift_ms", "shifted_colluder_index",
    "escaped", "attribution_margin", "pesq", "stoi", "si_sdr", "snr",
]


def snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    n = min(len(reference), len(degraded))
    reference = np.asarray(reference[:n], dtype=np.float64)
    degraded = np.asarray(degraded[:n], dtype=np.float64)
    error = degraded - reference
    return float(10.0 * np.log10(
        (np.sum(reference * reference) + 1e-12) /
        (np.sum(error * error) + 1e-12)))


def shift_fixed_length(waveform: np.ndarray, samples: int) -> np.ndarray:
    if samples == 0:
        return waveform.copy()
    if abs(samples) >= len(waveform):
        return np.zeros_like(waveform)
    if samples > 0:
        return np.concatenate([
            np.zeros(samples, dtype=waveform.dtype), waveform[:-samples]
        ])
    advance = -samples
    return np.concatenate([
        waveform[advance:], np.zeros(advance, dtype=waveform.dtype)
    ])


def attack_metrics(scores: np.ndarray,
                   coalition: list[int]) -> tuple[int, float]:
    coalition_array = np.asarray(coalition, dtype=np.int64)
    top = int(np.argmax(scores))
    coalition_best = float(np.max(scores[coalition_array]))
    mask = np.ones(len(scores), dtype=bool)
    mask[coalition_array] = False
    innocent_best = float(np.max(scores[mask]))
    return int(top not in set(coalition)), innocent_best - coalition_best


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def coalitions_for(model: str) -> dict[int, list[int]]:
    path = ROOT / "results" / "quality" / "all_trials.csv"
    frame = pd.read_csv(path)
    frame = frame[(frame["model"] == model) & (frame["k"] == K)]
    if len(frame) != 300:
        raise RuntimeError(f"{model}: expected 300 validated K=5 coalitions")
    return {
        int(row.trial_id): [int(value) for value in
                            json.loads(row.coalition_payloads)]
        for row in frame.itertuples()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(
        "audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"))
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--min-trial-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "alignment_stress_test")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require 0 <= shard-id < num-shards")

    model = args.model
    sample_rate = NATIVE_SAMPLE_RATE[model]
    registry = full_registry_bits(model)
    coalitions = coalitions_for(model)
    suffix = ("" if args.num_shards == 1 else
              f".shard{args.shard_id}of{args.num_shards}")
    out_path = args.output_dir / f"{model}{suffix}.csv"
    rows: list[dict] = []
    completed: set[int] = set()
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[int, int] = {}
        for row in rows:
            trial_id = int(row["trial_id"])
            grouped[trial_id] = grouped.get(trial_id, 0) + 1
        completed = {trial_id for trial_id, count in grouped.items()
                     if count == len(SHIFTS_MS)}
        rows = [row for row in rows if int(row["trial_id"]) in completed]

    started = time.time()
    for trial_id, (speaker, clip_slot) in enumerate(
            trial_schedule(args.n_trials)):
        if trial_id % args.num_shards != args.shard_id:
            continue
        if trial_id < args.min_trial_id:
            continue
        if trial_id in completed:
            continue
        coalition = coalitions[trial_id]
        members = [
            get_or_embed_native(model, speaker, payload, clip_slot)[0]
            for payload in coalition
        ]
        n = min(map(len, members))
        members = [np.asarray(member[:n], dtype=np.float32)
                   for member in members]
        reference = members[0]
        shifted_index = trial_id % K

        signals: list[np.ndarray] = []
        for shift_ms in SHIFTS_MS:
            condition = list(members)
            if shift_ms:
                samples = int(round(sample_rate * shift_ms / 1000.0))
                condition[shifted_index] = shift_fixed_length(
                    condition[shifted_index], samples)
            signals.append(np.mean(np.stack(condition), axis=0,
                                   dtype=np.float64).astype(np.float32))

        decoded = detect_many_native(model, signals, sample_rate, registry)
        source = source_record(speaker, clip_slot)
        reference_16 = (reference if sample_rate == 16000 else
                        resample_to(reference, sample_rate, 16000))
        for shift_ms, signal, (score, _, _) in zip(
                SHIFTS_MS, signals, decoded):
            escaped, margin = attack_metrics(score, coalition)
            signal_16 = (signal if sample_rate == 16000 else
                         resample_to(signal, sample_rate, 16000))
            rows.append({
                "model": model,
                "k": K,
                "trial_id": trial_id,
                "speaker": speaker,
                "clip_index": source["clip_index"],
                "source_path": source["path"],
                "coalition_payloads": json.dumps(coalition),
                "shift_ms": shift_ms,
                "shifted_colluder_index": (-1 if shift_ms == 0 else
                                             shifted_index),
                "escaped": escaped,
                "attribution_margin": f"{margin:.8f}",
                "pesq": f"{pesq_wb(reference_16, signal_16):.6f}",
                "stoi": f"{stoi(reference_16, signal_16):.6f}",
                "si_sdr": f"{si_sdr(reference_16, signal_16):.6f}",
                "snr": f"{snr_db(reference_16, signal_16):.6f}",
            })
        completed.add(trial_id)
        if len(completed) % 10 == 0:
            rows.sort(key=lambda row: (int(row["trial_id"]),
                                       int(row["shift_ms"])))
            atomic_csv(out_path, rows)
            assigned = sum(
                index >= args.min_trial_id and
                index % args.num_shards == args.shard_id
                for index in range(args.n_trials)
            )
            print(f"{model}: {len(completed)}/{assigned} "
                  f"({time.time() - started:.0f}s)", flush=True)

    rows.sort(key=lambda row: (int(row["trial_id"]),
                               int(row["shift_ms"])))
    atomic_csv(out_path, rows)
    print(json.dumps({"model": model, "trials": len(completed),
                      "rows": len(rows), "output": str(out_path),
                      "elapsed_seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()

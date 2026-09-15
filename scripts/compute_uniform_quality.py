#!/usr/bin/env python3
"""Reconstruct released uniform mixtures and add waveform quality metrics.

The script follows the released experiment protocol: the first valid
personalized copy is the reference, all coalition members are averaged with
equal weights, and native-rate signals are resampled to 16 kHz before metric
evaluation.  It writes restart-safe checkpoints under ``results/quality``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from native_audio import NATIVE_SAMPLE_RATE, get_or_embed_native  # noqa: E402
from registry import (  # noqa: E402
    coalition_seed,
    full_registry_bits,
    source_record,
    trial_schedule,
)
from run_average import valid_coalition  # noqa: E402
from watermarks import resample_to, si_sdr  # noqa: E402


FIELDS = [
    "model", "k", "trial_id", "speaker", "clip_index", "source_path",
    "coalition_payloads", "reference_payload", "published_pesq",
    "published_stoi", "published_si_sdr", "recomputed_si_sdr", "snr",
    "si_sdr_abs_error",
]


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    n = min(len(reference), len(degraded))
    reference = np.asarray(reference[:n], dtype=np.float64)
    degraded = np.asarray(degraded[:n], dtype=np.float64)
    signal_power = np.sum(reference * reference)
    error = degraded - reference
    noise_power = np.sum(error * error)
    return float(10.0 * np.log10((signal_power + 1e-12) /
                                 (noise_power + 1e-12)))


def load_rows(path: Path) -> tuple[list[dict], set[tuple[int, int]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete = {(int(row["k"]), int(row["trial_id"])) for row in rows}
    return rows, complete


def published_rows(model: str, k: int) -> dict[int, dict]:
    path = ROOT / "data" / "average" / f"k{k}" / f"{model}.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["trial_id"]): row for row in csv.DictReader(handle)}


def k8_record(model: str, trial_id: int) -> dict:
    path = (ROOT / "data" / "average" / "k8" / model /
            f"trial_{trial_id:03d}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct(model: str, k: int, trial_id: int, speaker: str,
                clip_slot: int, payloads_tested: int | None = None
                ) -> tuple[list[int], list[np.ndarray]]:
    sample_rate = NATIVE_SAMPLE_RATE[model]
    if k == 8:
        record = k8_record(model, trial_id)
        payloads = [int(value) for value in record["coalition_payloads"]]
        waveforms = [
            get_or_embed_native(model, speaker, payload, clip_slot)[0]
            for payload in payloads
        ]
        return payloads, waveforms

    rng = np.random.default_rng(coalition_seed(speaker, k, clip_slot))
    # The released audit records when the first k distinct candidates were all
    # source-valid.  In that case the coalition can be reconstructed exactly
    # without repeating an expensive decoder pass.
    if payloads_tested == k:
        payloads: list[int] = []
        limit = len(full_registry_bits(model))
        while len(payloads) < k:
            payload = int(rng.integers(0, limit))
            if payload not in payloads:
                payloads.append(payload)
        waveforms = [
            get_or_embed_native(model, speaker, payload, clip_slot)[0]
            for payload in payloads
        ]
        return payloads, waveforms
    return valid_coalition(
        model, speaker, clip_slot, k, rng, full_registry_bits(model),
        sample_rate,
    )[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=(
        "audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"))
    parser.add_argument("--ks", type=int, nargs="+", default=(2, 3, 5, 8),
                        choices=(2, 3, 5, 8))
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "quality")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require 0 <= shard-id < num-shards")

    model = args.model
    suffix = ("" if args.num_shards == 1 else
              f".shard{args.shard_id}of{args.num_shards}")
    out_path = args.output_dir / f"{model}{suffix}.csv"
    rows, complete = load_rows(out_path)
    schedule = trial_schedule(args.n_trials)
    sample_rate = NATIVE_SAMPLE_RATE[model]

    for k in args.ks:
        published = published_rows(model, k)
        for trial_id, (speaker, clip_slot) in enumerate(schedule):
            if trial_id % args.num_shards != args.shard_id:
                continue
            key = (k, trial_id)
            if key in complete:
                continue
            attempts = (None if k == 8 else
                        int(published[trial_id]["payloads_tested"]))
            payloads, waveforms = reconstruct(
                model, k, trial_id, speaker, clip_slot, attempts)
            n = min(map(len, waveforms))
            waveforms = [waveform[:n] for waveform in waveforms]
            reference = waveforms[0]
            mixture = np.mean(np.stack(waveforms), axis=0,
                              dtype=np.float64).astype(np.float32)
            reference_16 = (reference if sample_rate == 16000 else
                            resample_to(reference, sample_rate, 16000))
            mixture_16 = (mixture if sample_rate == 16000 else
                          resample_to(mixture, sample_rate, 16000))
            recomputed = float(si_sdr(reference_16, mixture_16))

            if k == 8:
                record = k8_record(model, trial_id)
                pesq_value = record["pesq"]
                stoi_value = record["stoi"]
                old_si_sdr = ""
                error = ""
                source_path = record["source_path"]
                clip_index = record["clip_index"]
            else:
                record = published[trial_id]
                pesq_value = record["pesq"]
                stoi_value = record["stoi"]
                old_si_sdr = record["si_sdr"]
                error = abs(recomputed - float(old_si_sdr))
                source_path = record["source_path"]
                clip_index = record["clip_index"]

            rows.append({
                "model": model,
                "k": k,
                "trial_id": trial_id,
                "speaker": speaker,
                "clip_index": clip_index,
                "source_path": source_path,
                "coalition_payloads": json.dumps(payloads),
                "reference_payload": payloads[0],
                "published_pesq": pesq_value,
                "published_stoi": stoi_value,
                "published_si_sdr": old_si_sdr,
                "recomputed_si_sdr": f"{recomputed:.6f}",
                "snr": f"{snr_db(reference_16, mixture_16):.6f}",
                "si_sdr_abs_error": "" if error == "" else f"{error:.6f}",
            })
            complete.add(key)
            if len(complete) % 10 == 0:
                rows.sort(key=lambda row: (int(row["k"]),
                                           int(row["trial_id"])))
                atomic_csv(out_path, rows)
                assigned = sum(
                    index % args.num_shards == args.shard_id
                    for index in range(args.n_trials)
                )
                print(f"{model}: {len(complete)}/{len(args.ks) * assigned}",
                      flush=True)

    rows.sort(key=lambda row: (int(row["k"]), int(row["trial_id"])))
    atomic_csv(out_path, rows)
    print(json.dumps({"model": model, "completed": len(rows),
                      "output": str(out_path)}), flush=True)


if __name__ == "__main__":
    main()

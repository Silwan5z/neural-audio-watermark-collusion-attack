#!/usr/bin/env python3
"""Current-protocol independent-codec stress test for K=5 averaging."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE,
    bits_to_int,
    detect_many_native,
    get_or_embed_native,
)
from registry import full_registry_bits, source_record, trial_schedule  # noqa: E402
from watermarks import pesq_wb, resample_to, si_sdr, stoi  # noqa: E402

K = 5
CODECS = ("none", "mp3_128k", "opus_64k")
CODEC_LABELS = {
    "none": "none",
    "mp3_128k": "MP3 128 kbps",
    "opus_64k": "Opus 64 kbps",
}
FIELDS = [
    "model", "k", "trial_id", "speaker", "clip_index", "source_path",
    "sample_rate", "coalition_payloads", "codec", "codec_setting",
    "valid_post_codec_copy_count", "all_post_codec_copies_valid",
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
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed ({completed.returncode}): {message}")


def codec_roundtrip(waveform: np.ndarray, sample_rate: int,
                    codec: str) -> np.ndarray:
    if codec == "none":
        return waveform.copy()
    with tempfile.TemporaryDirectory(prefix="collusion-codec-") as directory:
        work = Path(directory)
        source = work / "input.wav"
        encoded = work / ("encoded.mp3" if codec == "mp3_128k" else
                          "encoded.ogg")
        decoded = work / "decoded.wav"
        sf.write(source, waveform, sample_rate, subtype="FLOAT")
        codec_args = (["-c:a", "libmp3lame", "-b:a", "128k"]
                      if codec == "mp3_128k" else
                      ["-c:a", "libopus", "-b:a", "64k"])
        run_ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", "1", "-i", str(source), *codec_args, str(encoded),
        ])
        run_ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", "1", "-i", str(encoded), "-ar", str(sample_rate),
            "-ac", "1", "-c:a", "pcm_f32le", str(decoded),
        ])
        output, observed_rate = sf.read(decoded, dtype="float32")
    if observed_rate != sample_rate:
        raise RuntimeError(
            f"codec returned {observed_rate} Hz, expected {sample_rate} Hz")
    if len(output) < len(waveform):
        output = np.pad(output, (0, len(waveform) - len(output)))
    return np.asarray(output[:len(waveform)], dtype=np.float32)


def independently_code(waveforms: list[np.ndarray], sample_rate: int,
                       codec: str) -> list[np.ndarray]:
    if codec == "none":
        return [waveform.copy() for waveform in waveforms]
    with ThreadPoolExecutor(max_workers=len(waveforms)) as pool:
        return list(pool.map(
            lambda waveform: codec_roundtrip(waveform, sample_rate, codec),
            waveforms))


def coalitions_for(model: str, path: Path) -> dict[int, list[int]]:
    frame = pd.read_csv(path)
    frame = frame[(frame["model"] == model) & (frame["k"] == K)]
    if len(frame) != 300:
        raise RuntimeError(f"{model}: expected 300 validated K=5 coalitions")
    if frame["trial_id"].nunique() != 300:
        raise RuntimeError(f"{model}: duplicate K=5 trial IDs")
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
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "codec_stress_test")
    parser.add_argument(
        "--coalition-file", type=Path,
        default=ROOT / "data" / "supplementary" / "quality" /
        "all_trials.csv",
        help="validated uniform-trial records supplying the K=5 coalitions",
    )
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require 0 <= shard-id < num-shards")

    model = args.model
    sample_rate = NATIVE_SAMPLE_RATE[model]
    registry = full_registry_bits(model)
    coalitions = coalitions_for(model, args.coalition_file)
    suffix = ("" if args.num_shards == 1 else
              f".shard{args.shard_id}of{args.num_shards}")
    output = args.output_dir / f"{model}{suffix}.csv"
    rows: list[dict] = []
    completed: set[int] = set()
    if output.exists():
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and not set(FIELDS).issubset(rows[0]):
            print(f"{model}: ignoring legacy checkpoint without codec controls",
                  flush=True)
            rows = []
        counts: dict[int, int] = {}
        for row in rows:
            trial_id = int(row["trial_id"])
            counts[trial_id] = counts.get(trial_id, 0) + 1
        completed = {trial_id for trial_id, count in counts.items()
                     if count == len(CODECS)}
        rows = [row for row in rows if int(row["trial_id"]) in completed]

    started = time.time()
    schedule = trial_schedule(args.n_trials)
    for trial_id, (speaker, clip_slot) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
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
        transformed_by_codec: list[list[np.ndarray]] = []
        signals = []
        for codec in CODECS:
            transformed = independently_code(members, sample_rate, codec)
            transformed_by_codec.append(transformed)
            signals.append(np.mean(
                np.stack(transformed), axis=0,
                dtype=np.float64).astype(np.float32))

        member_decoded = [
            detect_many_native(model, transformed, sample_rate, registry)
            for transformed in transformed_by_codec
        ]
        decoded = detect_many_native(model, signals, sample_rate, registry)
        source = source_record(speaker, clip_slot)
        reference_16 = (reference if sample_rate == 16000 else
                        resample_to(reference, sample_rate, 16000))
        for codec, signal, decoded_members, (scores, _, _) in zip(
                CODECS, signals, member_decoded, decoded):
            if not np.isfinite(scores).all() or not np.any(scores):
                raise RuntimeError(
                    f"{model} trial={trial_id} codec={codec}: "
                    "decoder returned no usable payload scores")
            escaped, margin = attack_metrics(scores, coalition)
            valid_post_codec = sum(
                hard is not None and bits_to_int(hard) == payload
                for payload, (_, _, hard) in zip(coalition, decoded_members)
            )
            signal_16 = (signal if sample_rate == 16000 else
                         resample_to(signal, sample_rate, 16000))
            rows.append({
                "model": model,
                "k": K,
                "trial_id": trial_id,
                "speaker": speaker,
                "clip_index": source["clip_index"],
                "source_path": source["path"],
                "sample_rate": sample_rate,
                "coalition_payloads": json.dumps(coalition),
                "codec": codec,
                "codec_setting": CODEC_LABELS[codec],
                "valid_post_codec_copy_count": valid_post_codec,
                "all_post_codec_copies_valid": int(valid_post_codec == K),
                "escaped": escaped,
                "attribution_margin": f"{margin:.8f}",
                "pesq": f"{pesq_wb(reference_16, signal_16):.6f}",
                "stoi": f"{stoi(reference_16, signal_16):.6f}",
                "si_sdr": f"{si_sdr(reference_16, signal_16):.6f}",
                "snr": f"{snr_db(reference_16, signal_16):.6f}",
            })
        completed.add(trial_id)
        if len(completed) % 10 == 0:
            rows.sort(key=lambda row: (int(row["trial_id"]), row["codec"]))
            atomic_csv(output, rows)
            assigned = sum(index % args.num_shards == args.shard_id
                           for index in range(args.n_trials))
            print(f"{model}: {len(completed)}/{assigned} "
                  f"({time.time() - started:.0f}s)", flush=True)

    rows.sort(key=lambda row: (int(row["trial_id"]), row["codec"]))
    atomic_csv(output, rows)
    print(json.dumps({
        "model": model,
        "trials": len(completed),
        "rows": len(rows),
        "output": str(output),
        "elapsed_seconds": time.time() - started,
    }), flush=True)


if __name__ == "__main__":
    main()

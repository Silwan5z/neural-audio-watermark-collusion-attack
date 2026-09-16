#!/usr/bin/env python3
"""Compute speech-mode ViSQOL for the released uniform mixtures.

This entry point uses the official ViSQOL command-line binary.  It reconstructs
the same personalized-copy reference and uniform mixture recorded by
``compute_uniform_quality.py``, writes short-lived 16 kHz PCM WAV batches, and
stores restart-safe trial scores under ``results/quality``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from native_audio import NATIVE_SAMPLE_RATE, get_or_embed_native  # noqa: E402
from watermarks import resample_to  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = ("model", "k", "trial_id", "visqol")


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


def load_checkpoint(path: Path) -> tuple[list[dict], set[tuple[int, int]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    completed = {(int(row["k"]), int(row["trial_id"])) for row in rows}
    return rows, completed


def reconstruct(row) -> tuple[np.ndarray, np.ndarray]:
    model = str(row.model)
    sample_rate = NATIVE_SAMPLE_RATE[model]
    clip_slot = int(row.clip_index) - 1
    payloads = [int(value) for value in json.loads(row.coalition_payloads)]
    waveforms = [
        get_or_embed_native(model, str(row.speaker), payload, clip_slot)[0]
        for payload in payloads
    ]
    length = min(map(len, waveforms))
    waveforms = [np.asarray(waveform[:length], dtype=np.float32)
                 for waveform in waveforms]
    reference = waveforms[0]
    mixture = np.mean(np.stack(waveforms), axis=0,
                      dtype=np.float64).astype(np.float32)
    if sample_rate != 16000:
        reference = resample_to(reference, sample_rate, 16000)
        mixture = resample_to(mixture, sample_rate, 16000)
    return (np.asarray(reference, dtype=np.float32),
            np.asarray(mixture, dtype=np.float32))


def run_batch(binary: Path, working_directory: Path,
              batch: list, temporary_directory: Path) -> list[float]:
    input_path = temporary_directory / "pairs.csv"
    output_path = temporary_directory / "scores.csv"
    pairs = []
    for index, row in enumerate(batch):
        reference, mixture = reconstruct(row)
        reference_path = temporary_directory / f"reference_{index:04d}.wav"
        degraded_path = temporary_directory / f"mixture_{index:04d}.wav"
        sf.write(reference_path, reference, 16000, subtype="PCM_16")
        sf.write(degraded_path, mixture, 16000, subtype="PCM_16")
        pairs.append((str(reference_path), str(degraded_path)))

    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("reference", "degraded"))
        writer.writerows(pairs)

    command = [
        str(binary),
        "--batch_input_csv", str(input_path),
        "--results_csv", str(output_path),
        "--use_speech_mode",
    ]
    completed = subprocess.run(
        command, cwd=working_directory, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ViSQOL failed with exit code {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}")

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RuntimeError("ViSQOL returned an empty results file")
    header = [value.strip().lower() for value in rows[0]]
    try:
        score_index = header.index("moslqo")
    except ValueError as error:
        raise RuntimeError(
            f"ViSQOL results do not contain a moslqo column: {header}") from error
    rows = rows[1:]
    if len(rows) != len(batch):
        raise RuntimeError(
            f"ViSQOL returned {len(rows)} scores for {len(batch)} pairs")
    return [float(row[score_index]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--visqol-bin", type=Path, required=True)
    parser.add_argument("--visqol-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data" / "supplementary" / "quality" /
        "all_trials.csv")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "quality")
    args = parser.parse_args()

    binary = args.visqol_bin.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"ViSQOL binary is not executable: {binary}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require 0 <= shard-id < num-shards")
    working_directory = (args.visqol_root.resolve() if args.visqol_root
                         else binary.parent.parent.resolve())

    frame = pd.read_csv(args.input)
    frame = frame[frame.model == args.model].sort_values(
        ["k", "trial_id"]).reset_index(drop=True)
    if len(frame) != 1200:
        raise RuntimeError(f"{args.model}: expected 1200 released trials")
    frame = frame.iloc[args.shard_id::args.num_shards].reset_index(drop=True)
    if args.limit is not None:
        frame = frame.iloc[:args.limit]

    suffix = ("" if args.num_shards == 1 else
              f".shard{args.shard_id}of{args.num_shards}")
    output = args.output_dir / f"visqol_{args.model}{suffix}.csv"
    rows, completed = load_checkpoint(output)
    pending = [row for row in frame.itertuples()
               if (int(row.k), int(row.trial_id)) not in completed]

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        with tempfile.TemporaryDirectory(prefix="visqol-batch-") as directory:
            scores = run_batch(
                binary, working_directory, batch, Path(directory))
        for row, score in zip(batch, scores):
            rows.append({
                "model": args.model,
                "k": int(row.k),
                "trial_id": int(row.trial_id),
                "visqol": f"{score:.8f}",
            })
            completed.add((int(row.k), int(row.trial_id)))
        rows.sort(key=lambda row: (int(row["k"]), int(row["trial_id"])))
        atomic_csv(output, rows)
        print(f"{args.model}: {len(completed)}/{len(frame)}", flush=True)

    print(json.dumps({
        "model": args.model,
        "completed": len(completed),
        "output": str(output),
        "mode": "speech",
        "sample_rate": 16000,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }), flush=True)


if __name__ == "__main__":
    main()

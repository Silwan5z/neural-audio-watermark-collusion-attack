#!/usr/bin/env python3
"""Evaluate adaptive mixture paths between valid one-bit payload pairs.

For each existing K=5 one-bit pair, first decode weights 0.0, 0.1, ..., 1.0. Every
coarse interval in which the designated flipped bit changes native hard state
is subdivided into ten parts (nine new interior points).  If endpoint decoder
errors yield no hard crossing, the interval with the largest target-bit
probability change is refined and explicitly marked as a fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from registry import cache_path_for  # noqa: E402
from watermarks import get_audioseal, get_voicemark  # noqa: E402

SR = 16000
COALITION_SIZE = 5
COARSE_INDEX = tuple(range(0, 101, 10))
POINT_FIELDS = [
    "model", "k", "trial_id", "speaker", "clip_index", "base_payload",
    "flipped_payload", "flipped_bit", "weight_index", "flipped_copy_weight",
    "stage", "refined_interval_low", "refinement_reason",
    "audio_path", "bit_probabilities", "bit_logits", "decoded_bits",
    "threshold_bits", "decoded_payload", "threshold_payload",
    "payload_confidence", "payload_log_confidence", "payload_margin",
    "watermark_score", "watermark_logits",
    "chunk_probabilities", "chunk_logits",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("audioseal", "voicemark"))
    p.add_argument("--shard-id", required=True, type=int)
    p.add_argument("--num-shards", default=7, type=int)
    p.add_argument("--n-trials", default=300, type=int)
    p.add_argument("--checkpoint-every", default=2, type=int)
    p.add_argument("--input-dir", type=Path,
                   default=ROOT / "results" / "one_bit" / "pairs")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "one_bit" / "paths")
    return p.parse_args()


def bits_to_int(bits: np.ndarray) -> int:
    return int(sum(int(v) << i for i, v in enumerate(bits)))


def load_wav(path: Path) -> np.ndarray:
    sr, wav = wavfile.read(path)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}: {path}")
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        wav = wav.astype(np.float32) / max(abs(info.min), info.max)
    return wav.astype(np.float32, copy=False)


def jdump(value) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"), allow_nan=False)


def atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=POINT_FIELDS)
        w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_complete_trials(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    completed = set()
    for trial, rr in grouped.items():
        idx = {int(r["weight_index"]) for r in rr}
        if set(COARSE_INDEX).issubset(idx) and any(
                row["stage"] == "refined" for row in rr):
            completed.add(trial)
    kept = [r for r in rows if int(r["trial_id"]) in completed]
    return kept, completed


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def decode_full(model: str, wav: np.ndarray) -> dict:
    """Return native decisions plus every available confidence representation."""
    import torch

    if model == "audioseal":
        m = get_audioseal()
        t = torch.from_numpy(wav).float().to(m["dev"])[None, None]
        with torch.no_grad():
            response = m["det"].detector(t)
        bit_logits = response[:, 2:].mean(dim=-1)[0].cpu().numpy().astype(np.float64)
        bit_probs = sigmoid(bit_logits)
        native_hard = (bit_probs > 0.5).astype(np.int8)
        presence_logits = response[:, :2].mean(dim=-1)[0].cpu().numpy().astype(np.float64)
        presence_probability = float(torch.softmax(torch.from_numpy(presence_logits), dim=0)[1])
        log_selected = np.log(np.clip(np.maximum(bit_probs, 1.0-bit_probs), 1e-300, 1.0))
        payload_log_confidence = float(log_selected.sum())
        payload_confidence = float(np.exp(payload_log_confidence))
        payload_margin = float(np.min(np.abs(bit_logits)))
        chunk_probs = np.empty((0, 0), dtype=np.float64)
        chunk_logits = np.empty((0, 0), dtype=np.float64)
    else:
        m = get_voicemark()
        t = torch.from_numpy(wav).float().to(m["dev"])[None, None]
        with torch.no_grad():
            presence_raw, raw_chunk_logits = m["solver"].model.detect_watermark(
                t, return_logits=True)
        chunk_logits = raw_chunk_logits[0].cpu().numpy().astype(np.float64)
        chunk_probs = torch.softmax(raw_chunk_logits, dim=-1)[0].cpu().numpy().astype(np.float64)
        vals = np.argmax(chunk_probs, axis=1)
        native_hard = np.asarray(
            [(int(v) >> j) & 1 for v in vals for j in range(4)], dtype=np.int8)
        # Exact bit marginals of each native 16-way chunk distribution.
        bit_probs_list = []
        for chunk in range(4):
            for bit in range(4):
                keep = ((np.arange(16) >> bit) & 1).astype(bool)
                bit_probs_list.append(float(chunk_probs[chunk, keep].sum()))
        bit_probs = np.asarray(bit_probs_list, dtype=np.float64)
        clipped = np.clip(bit_probs, 1e-12, 1.0-1e-12)
        bit_logits = np.log(clipped) - np.log1p(-clipped)
        top2 = np.sort(chunk_probs, axis=1)[:, -2:]
        payload_log_confidence = float(
            np.log(np.clip(top2[:, 1], 1e-300, 1.0)).sum())
        payload_confidence = float(np.exp(payload_log_confidence))
        payload_margin = float(np.min(
            np.log(np.clip(top2[:, 1], 1e-300, 1.0))
            - np.log(np.clip(top2[:, 0], 1e-300, 1.0))))
        presence_probability = float(torch.sigmoid(presence_raw).mean().cpu())
        presence_logits = presence_raw.detach().float().cpu().numpy().reshape(-1).astype(np.float64)

    threshold_hard = (bit_probs > 0.5).astype(np.int8)
    decoded_payload = bits_to_int(native_hard)
    return {
        "bit_probabilities": bit_probs,
        "bit_logits": bit_logits,
        "decoded_bits": native_hard,
        "threshold_bits": threshold_hard,
        "decoded_payload": decoded_payload,
        "threshold_payload": bits_to_int(threshold_hard),
        "payload_confidence": payload_confidence,
        "payload_log_confidence": payload_log_confidence,
        "payload_margin": payload_margin,
        "watermark_score": presence_probability,
        "watermark_logits": presence_logits,
        "chunk_probabilities": chunk_probs,
        "chunk_logits": chunk_logits,
    }


def point_row(model: str, source: dict, weight_index: int, stage: str,
              interval_low: int | None, reason: str, audio_path: Path,
              decoded: dict) -> dict:
    return {
        "model": model, "k": COALITION_SIZE, "trial_id": source["trial_id"],
        "speaker": source["speaker"], "clip_index": source["clip_index"],
        "base_payload": source["base_payload"],
        "flipped_payload": source["flipped_payload"],
        "flipped_bit": source["flipped_bit"],
        "weight_index": weight_index,
        "flipped_copy_weight": f"{weight_index/100:.2f}", "stage": stage,
        "refined_interval_low": "" if interval_low is None else f"{interval_low/100:.2f}",
        "refinement_reason": reason, "audio_path": str(audio_path),
        "bit_probabilities": jdump(decoded["bit_probabilities"]),
        "bit_logits": jdump(decoded["bit_logits"]),
        "decoded_bits": jdump(decoded["decoded_bits"]),
        "threshold_bits": jdump(decoded["threshold_bits"]),
        "decoded_payload": decoded["decoded_payload"],
        "threshold_payload": decoded["threshold_payload"],
        "payload_confidence": f"{decoded['payload_confidence']:.17g}",
        "payload_log_confidence": f"{decoded['payload_log_confidence']:.17g}",
        "payload_margin": f"{decoded['payload_margin']:.17g}",
        "watermark_score": f"{decoded['watermark_score']:.17g}",
        "watermark_logits": jdump(decoded["watermark_logits"]),
        "chunk_probabilities": jdump(decoded["chunk_probabilities"]),
        "chunk_logits": jdump(decoded["chunk_logits"]),
    }


def evaluate_point(model: str, source: dict, wav_a: np.ndarray, wav_b: np.ndarray,
                   weight_index: int, stage: str, interval_low: int | None,
                   reason: str, audio_dir: Path) -> tuple[dict, dict]:
    weight = weight_index / 100.0
    clip_slot = int(source["clip_index"]) - 1
    if weight_index == 0:
        wav = wav_a
        audio_path = cache_path_for(
            model, source["speaker"], clip_slot, int(source["base_payload"]))
    elif weight_index == 100:
        wav = wav_b
        audio_path = cache_path_for(
            model, source["speaker"], clip_slot, int(source["flipped_payload"]))
    else:
        wav = ((1.0 - weight) * wav_a + weight * wav_b).astype(np.float32)
        audio_path = audio_dir / f"weight_{weight_index:03d}.wav"
        if not audio_path.exists():
            wavfile.write(audio_path, SR, wav.astype(np.float32, copy=False))
    decoded = decode_full(model, wav)
    return point_row(model, source, weight_index, stage, interval_low, reason,
                     audio_path, decoded), decoded


def analyse_trial(model: str, source: dict, output_dir: Path) -> list[dict]:
    clip_slot = int(source["clip_index"]) - 1
    wav_a = load_wav(cache_path_for(
        model, source["speaker"], clip_slot, int(source["base_payload"])))
    wav_b = load_wav(cache_path_for(
        model, source["speaker"], clip_slot, int(source["flipped_payload"])))
    n = min(len(wav_a), len(wav_b))
    wav_a, wav_b = wav_a[:n], wav_b[:n]
    trial = int(source["trial_id"])
    audio_dir = output_dir / "audio" / model / f"trial_{trial:03d}"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows, decoded_by_index = [], {}
    for idx in COARSE_INDEX:
        row, decoded = evaluate_point(model, source, wav_a, wav_b, idx, "coarse",
                                      None, "coarse_grid", audio_dir)
        rows.append(row); decoded_by_index[idx] = decoded

    bit = int(source["flipped_bit"])
    changed = []
    for low, high in zip(COARSE_INDEX[:-1], COARSE_INDEX[1:]):
        if decoded_by_index[low]["decoded_bits"][bit] != \
                decoded_by_index[high]["decoded_bits"][bit]:
            changed.append(low)
    if changed:
        intervals = changed
        reason = "target_bit_hard_change"
    else:
        deltas = [abs(decoded_by_index[high]["bit_probabilities"][bit] -
                      decoded_by_index[low]["bit_probabilities"][bit])
                  for low, high in zip(COARSE_INDEX[:-1], COARSE_INDEX[1:])]
        intervals = [int(COARSE_INDEX[int(np.argmax(deltas))])]
        reason = "fallback_largest_target_probability_change"

    for low in intervals:
        for idx in range(low + 1, low + 10):
            row, decoded = evaluate_point(model, source, wav_a, wav_b, idx, "refined",
                                          low, reason, audio_dir)
            rows.append(row); decoded_by_index[idx] = decoded
    rows.sort(key=lambda row: int(row["weight_index"]))
    return rows


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard-id")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.input_dir / f"{args.model}.csv"
    with source_path.open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    if len(source_rows) < args.n_trials:
        raise ValueError(f"{source_path} has only {len(source_rows)} rows")
    assigned = [r for r in source_rows[:args.n_trials]
                if int(r["trial_id"]) % args.num_shards == args.shard_id]
    partial = args.output_dir / "shards" / (
        f"{args.model}.shard{args.shard_id}of{args.num_shards}.partial.csv")
    final = args.output_dir / "shards" / (
        f"{args.model}.shard{args.shard_id}of{args.num_shards}.csv")
    partial.parent.mkdir(parents=True, exist_ok=True)
    rows, completed = load_complete_trials(partial)

    # Load exactly one detector per worker before the timed loop.
    if args.model == "audioseal":
        get_audioseal()
    else:
        get_voicemark()

    new_trials = 0
    for source in assigned:
        trial = int(source["trial_id"])
        if trial in completed:
            continue
        rows.extend(analyse_trial(args.model, source, args.output_dir))
        completed.add(trial); new_trials += 1
        if new_trials % args.checkpoint_every == 0:
            rows.sort(key=lambda r: (
                int(r["trial_id"]), int(r["weight_index"])))
            atomic_write(partial, rows)
            print(f"{args.model} shard {args.shard_id}: {len(completed)}/{len(assigned)} trials",
                  flush=True)
    rows.sort(key=lambda row: (
        int(row["trial_id"]), int(row["weight_index"])))
    atomic_write(partial, rows)
    if len(completed) != len(assigned):
        raise RuntimeError(
            f"incomplete shard: {len(completed)}/{len(assigned)} trials")
    atomic_write(final, rows)
    print(f"complete {args.model} shard {args.shard_id}: trials={len(completed)} points={len(rows)}")


if __name__ == "__main__":
    main()

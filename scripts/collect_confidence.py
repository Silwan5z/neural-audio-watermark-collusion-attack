#!/usr/bin/env python3
"""Collect weakest decoded-bit confidence for the Section 5 mechanism figure.

For every K=8 trial, this script records three conditions:

1. Single: one correctly decoded personalized copy,
2. Average: the valid K=8 uniform average,
3. Targeted: the highest-ranked exact Bit Margin hit, when one exists.

Each trial contributes at most one observation per condition.  The reported
quantity is the minimum confidence assigned to any bit of the decoded payload,
not watermark presence.  For chunk/digit decoders, class probabilities are
marginalized to bits.  WavMark uses the fraction of valid windows voting for 1.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from registry import NBITS, get_or_embed, int_to_bits  # noqa: E402
from watermarks import extract_evidence, get_wavmark  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "k", "trial_id", "condition", "decoded_payload", "bit_count",
    "minimum_confidence", "weakest_bit_index", "source_path", "target_rank",
    "confidence_source",
)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def wavmark_bit_probabilities(waveform: np.ndarray) -> np.ndarray:
    """Return valid-window vote fractions for WavMark payload bits."""
    import torch
    from wavmark.utils import wm_add_util

    model = get_wavmark()
    signal = np.asarray(waveform, dtype=np.float32)
    window, step = 16000, 800
    total = (len(signal) - window) // step
    windows = np.stack([
        signal[position * step:position * step + window]
        for position in range(total)
    ])
    batch_size = int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE", "400"))
    decoded = []
    for offset in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[offset:offset + batch_size]).to(model["dev"])
        with torch.no_grad():
            bits = (model["model"].decode(batch) >= 0.5).int().cpu().numpy()
        decoded.append(bits)
    decoded = np.concatenate(decoded, axis=0)
    start = np.asarray(wm_add_util.fix_pattern[:16], dtype=np.int8)
    valid = decoded[np.all(decoded[:, :16] == start[None, :], axis=1)]
    if len(valid) == 0:
        raise RuntimeError("WavMark produced no valid start-pattern windows")
    return valid[:, 16:32].mean(axis=0).astype(np.float64)


def bit_probabilities(model: str, waveform: np.ndarray) -> tuple[np.ndarray, str]:
    if model == "wavmark":
        return wavmark_bit_probabilities(waveform), "valid-window vote fraction"
    evidence = extract_evidence(model, np.asarray(waveform, dtype=np.float32))
    if evidence is None:
        raise RuntimeError(f"missing soft evidence for {model}")
    probability = (np.asarray(evidence, dtype=np.float64) + 1.0) / 2.0
    return np.clip(probability, 0.0, 1.0), "native decoder bit marginal"


def minimum_confidence(probability: np.ndarray, payload: int,
                       payload_bits: int) -> tuple[float, int]:
    bits = int_to_bits(payload, payload_bits).astype(np.int8)
    confidence = np.where(bits == 1, probability, 1.0 - probability)
    weakest = int(np.argmin(confidence))
    return float(confidence[weakest]), weakest


def load_targeted_rows(root: Path, model: str) -> tuple[dict[int, dict], dict[int, dict]]:
    merged = root / "k8" / "bit_margin" / f"{model}.csv"
    if merged.exists():
        with merged.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        pattern = f"{model}.shard*of7.csv"
        for path in sorted((root / "k8" / "bit_margin" / "shards").glob(pattern)):
            if path.name.endswith(".partial.csv"):
                continue
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    if len(rows) != 3000:
        raise RuntimeError(f"expected 3000 final Bit Margin rows for {model}, got {len(rows)}")
    source_rows: dict[int, dict] = {}
    selected: dict[int, dict] = {}
    for row in rows:
        trial = int(row["trial_id"])
        source_rows.setdefault(trial, row)
        if int(row["target_hit"]) != 1:
            continue
        if trial not in selected or int(row["target_rank"]) < int(selected[trial]["target_rank"]):
            selected[trial] = row
    if len(source_rows) != 300:
        raise RuntimeError(f"expected 300 Bit Margin trials for {model}, got {len(source_rows)}")
    return source_rows, selected


def load_checkpoint(path: Path) -> tuple[list[dict], set[tuple[int, str]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = {(int(row["trial_id"]), row["condition"]) for row in rows}
    return rows, complete


def append_row(rows: list[dict], complete: set[tuple[int, str]], *, model: str,
               trial: int, condition: str, payload: int, probability: np.ndarray,
               source_path: str, target_rank: str,
               confidence_source: str) -> None:
    key = (trial, condition)
    if key in complete:
        return
    value, weakest = minimum_confidence(probability, payload, NBITS[model])
    rows.append({
        "model": model,
        "k": 8,
        "trial_id": trial,
        "condition": condition,
        "decoded_payload": payload,
        "bit_count": NBITS[model],
        "minimum_confidence": f"{value:.10f}",
        "weakest_bit_index": weakest,
        "source_path": source_path,
        "target_rank": target_rank,
        "confidence_source": confidence_source,
    })
    complete.add(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--average-dir", type=Path,
        default=ROOT / "data" / "average" / "k8")
    parser.add_argument(
        "--targeted-dir", type=Path,
        default=ROOT / "data" / "targeted")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "confidence")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard-id < num-shards")

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_id}of{args.num_shards}"
    final = args.output_dir / f"{args.model}{suffix}.csv"
    partial = args.output_dir / f"{args.model}{suffix}.partial.csv"
    rows, complete = load_checkpoint(partial)
    assigned = [trial for trial in range(300) if trial % args.num_shards == args.shard_id]

    source_rows, successful = load_targeted_rows(args.targeted_dir, args.model)

    for position, trial in enumerate(assigned, start=1):
        record_path = args.average_dir / args.model / f"trial_{trial:03d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source_path = str(record["source_path"])

        uniform_probability = np.asarray(record["bit_probabilities"], dtype=np.float64)
        append_row(
            rows, complete, model=args.model, trial=trial,
            condition="average", payload=int(record["decoded_payload"]),
            probability=uniform_probability, source_path=source_path, target_rank="",
            confidence_source=str(record["confidence_source"]),
        )

        if (trial, "single") not in complete:
            source_row = source_rows[trial]
            coalition = [int(value) for value in json.loads(source_row["coalition_payloads"])]
            benign = np.asarray(
                get_or_embed(
                    args.model, source_row["speaker"], coalition[0],
                    int(source_row["clip_index"]) - 1),
                dtype=np.float32,
            )
            probability, kind = bit_probabilities(args.model, benign)
            append_row(
                rows, complete, model=args.model, trial=trial,
                condition="single", payload=coalition[0], probability=probability,
                source_path=source_path, target_rank="",
                confidence_source=kind,
            )

        targeted = successful.get(trial)
        if targeted is not None and (trial, "targeted") not in complete:
            targeted_coalition = [
                int(value) for value in json.loads(targeted["coalition_payloads"])
            ]
            weights = np.asarray(json.loads(targeted["weights"]), dtype=np.float64)
            waveforms = [
                np.asarray(
                    get_or_embed(args.model, targeted["speaker"], payload,
                                 int(targeted["clip_index"]) - 1),
                    dtype=np.float32,
                )
                for payload in targeted_coalition
            ]
            length = min(map(len, waveforms))
            attacked = np.asarray(sum(
                weights[index] * waveforms[index][:length]
                for index in range(8)
            ), dtype=np.float32)
            probability, kind = bit_probabilities(args.model, attacked)
            target = int(targeted["target_payload"])
            append_row(
                rows, complete, model=args.model, trial=trial,
                condition="targeted", payload=target, probability=probability,
                source_path=source_path, target_rank=targeted["target_rank"],
                confidence_source=kind,
            )

        rows.sort(key=lambda row: (int(row["trial_id"]), row["condition"]))
        atomic_csv(partial, rows)
        if position % 5 == 0 or position == len(assigned):
            print(
                f"checkpoint model={args.model} shard={args.shard_id}/{args.num_shards} "
                f"assigned={position}/{len(assigned)} "
                f"rows={len(rows)} targeted_trials={len(successful)}",
                flush=True,
            )

    atomic_csv(final, rows)
    print(
        f"COMPLETE model={args.model} shard={args.shard_id}/{args.num_shards} "
        f"rows={len(rows)} "
        f"targeted_trials={len(successful)} output={final}",
        flush=True,
    )


if __name__ == "__main__":
    main()

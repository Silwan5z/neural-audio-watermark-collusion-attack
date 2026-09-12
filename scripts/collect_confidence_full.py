#!/usr/bin/env python3
"""Collect full bit-confidence vectors for confidence screening.

For every K=8 trial, this script records one valid Single output, one uniform
Average output, and every exact Target-Bit Margin hit. The complete vectors
are required by ``screen_confidence.py``; ``collect_confidence.py`` remains the
compact collector used by the confidence-distribution figure.
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

from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE, bit_probabilities_native, get_or_embed_native,
)
from registry import NBITS, int_to_bits  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "k", "trial_id", "speaker", "condition", "identity",
    "target_rank", "bit_count", "minimum_confidence", "mean_confidence",
    "log_confidence_variance", "bit_probabilities", "bit_confidences",
    "source_path", "confidence_source",
)


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


def load_targeted_rows(root: Path, model: str) -> dict[int, list[dict]]:
    merged = root / "k8" / "bit_margin" / f"{model}.csv"
    if merged.exists():
        paths = [merged]
    else:
        paths = sorted(
            path for path in (root / "k8" / "bit_margin" / "shards").glob(
                f"{model}.shard*of*.csv")
            if not path.name.endswith(".partial.csv")
        )
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 3000:
        raise RuntimeError(
            f"expected 3000 final Target-Bit Margin rows for {model}, got {len(rows)}")
    grouped = {trial: [] for trial in range(300)}
    for row in rows:
        grouped[int(row["trial_id"])].append(row)
    for trial, group in grouped.items():
        ranks = {int(row["target_rank"]) for row in group}
        if len(group) != 10 or ranks != set(range(1, 11)):
            raise RuntimeError(f"incomplete target attempts for {model} trial={trial}")
        group.sort(key=lambda row: int(row["target_rank"]))
    return grouped


def load_checkpoint(path: Path, model: str, assigned: set[int],
                    grouped: dict[int, list[dict]]) -> tuple[list[dict], set[int]]:
    """Keep only complete trials from a prior partial shard."""
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_trial: dict[int, list[dict]] = {}
    for row in rows:
        trial = int(row["trial_id"])
        if row["model"] == model and trial in assigned:
            by_trial.setdefault(trial, []).append(row)
    complete: set[int] = set()
    for trial, trial_rows in by_trial.items():
        expected_ranks = {
            str(int(row["target_rank"])) for row in grouped[trial]
            if int(row["target_hit"]) == 1
        }
        observed_ranks = {
            row["target_rank"] for row in trial_rows
            if row["condition"] == "targeted"
        }
        conditions = [row["condition"] for row in trial_rows]
        if (conditions.count("single") == 1
                and conditions.count("average") == 1
                and observed_ranks == expected_ranks
                and len(trial_rows) == 2 + len(expected_ranks)):
            complete.add(trial)
    kept = [row for trial in complete for row in by_trial[trial]]
    kept.sort(key=lambda row: (
        int(row["trial_id"]),
        {"single": 0, "average": 1, "targeted": 2}[row["condition"]],
        int(row["target_rank"] or 0)))
    return kept, complete


def confidence_row(*, model: str, trial: int, speaker: str, condition: str,
                   identity: int, target_rank: str, probability: np.ndarray,
                   source_path: str, source: str) -> dict:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    bit_count = NBITS[model]
    if len(probability) != bit_count:
        raise RuntimeError(
            f"{model} trial={trial} {condition}: expected {bit_count} bits, "
            f"got {len(probability)}")
    bits = int_to_bits(identity, bit_count).astype(np.int8)
    confidence = np.where(bits == 1, probability, 1.0 - probability)
    return {
        "model": model,
        "k": 8,
        "trial_id": trial,
        "speaker": speaker,
        "condition": condition,
        "identity": identity,
        "target_rank": target_rank,
        "bit_count": bit_count,
        "minimum_confidence": f"{float(confidence.min()):.10f}",
        "mean_confidence": f"{float(confidence.mean()):.10f}",
        "log_confidence_variance": f"{float(np.log(confidence.var() + 1e-12)):.10f}",
        "bit_probabilities": json.dumps(probability.tolist()),
        "bit_confidences": json.dumps(confidence.tolist()),
        "source_path": source_path,
        "confidence_source": source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--average-dir", type=Path, default=ROOT / "data" / "average" / "k8")
    parser.add_argument(
        "--targeted-dir", type=Path, default=ROOT / "results" / "targeted")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "confidence_full")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= shard-id < num-shards")
    return args


def main() -> None:
    args = parse_args()
    model = args.model
    sample_rate = NATIVE_SAMPLE_RATE[model]
    grouped = load_targeted_rows(args.targeted_dir, model)
    assigned = [trial for trial in range(300)
                if trial % args.num_shards == args.shard_id]
    assigned_set = set(assigned)
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_id}of{args.num_shards}"
    final = args.output_dir / f"{model}{suffix}.csv"
    partial = args.output_dir / f"{model}{suffix}.partial.csv"
    rows, complete = load_checkpoint(
        partial, model, assigned_set, grouped)
    print(
        f"resume model={model} shard={args.shard_id}/{args.num_shards}: "
        f"{len(complete)}/{len(assigned)} trials",
        flush=True)

    for position, trial in enumerate(assigned, start=1):
        if trial in complete:
            continue
        average = json.loads(
            (args.average_dir / model / f"trial_{trial:03d}.json").read_text(
                encoding="utf-8"))
        source_path = str(average["source_path"])
        speaker = str(average["speaker"])
        if (average["model"] != model or int(average["k"]) != 8
                or int(average["trial_id"]) != trial):
            raise RuntimeError(f"average record mismatch for {model} trial={trial}")
        rows.append(confidence_row(
            model=model, trial=trial, speaker=speaker, condition="average",
            identity=int(average["decoded_payload"]), target_rank="",
            probability=np.asarray(average["bit_probabilities"], dtype=np.float64),
            source_path=source_path, source=str(average["confidence_source"])))

        source_row = grouped[trial][0]
        if source_row["speaker"] != speaker:
            raise RuntimeError(f"speaker mismatch for {model} trial={trial}")
        coalition = [int(value) for value in json.loads(source_row["coalition_payloads"])]
        clip_slot = int(source_row["clip_index"]) - 1
        cached: dict[int, np.ndarray] = {
            coalition[0]: np.asarray(
                get_or_embed_native(
                    model, speaker, coalition[0], clip_slot)[0], dtype=np.float32)
        }
        probability, kind = bit_probabilities_native(
            model, cached[coalition[0]], sample_rate)
        rows.append(confidence_row(
            model=model, trial=trial, speaker=speaker, condition="single",
            identity=coalition[0], target_rank="", probability=probability,
            source_path=source_path, source=kind))

        hits = [attack for attack in grouped[trial]
                if int(attack["target_hit"]) == 1]
        waveforms = None
        if hits:
            for payload in coalition:
                if payload not in cached:
                    cached[payload] = np.asarray(
                        get_or_embed_native(
                            model, speaker, payload, clip_slot)[0], dtype=np.float32)
            waveforms = [cached[payload] for payload in coalition]
        for attack in hits:
            attack_coalition = [
                int(value) for value in json.loads(attack["coalition_payloads"])]
            if attack_coalition != coalition:
                raise RuntimeError(f"coalition mismatch for {model} trial={trial}")
            assert waveforms is not None
            length = min(map(len, waveforms))
            weights = np.asarray(json.loads(attack["weights"]), dtype=np.float64)
            mixture = np.asarray(sum(
                weights[index] * waveforms[index][:length]
                for index in range(8)), dtype=np.float32)
            probability, kind = bit_probabilities_native(
                model, mixture, sample_rate)
            rows.append(confidence_row(
                model=model, trial=trial, speaker=speaker, condition="targeted",
                identity=int(attack["target_payload"]),
                target_rank=attack["target_rank"], probability=probability,
                source_path=source_path, source=kind))

        rows.sort(key=lambda row: (
            int(row["trial_id"]),
            {"single": 0, "average": 1, "targeted": 2}[row["condition"]],
            int(row["target_rank"] or 0)))
        atomic_csv(partial, rows)
        complete.add(trial)
        if position % 5 == 0 or position == len(assigned):
            print(
                f"checkpoint model={model} shard={args.shard_id}/{args.num_shards} "
                f"assigned={position}/{len(assigned)} rows={len(rows)}",
                flush=True)

    atomic_csv(final, rows)
    print(f"COMPLETE model={model} rows={len(rows)} output={final}", flush=True)


if __name__ == "__main__":
    main()

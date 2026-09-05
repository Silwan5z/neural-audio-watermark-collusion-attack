#!/usr/bin/env python3
"""Collect full bit confidence for a cached 10-trial rejection pilot.

No watermark embedding or target search is performed. Average probabilities are
read from the source-correct K=8 records; Single and successful MRC Targeted
mixtures are decoded from cached marked copies and saved with their complete bit
probability and decoded-bit confidence vectors.
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

from collect_identity_bit_confidence_k8 import bit_probabilities  # noqa: E402
from registry import NBITS, get_or_embed, int_to_bits  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "K", "trial_id", "condition", "identity", "target_rank",
    "nbits", "minimum", "second_lowest", "weakest_bit",
    "second_weakest_bit", "bit_probability", "bit_confidence",
    "source_path", "confidence_kind",
)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_mrc_rows(root: Path, model: str, n_trials: int) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {trial: [] for trial in range(n_trials)}
    pattern = f"mrc_pm_native_mrc_{model}_K8_shard*of7.csv"
    paths = [p for p in sorted((root / "shards").glob(pattern))
             if not p.name.endswith(".partial.csv")]
    if len(paths) != 7:
        raise RuntimeError(f"expected 7 final MRC shards for {model}, got {len(paths)}")
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                trial = int(row["trial_id"])
                if trial < n_trials:
                    grouped[trial].append(row)
    for trial, rows in grouped.items():
        if len(rows) != 10 or {int(r["target_rank"]) for r in rows} != set(range(1, 11)):
            raise RuntimeError(f"incomplete target attempts for {model} trial={trial}: {len(rows)}")
    return grouped


def append(rows: list[dict], *, model: str, trial: int, condition: str,
           identity: int, probability: np.ndarray, source_path: str,
           target_rank: str, kind: str) -> None:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(probability) != NBITS[model]:
        raise RuntimeError(
            f"{model} trial={trial} {condition}: expected {NBITS[model]} bits, "
            f"got {len(probability)}")
    bits = int_to_bits(identity, NBITS[model]).astype(np.int8)
    confidence = np.where(bits == 1, probability, 1.0 - probability)
    order = np.argsort(confidence, kind="stable")
    rows.append({
        "model": model,
        "K": 8,
        "trial_id": trial,
        "condition": condition,
        "identity": identity,
        "target_rank": target_rank,
        "nbits": NBITS[model],
        "minimum": f"{float(confidence[order[0]]):.10f}",
        "second_lowest": f"{float(confidence[order[1]]):.10f}",
        "weakest_bit": int(order[0]),
        "second_weakest_bit": int(order[1]),
        "bit_probability": json.dumps(probability.tolist()),
        "bit_confidence": json.dumps(confidence.tolist()),
        "source_path": source_path,
        "confidence_kind": kind,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trial-end", type=int)
    parser.add_argument(
        "--k8-dir", type=Path,
        default=ROOT / "data" / "k8_population_source_correct_300clips_20260902" / "raw")
    parser.add_argument(
        "--attack-dir", type=Path,
        default=ROOT / "results" / "mrc_pm_native_shared4_fulltop10_300clips_20260904")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "confidence_rejection_pilot10_20260905")
    args = parser.parse_args()
    if not 1 <= args.n_trials <= 300:
        parser.error("--n-trials must be in [1, 300]")
    trial_end = args.n_trials if args.trial_end is None else args.trial_end
    if not 0 <= args.trial_start < trial_end <= args.n_trials:
        parser.error("require 0 <= trial-start < trial-end <= n-trials")

    grouped = load_mrc_rows(args.attack_dir, args.model, args.n_trials)
    rows: list[dict] = []
    suffix = ("" if args.trial_start == 0 and trial_end == args.n_trials else
              f"_trials{args.trial_start:03d}-{trial_end - 1:03d}")
    partial = args.output_dir / f"confidence_pilot10_{args.model}{suffix}.partial.csv"
    final = args.output_dir / f"confidence_pilot10_{args.model}{suffix}.csv"

    selected_trials = range(args.trial_start, trial_end)
    for position, trial in enumerate(selected_trials, start=1):
        record_path = args.k8_dir / args.model / f"trial_{trial:03d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source_path = str(record["source_path"])
        average_probability = np.asarray(record["soft_bit_probability"], dtype=np.float64)
        append(
            rows, model=args.model, trial=trial, condition="Average",
            identity=int(record["decoded_identity"]), probability=average_probability,
            source_path=source_path, target_rank="",
            kind=str(record["soft_probability_kind"]),
        )

        source_row = grouped[trial][0]
        coalition = [int(v) for v in json.loads(source_row["coalition_payloads"])]
        single = np.asarray(
            get_or_embed(args.model, source_row["spk"], coalition[0],
                         int(source_row["local_t"])), dtype=np.float32)
        probability, kind = bit_probabilities(args.model, single)
        append(
            rows, model=args.model, trial=trial, condition="Single",
            identity=coalition[0], probability=probability,
            source_path=source_path, target_rank="", kind=kind,
        )

        hits = [row for row in grouped[trial] if int(row["target_top1"]) == 1]
        cached = {coalition[0]: single}
        for attack in hits:
            attack_coalition = [int(v) for v in json.loads(attack["coalition_payloads"])]
            if attack_coalition != coalition:
                raise RuntimeError(f"coalition mismatch for {args.model} trial={trial}")
            weights = np.asarray(json.loads(attack["weights"]), dtype=np.float64)
            waveforms = []
            for payload in coalition:
                if payload not in cached:
                    cached[payload] = np.asarray(
                        get_or_embed(args.model, attack["spk"], payload,
                                     int(attack["local_t"])), dtype=np.float32)
                waveforms.append(cached[payload])
            length = min(map(len, waveforms))
            mixture = np.asarray(sum(
                weights[i] * waveforms[i][:length] for i in range(8)
            ), dtype=np.float32)
            probability, kind = bit_probabilities(args.model, mixture)
            append(
                rows, model=args.model, trial=trial, condition="Targeted",
                identity=int(attack["target"]), probability=probability,
                source_path=source_path, target_rank=attack["target_rank"], kind=kind,
            )

        rows.sort(key=lambda row: (
            int(row["trial_id"]), {"Single": 0, "Average": 1, "Targeted": 2}[row["condition"]],
            int(row["target_rank"] or 0)))
        atomic_csv(partial, rows)
        print(
            f"checkpoint model={args.model} assigned={position}/{trial_end - args.trial_start} "
            f"trial={trial} "
            f"successful_targets={len(hits)} rows={len(rows)}", flush=True)

    atomic_csv(final, rows)
    print(f"COMPLETE model={args.model} rows={len(rows)} output={final}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect weakest decoded-bit confidence for the Section 5 mechanism figure.

For every K=8 trial, this script records three conditions:

1. a source-correct personalized copy (the first coalition member),
2. the source-correct uniform coalition average already stored by the K=8 run,
3. the highest-ranked MRC target that was hit exactly, when one exists.

Each trial contributes at most one observation per condition.  The reported
quantity is the minimum confidence assigned to any bit of the decoded identity,
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

from registry import NBITS, full_registry_bits, get_or_embed, int_to_bits  # noqa: E402
from watermarks import extract_evidence, get_wavmark  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
FIELDS = (
    "model", "K", "trial_id", "condition", "identity", "nbits",
    "min_bit_confidence", "weakest_bit", "source_path", "target_rank",
    "confidence_kind",
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


def weakest_confidence(probability: np.ndarray, identity: int, nbits: int) -> tuple[float, int]:
    bits = int_to_bits(identity, nbits).astype(np.int8)
    confidence = np.where(bits == 1, probability, 1.0 - probability)
    weakest = int(np.argmin(confidence))
    return float(confidence[weakest]), weakest


def load_attack_rows(root: Path, model: str) -> tuple[dict[int, dict], dict[int, dict]]:
    rows: list[dict] = []
    pattern = f"mrc_pm_native_mrc_{model}_K8_shard*of7.csv"
    for path in sorted((root / "shards").glob(pattern)):
        if path.name.endswith(".partial.csv"):
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 3000:
        raise RuntimeError(f"expected 3000 final MRC rows for {model}, got {len(rows)}")
    source_rows: dict[int, dict] = {}
    selected: dict[int, dict] = {}
    for row in rows:
        trial = int(row["trial_id"])
        source_rows.setdefault(trial, row)
        if int(row["target_top1"]) != 1:
            continue
        if trial not in selected or int(row["target_rank"]) < int(selected[trial]["target_rank"]):
            selected[trial] = row
    if len(source_rows) != 300:
        raise RuntimeError(f"expected 300 MRC trials for {model}, got {len(source_rows)}")
    return source_rows, selected


def load_checkpoint(path: Path) -> tuple[list[dict], set[tuple[int, str]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = {(int(row["trial_id"]), row["condition"]) for row in rows}
    return rows, complete


def append_row(rows: list[dict], complete: set[tuple[int, str]], *, model: str,
               trial: int, condition: str, identity: int, probability: np.ndarray,
               source_path: str, target_rank: str, confidence_kind: str) -> None:
    key = (trial, condition)
    if key in complete:
        return
    value, weakest = weakest_confidence(probability, identity, NBITS[model])
    rows.append({
        "model": model,
        "K": 8,
        "trial_id": trial,
        "condition": condition,
        "identity": identity,
        "nbits": NBITS[model],
        "min_bit_confidence": f"{value:.10f}",
        "weakest_bit": weakest,
        "source_path": source_path,
        "target_rank": target_rank,
        "confidence_kind": confidence_kind,
    })
    complete.add(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--k8-dir", type=Path,
        default=ROOT / "data" / "k8_population_source_correct_300clips_20260902" / "raw")
    parser.add_argument(
        "--attack-dir", type=Path,
        default=ROOT / "results" / "mrc_pm_native_shared4_fulltop10_300clips_20260904")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "identity_bit_confidence_k8_20260905")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard-id < num-shards")

    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_id}of{args.num_shards}"
    final = args.output_dir / f"identity_bit_confidence_k8_{args.model}{suffix}.csv"
    partial = args.output_dir / f"identity_bit_confidence_k8_{args.model}{suffix}.partial.csv"
    rows, complete = load_checkpoint(partial)
    assigned = [trial for trial in range(300) if trial % args.num_shards == args.shard_id]

    # Seed a sharded run from the earlier single-process checkpoint.  Filtering
    # by trial keeps every shard disjoint while preserving already computed rows.
    legacy = args.output_dir / f"identity_bit_confidence_k8_{args.model}.partial.csv"
    if args.num_shards > 1 and not rows and legacy.exists():
        legacy_rows, _ = load_checkpoint(legacy)
        assigned_set = set(assigned)
        rows = [row for row in legacy_rows if int(row["trial_id"]) in assigned_set]
        complete = {(int(row["trial_id"]), row["condition"]) for row in rows}
        if rows:
            atomic_csv(partial, rows)
            print(
                f"seeded shard={args.shard_id}/{args.num_shards} rows={len(rows)} "
                f"from={legacy.name}", flush=True)
    source_rows, successful = load_attack_rows(args.attack_dir, args.model)
    registry_bits = full_registry_bits(args.model)
    del registry_bits  # validates registry availability; identities are stored explicitly below

    for position, trial in enumerate(assigned, start=1):
        record_path = args.k8_dir / args.model / f"trial_{trial:03d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        uniform_coalition = [int(value) for value in record["coalition_payloads"]]
        source_path = str(record["source_path"])

        uniform_probability = np.asarray(record["soft_bit_probability"], dtype=np.float64)
        append_row(
            rows, complete, model=args.model, trial=trial,
            condition="uniform_collusion", identity=int(record["decoded_identity"]),
            probability=uniform_probability, source_path=source_path, target_rank="",
            confidence_kind=str(record["soft_probability_kind"]),
        )

        if (trial, "benign_copy") not in complete:
            source_row = source_rows[trial]
            coalition = [int(value) for value in json.loads(source_row["coalition_payloads"])]
            benign = np.asarray(
                get_or_embed(
                    args.model, source_row["spk"], coalition[0], int(source_row["local_t"])),
                dtype=np.float32,
            )
            probability, kind = bit_probabilities(args.model, benign)
            append_row(
                rows, complete, model=args.model, trial=trial,
                condition="benign_copy", identity=coalition[0], probability=probability,
                source_path=source_path, target_rank="", confidence_kind=kind,
            )

        attack = successful.get(trial)
        if attack is not None and (trial, "successful_mrc") not in complete:
            attack_coalition = [int(value) for value in json.loads(attack["coalition_payloads"])]
            weights = np.asarray(json.loads(attack["weights"]), dtype=np.float64)
            waveforms = [
                np.asarray(
                    get_or_embed(args.model, attack["spk"], payload, int(attack["local_t"])),
                    dtype=np.float32,
                )
                for payload in attack_coalition
            ]
            length = min(map(len, waveforms))
            attacked = np.asarray(sum(
                weights[index] * waveforms[index][:length]
                for index in range(8)
            ), dtype=np.float32)
            probability, kind = bit_probabilities(args.model, attacked)
            target = int(attack["target"])
            append_row(
                rows, complete, model=args.model, trial=trial,
                condition="successful_mrc", identity=target, probability=probability,
                source_path=source_path, target_rank=attack["target_rank"],
                confidence_kind=kind,
            )

        rows.sort(key=lambda row: (int(row["trial_id"]), row["condition"]))
        atomic_csv(partial, rows)
        if position % 5 == 0 or position == len(assigned):
            print(
                f"checkpoint model={args.model} shard={args.shard_id}/{args.num_shards} "
                f"assigned={position}/{len(assigned)} "
                f"rows={len(rows)} successful_mrc_trials={len(successful)}",
                flush=True,
            )

    atomic_csv(final, rows)
    print(
        f"COMPLETE model={args.model} shard={args.shard_id}/{args.num_shards} "
        f"rows={len(rows)} "
        f"successful_mrc_trials={len(successful)} output={final}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Payload Match and Target-Bit Margin on ten nonmembers per trial.

The two methods preserve the definitions used by the corrected paper analysis:

Payload Match selects the ten nonmember payloads closest to the coalition's
convex payload region. Target-Bit Margin selects the ten nonmember payloads whose
weakest target bit can receive the largest margin. Both methods optimize valid
mixture weights and count exact full-payload matches.

Every coalition member is first required to decode exactly to its assigned
payload.  The schedule is the clip-indexed manifest: 100 speakers x 3 distinct
clips.  Shards are isolated and atomically checkpointed after every trial.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from payload_match import (  # noqa: E402
    decoded_payload_and_margin, payload_distances, payload_match_weights,
)
from registry import (  # noqa: E402
    NBITS, CAP, coalition_seed, full_registry_bits, full_registry_size,
    int_to_bits, source_record, trial_schedule,
)
from bit_margin import score_target_task  # noqa: E402
from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE, detect_many_native, get_or_embed_native,
)
from watermarks import pesq_wb, resample_to, si_sdr, stoi  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
KS = (5, 8)
METHODS = ("payload_match", "bit_margin")
TARGET_COUNT = 10
BIT_MARGIN_BETA = 8.0
BIT_MARGIN_ENTROPY = 0.05
BIT_MARGIN_EFFECTIVE_K_FRACTION = 0.6
DATASET_TAG = "collusion_300"
SHARED_MODELS = {"audioseal", "wavmark", "voicemark", "wmcodec"}
FIELDS = [
    "dataset", "model", "k", "method", "trial_id", "speaker", "clip_index",
    "source_path", "coalition_payloads", "valid_copy_count", "payloads_tested",
    "target_rank", "target_payload", "selection_rule", "candidate_count",
    "selection_score", "payload_distance", "weights", "effective_members",
    "max_weight", "weights_valid", "decoded_payload", "target_hit",
    "target_margin", "hits_out_of_10", "escaped",
    "closest_member_bit_accuracy", "watermark_score", "decoded_bits",
    "target_bit_accuracy", "quality_reference", "pesq", "stoi", "si_sdr",
]


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint(path: Path, assigned: set[int]) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        trial = int(row["trial_id"])
        if trial in assigned:
            grouped.setdefault(trial, []).append(row)
    complete = {
        trial for trial, group in grouped.items()
        if len(group) == TARGET_COUNT
        and {int(row["target_rank"]) for row in group}
        == set(range(1, TARGET_COUNT + 1))
    }
    kept = [row for row in rows if int(row["trial_id"]) in complete]
    kept.sort(key=lambda row: (int(row["trial_id"]), int(row["target_rank"])))
    return kept, complete


def decoded_payload(hard: np.ndarray | None) -> int | None:
    if hard is None:
        return None
    return int(sum(int(value) << index for index, value in enumerate(hard)))


def score_target_block_task(task):
    """Score one target block while sending the coalition matrix only once."""
    coalition_bits, targets, nbits, beta, minimum_effective_k, entropy_weight = task
    return [
        score_target_task((coalition_bits, int(target), nbits, beta,
                           minimum_effective_k, entropy_weight))
        for target in targets
    ]


def valid_coalition(model: str, speaker: str, clip_slot: int, k: int,
                    registry_bits: np.ndarray, sample_rate: int,
                    max_attempts: int = 2000
                    ) -> tuple[list[int], list[np.ndarray], int]:
    """Deterministically draw k distinct payloads that decode exactly."""
    rng = np.random.default_rng(coalition_seed(speaker, k, clip_slot))
    selected: list[int] = []
    waveforms: list[np.ndarray] = []
    attempts = 0
    limit = full_registry_size(model)
    while len(selected) < k:
        needed = k - len(selected)
        candidates: list[int] = []
        while len(candidates) < needed:
            payload = int(rng.integers(0, limit))
            if payload not in selected and payload not in candidates:
                candidates.append(payload)
        candidate_waveforms = [
            get_or_embed_native(model, speaker, payload, clip_slot)[0]
            for payload in candidates
        ]
        decoded = detect_many_native(
            model, candidate_waveforms, sample_rate, registry_bits)
        attempts += len(candidates)
        for payload, waveform, (_, _, hard) in zip(
                candidates, candidate_waveforms, decoded):
            if decoded_payload(hard) == payload:
                selected.append(payload)
                waveforms.append(np.asarray(waveform, dtype=np.float32))
        if attempts >= max_attempts and len(selected) < k:
            raise RuntimeError(
                f"{model} {speaker} clip_slot={clip_slot}: only {len(selected)}/{k} "
                f"valid payloads after {attempts} attempts")
    return selected, waveforms, attempts


def bit_margin_targets(coalition_bits: np.ndarray, coalition: list[int], model: str,
                       pool: ProcessPoolExecutor | None
                       ) -> list[tuple[float, int, np.ndarray, bool, float]]:
    full_size = full_registry_size(model)
    candidates = np.arange(full_size, dtype=np.int64)
    candidates = candidates[
        ~np.isin(candidates, np.asarray(coalition, dtype=np.int64))]
    keff_min = max(1.0, BIT_MARGIN_EFFECTIVE_K_FRACTION * len(coalition))
    if pool is not None:
        block_size = 256
        tasks = [
            (coalition_bits, candidates[start:start + block_size].tolist(), NBITS[model],
             BIT_MARGIN_BETA, keff_min, BIT_MARGIN_ENTROPY)
            for start in range(0, len(candidates), block_size)
        ]
        scored = [
            item
            for block in pool.map(score_target_block_task, tasks, chunksize=1)
            for item in block
        ]
    else:
        scored = [
            score_target_task(
                (coalition_bits, int(target), NBITS[model], BIT_MARGIN_BETA,
                 keff_min, BIT_MARGIN_ENTROPY))
            for target in candidates.tolist()
        ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:TARGET_COUNT]
    failed = [(target, score) for score, target, _, success, _ in selected if not success]
    if failed:
        raise RuntimeError(f"Target-Bit Margin solver failure(s): {failed[:3]}")
    return selected


def payload_match_targets(coalition_bits: np.ndarray, coalition: list[int], model: str,
                          registry_bits: np.ndarray
                          ) -> list[tuple[float, int, np.ndarray, bool, float]]:
    full_size = full_registry_size(model)
    candidates = np.arange(full_size, dtype=np.int64)
    candidates = candidates[
        ~np.isin(candidates, np.asarray(coalition, dtype=np.int64))]
    distances = payload_distances(coalition_bits, registry_bits[candidates])
    order = np.lexsort((candidates, distances))[:TARGET_COUNT]
    selected = []
    for index in order:
        target = int(candidates[index])
        distance = float(distances[index])
        weights = payload_match_weights(
            coalition_bits, int_to_bits(target, NBITS[model]), CAP)
        valid = (abs(float(weights.sum()) - 1.0) <= 1e-7
                 and float(weights.min()) >= -1e-9
                 and float(weights.max()) <= CAP + 1e-7)
        if not valid:
            raise RuntimeError(
                f"Payload Match constraint failure target={target} weights={weights.tolist()}")
        selected.append((-distance, target, weights, True, distance))
    return selected


def fmt(value: float | None, digits: int = 8) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def shared_coalition_record(root: Path, k: int, trial_id: int) -> dict:
    path = root / f"k{k}" / f"trial_{trial_id:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing shared coalition manifest: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("dataset") != DATASET_TAG:
        raise RuntimeError(f"unexpected shared coalition dataset tag in {path}")
    return record


def selection_cache_path(root: Path, method: str, k: int,
                         trial_id: int) -> Path:
    return root / method / f"k{k}" / f"trial_{trial_id:03d}.json"


def load_shared_selection(path: Path, method: str, k: int,
                          coalition: list[int]) -> list | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("method") != method or int(record.get("k", -1)) != k
            or [int(value) for value in record.get("coalition_payloads", [])]
            != coalition):
        raise RuntimeError(f"shared selection cache mismatch: {path}")
    selected = []
    for item in record["targets"]:
        selected.append((
            float(item["selection_score"]), int(item["target_payload"]),
            np.asarray(item["weights"], dtype=np.float64),
            bool(item["weights_valid"]),
            (float("nan") if item["payload_distance"] is None
             else float(item["payload_distance"])),
        ))
    if len(selected) != TARGET_COUNT:
        raise RuntimeError(f"incomplete shared selection cache: {path}")
    return selected


def write_shared_selection(path: Path, method: str, k: int,
                           coalition: list[int], selected: list) -> None:
    atomic_json(path, {
        "dataset": DATASET_TAG, "method": method, "k": k,
        "coalition_payloads": coalition,
        "targets": [{
            "rank": rank, "selection_score": float(score),
            "target_payload": int(target),
            "weights": np.asarray(weights, dtype=float).tolist(),
            "weights_valid": bool(success),
            "payload_distance": (
                None if not math.isfinite(float(hull_distance))
                else float(hull_distance)),
        } for rank, (score, target, weights, success, hull_distance)
        in enumerate(selected, start=1)],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--k", choices=KS, type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=7)
    parser.add_argument("--target-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "targeted")
    parser.add_argument(
        "--coalition-dir", type=Path, default=ROOT / "results" / "coalitions")
    parser.add_argument(
        "--target-dir", type=Path, default=ROOT / "results" / "targets")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= shard-id < num-shards")
    if args.n_trials != 300:
        parser.error("this corrected dataset runner requires exactly 300 trials")

    schedule = trial_schedule(n_total=args.n_trials)
    assigned_list = [trial for trial in range(args.n_trials)
                     if trial % args.num_shards == args.shard_id]
    assigned = set(assigned_list)
    stem = f"{args.model}.shard{args.shard_id}of{args.num_shards}"
    shard_dir = args.output_dir / f"k{args.k}" / args.method / "shards"
    partial = shard_dir / f"{stem}.partial.csv"
    final = shard_dir / f"{stem}.csv"
    rows, complete = load_checkpoint(partial, assigned)
    print(
        f"resume method={args.method} model={args.model} k={args.k} "
        f"shard={args.shard_id}/{args.num_shards}: "
        f"{len(complete)}/{len(assigned_list)}",
        flush=True,
    )

    registry_bits = full_registry_bits(args.model)
    sample_rate = NATIVE_SAMPLE_RATE[args.model]
    native_ids = np.arange(full_registry_size(args.model), dtype=np.int64)
    target_pool = None
    if args.method == "bit_margin" and args.target_workers > 1:
        target_pool = ProcessPoolExecutor(
            max_workers=args.target_workers, mp_context=get_context("spawn"))
    started = time.time()
    for trial_id in assigned_list:
        if trial_id in complete:
            continue
        trial_started = time.time()
        speaker, clip_slot = schedule[trial_id]
        source = source_record(speaker, clip_slot)
        shared_record = None
        if args.model in SHARED_MODELS:
            shared_record = shared_coalition_record(
                args.coalition_dir, args.k, trial_id)
            if (shared_record["speaker"] != speaker
                    or int(shared_record["clip_index"]) != int(source["clip_index"])):
                raise RuntimeError(f"shared coalition schedule mismatch trial={trial_id}")
            coalition = [int(value) for value in shared_record["coalition_payloads"]]
            waveforms = [
                np.asarray(get_or_embed_native(
                    args.model, speaker, payload, clip_slot)[0], dtype=np.float32)
                for payload in coalition
            ]
            source_decoded = detect_many_native(
                args.model, waveforms, sample_rate, registry_bits)
            valid_copy_count = sum(
                decoded_payload(hard) == payload
                for payload, (_, _, hard) in zip(coalition, source_decoded)
            )
            if valid_copy_count != args.k:
                raise RuntimeError(
                    f"shared coalition source validation failed model={args.model} "
                    f"trial={trial_id}: {valid_copy_count}/{args.k}")
            payloads_tested = int(shared_record["payloads_tested"])
        else:
            coalition, waveforms, payloads_tested = valid_coalition(
                args.model, speaker, clip_slot, args.k, registry_bits,
                sample_rate)
        length = min(map(len, waveforms))
        waveforms = [waveform[:length] for waveform in waveforms]
        coalition_bits = np.stack([
            int_to_bits(payload, NBITS[args.model]) for payload in coalition
        ])

        cache_path = None
        selected = None
        if shared_record is not None:
            cache_path = selection_cache_path(
                args.target_dir, args.method, args.k, trial_id)
            selected = load_shared_selection(
                cache_path, args.method, args.k, coalition)
        if args.method == "bit_margin":
            if selected is None:
                selected = bit_margin_targets(
                    coalition_bits, coalition, args.model, target_pool)
            selection_policy = "largest_minimum_bit_margin"
            selection_n = full_registry_size(args.model) - args.k
        else:
            if selected is None:
                selected = payload_match_targets(
                    coalition_bits, coalition, args.model, registry_bits)
            selection_policy = "nearest_nonmember_payloads"
            selection_n = full_registry_size(args.model) - args.k
        if cache_path is not None and not cache_path.exists():
            write_shared_selection(
                cache_path, args.method, args.k, coalition, selected)

        attacked_waveforms = [
            np.asarray(sum(weights[member] * waveforms[member]
                           for member in range(args.k)), dtype=np.float32)
            for _, _, weights, _, _ in selected
        ]
        decoded = detect_many_native(
            args.model, attacked_waveforms, sample_rate, registry_bits)
        reference = waveforms[0]
        reference_16 = (reference if sample_rate == 16000 else
                        resample_to(reference, sample_rate, 16000))
        trial_rows: list[dict] = []
        decoded_records = []
        for selected_item, attacked, decoded_item in zip(
                selected, attacked_waveforms, decoded):
            score, target, weights, success, hull_distance = selected_item
            native_scores, presence, hard = decoded_item
            decoded_payload, margin = decoded_payload_and_margin(
                native_scores, native_ids, target)
            decoded_records.append((
                score, target, weights, success, hull_distance, attacked,
                native_scores, presence, hard, decoded_payload, margin,
            ))
        hits = sum(
            int(decoded_payload == target)
            for _, target, _, _, _, _, _, _, _, decoded_payload, _
            in decoded_records)

        for rank, ((score, target, weights, success, hull_distance, attacked,
                    native_scores, presence, hard, decoded_payload, margin)) in enumerate(
                       decoded_records, start=1):
            hard_array = (None if hard is None
                          else np.asarray(hard, dtype=np.int8))
            target_bits = int_to_bits(target, NBITS[args.model])
            closest_member_accuracy = (
                float("nan") if hard_array is None else
                float(max(np.mean(hard_array == member_bits)
                          for member_bits in coalition_bits)))
            target_accuracy = (float("nan") if hard_array is None else
                               float(np.mean(hard_array == target_bits)))
            effective_k = 1.0 / float(np.sum(weights ** 2))
            attacked_16 = (attacked if sample_rate == 16000 else
                           resample_to(attacked, sample_rate, 16000))
            trial_rows.append({
                "dataset": (shared_record["dataset"]
                            if shared_record is not None else DATASET_TAG),
                "model": args.model, "k": args.k,
                "method": args.method, "trial_id": trial_id,
                "speaker": speaker, "clip_index": int(source["clip_index"]),
                "source_path": source["path"],
                "coalition_payloads": json.dumps(coalition),
                "valid_copy_count": args.k, "payloads_tested": payloads_tested,
                "target_rank": rank, "target_payload": target,
                "selection_rule": selection_policy,
                "candidate_count": selection_n,
                "selection_score": fmt(score),
                "payload_distance": fmt(hull_distance),
                "weights": json.dumps(np.asarray(weights).tolist()),
                "effective_members": fmt(effective_k),
                "max_weight": fmt(float(np.max(weights))),
                "weights_valid": int(success),
                "decoded_payload": decoded_payload,
                "target_hit": int(decoded_payload == target),
                "target_margin": fmt(margin), "hits_out_of_10": hits,
                "escaped": int(decoded_payload not in coalition),
                "closest_member_bit_accuracy": fmt(closest_member_accuracy),
                "watermark_score": fmt(presence, 6),
                "decoded_bits": ("" if hard_array is None else
                                 json.dumps(hard_array.astype(int).tolist())),
                "target_bit_accuracy": fmt(target_accuracy),
                "quality_reference": "first_coalition_copy",
                "pesq": fmt(float(pesq_wb(reference_16, attacked_16)), 4),
                "stoi": fmt(float(stoi(reference_16, attacked_16)), 4),
                "si_sdr": fmt(float(si_sdr(reference_16, attacked_16)), 2),
            })
        elapsed = time.time() - trial_started
        rows.extend(trial_rows)
        complete.add(trial_id)
        rows.sort(key=lambda row: (int(row["trial_id"]), int(row["target_rank"])))
        atomic_csv(partial, rows)
        print(
            f"checkpoint method={args.method} model={args.model} k={args.k} "
            f"shard={args.shard_id}: {len(complete)}/{len(assigned_list)} "
            f"trial={trial_id} elapsed={elapsed:.1f}s",
            flush=True,
        )

    if target_pool is not None:
        target_pool.shutdown(wait=True)
    if complete != assigned:
        raise RuntimeError(
            f"incomplete shard: {len(complete)}/{len(assigned)} trials")
    atomic_csv(final, rows)
    print(
        f"COMPLETE {final} trials={len(complete)} rows={len(rows)} "
        f"wall={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run source-correct MRC or PM on the 300-distinct-clip dataset.

The two methods preserve the definitions used by the corrected paper analysis:

* MRC: beta=8, entropy gamma=0.05, and K_eff >= 0.6 K. Every native
  non-member identity is scored by the MRC objective and the ten highest
  scoring identities are selected.
* PM: payload matching with a simplex constraint and a 0.5 per-weight cap.
  Every native non-member identity is screened by exact convex-hull distance
  and the globally closest ten identities are selected.

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

from framing import (  # noqa: E402
    convex_dist_batch_exact, restricted_top1_and_margin, tct,
)
from registry import (  # noqa: E402
    NBITS, CAP, coalition_seed, full_registry_bits, full_registry_size,
    get_or_embed, int_to_bits, source_record, speaker_trial_index,
)
from mrc_solver import optimize_softmin, score_target_task  # noqa: E402
from watermarks import detect_many, pesq_wb, si_sdr, stoi  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
KS = (2, 3, 5, 8)
METHODS = ("mrc", "pm")
N_TARGETS = 10
MRC_BETA = 8.0
MRC_ENTROPY = 0.05
MRC_KEFF_FRAC = 0.6
DATASET_TAG = "collusion_300_manifest_clip_indexed_v20"
SHARED_DATASET_TAG = "collusion_300_shared4_source_correct_v21"
SHARED_PENDING_DATASET_TAG = (
    "collusion_300_shared3_source_correct_wavmark_pending_v21")
SHARED_MODELS = {"audioseal", "wavmark", "voicemark", "wmcodec"}
FIELDS = [
    "dataset", "model", "K", "method", "trial_id", "spk", "local_t",
    "clip_index", "source_path", "shard_id", "num_shards",
    "coalition_payloads", "source_exact_count", "source_payload_attempts",
    "target_rank", "target", "target_selection_policy",
    "selection_N_registry", "N_registry", "selection_score",
    "hull_distance", "weights", "effective_K", "max_weight",
    "solver_success", "decoded_identity", "target_top1", "target_margin",
    "hits_selected_10",
    "tracing_failure", "NCA", "presence", "decoded_bits",
    "target_bit_accuracy", "quality_reference", "PESQ", "STOI", "SI_SDR",
    "elapsed_sec",
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
        if len(group) == N_TARGETS
        and {int(row["target_rank"]) for row in group} == set(range(1, N_TARGETS + 1))
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
    C, targets, nbits, beta, keff_min, entropy_gamma = task
    return [
        score_target_task((C, int(target), nbits, beta, keff_min, entropy_gamma))
        for target in targets
    ]


def source_correct_coalition(model: str, spk: str, local_t: int, K: int,
                             registry_bits: np.ndarray,
                             max_attempts: int = 2000
                             ) -> tuple[list[int], list[np.ndarray], int]:
    """Deterministically draw K distinct payloads that decode exactly."""
    rng = np.random.default_rng(coalition_seed(spk, K, local_t))
    selected: list[int] = []
    waveforms: list[np.ndarray] = []
    attempts = 0
    limit = full_registry_size(model)
    while len(selected) < K:
        needed = K - len(selected)
        candidates: list[int] = []
        while len(candidates) < needed:
            payload = int(rng.integers(0, limit))
            if payload not in selected and payload not in candidates:
                candidates.append(payload)
        candidate_waveforms = [
            get_or_embed(model, spk, payload, local_t) for payload in candidates
        ]
        decoded = detect_many(model, candidate_waveforms, registry_bits)
        attempts += len(candidates)
        for payload, waveform, (_, _, hard) in zip(
                candidates, candidate_waveforms, decoded):
            if decoded_payload(hard) == payload:
                selected.append(payload)
                waveforms.append(np.asarray(waveform, dtype=np.float32))
        if attempts >= max_attempts and len(selected) < K:
            raise RuntimeError(
                f"{model} {spk} local_t={local_t}: only {len(selected)}/{K} "
                f"source-correct payloads after {attempts} attempts")
    return selected, waveforms, attempts


def mrc_targets(C: np.ndarray, coalition: list[int], model: str, spk: str,
                local_t: int, pool: ProcessPoolExecutor | None
                ) -> list[tuple[float, int, np.ndarray, bool, float]]:
    full_size = full_registry_size(model)
    candidates = np.arange(full_size, dtype=np.int64)
    candidates = candidates[
        ~np.isin(candidates, np.asarray(coalition, dtype=np.int64))]
    keff_min = max(1.0, MRC_KEFF_FRAC * len(coalition))
    if pool is not None:
        block_size = 256
        tasks = [
            (C, candidates[start:start + block_size].tolist(), NBITS[model],
             MRC_BETA, keff_min, MRC_ENTROPY)
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
                (C, int(target), NBITS[model], MRC_BETA, keff_min, MRC_ENTROPY))
            for target in candidates.tolist()
        ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:N_TARGETS]
    failed = [(target, score) for score, target, _, success, _ in selected if not success]
    if failed:
        raise RuntimeError(f"MRC solver failure(s): {failed[:3]}")
    return selected


def pm_targets(C: np.ndarray, coalition: list[int], model: str, spk: str,
               local_t: int, registry_bits: np.ndarray
               ) -> list[tuple[float, int, np.ndarray, bool, float]]:
    full_size = full_registry_size(model)
    candidates = np.arange(full_size, dtype=np.int64)
    candidates = candidates[
        ~np.isin(candidates, np.asarray(coalition, dtype=np.int64))]
    distances = convex_dist_batch_exact(C, registry_bits[candidates])
    order = np.lexsort((candidates, distances))[:N_TARGETS]
    selected = []
    for index in order:
        target = int(candidates[index])
        distance = float(distances[index])
        weights = np.asarray(tct(C, int_to_bits(target, NBITS[model]), CAP),
                             dtype=np.float64)
        valid = (abs(float(weights.sum()) - 1.0) <= 1e-7
                 and float(weights.min()) >= -1e-9
                 and float(weights.max()) <= CAP + 1e-7)
        if not valid:
            raise RuntimeError(
                f"PM constraint failure target={target} weights={weights.tolist()}")
        selected.append((-distance, target, weights, True, distance))
    return selected


def fmt(value: float | None, digits: int = 8) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def shared_coalition_record(root: Path, K: int, trial_id: int) -> dict:
    path = root / "final" / f"K{K}" / f"trial_{trial_id:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing shared coalition manifest: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("dataset") not in {
            SHARED_DATASET_TAG, SHARED_PENDING_DATASET_TAG}:
        raise RuntimeError(f"unexpected shared coalition dataset tag in {path}")
    return record


def selection_cache_path(root: Path, method: str, K: int,
                         trial_id: int) -> Path:
    return root / "selection_cache" / method / f"K{K}" / f"trial_{trial_id:03d}.json"


def load_shared_selection(path: Path, method: str, K: int,
                          coalition: list[int]) -> list | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("method") != method or int(record.get("K", -1)) != K
            or [int(value) for value in record.get("coalition_payloads", [])]
            != coalition):
        raise RuntimeError(f"shared selection cache mismatch: {path}")
    selected = []
    for item in record["selected"]:
        selected.append((
            float(item["score"]), int(item["target"]),
            np.asarray(item["weights"], dtype=np.float64),
            bool(item["success"]),
            (float("nan") if item["hull_distance"] is None
             else float(item["hull_distance"])),
        ))
    if len(selected) != N_TARGETS:
        raise RuntimeError(f"incomplete shared selection cache: {path}")
    return selected


def write_shared_selection(path: Path, method: str, K: int,
                           coalition: list[int], selected: list) -> None:
    atomic_json(path, {
        "dataset": SHARED_DATASET_TAG, "method": method, "K": K,
        "coalition_payloads": coalition,
        "selected": [{
            "score": float(score), "target": int(target),
            "weights": np.asarray(weights, dtype=float).tolist(),
            "success": bool(success),
            "hull_distance": (None if not math.isfinite(float(hull_distance))
                              else float(hull_distance)),
        } for score, target, weights, success, hull_distance in selected],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--K", choices=KS, type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=7)
    parser.add_argument("--target-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "mrc_pm_native_fulltop10_300clips_20260904")
    parser.add_argument("--shared-coalition-dir", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= shard-id < num-shards")
    if args.n_trials != 300:
        parser.error("this corrected dataset runner requires exactly 300 trials")

    schedule = speaker_trial_index(n_total=args.n_trials)
    assigned_list = [trial for trial in range(args.n_trials)
                     if trial % args.num_shards == args.shard_id]
    assigned = set(assigned_list)
    stem = (f"mrc_pm_native_{args.method}_{args.model}_K{args.K}_"
            f"shard{args.shard_id}of{args.num_shards}")
    shard_dir = args.output_dir / "shards"
    partial = shard_dir / f"{stem}.partial.csv"
    final = shard_dir / f"{stem}.csv"
    rows, complete = load_checkpoint(partial, assigned)
    print(
        f"resume method={args.method} model={args.model} K={args.K} "
        f"shard={args.shard_id}/{args.num_shards}: "
        f"{len(complete)}/{len(assigned_list)}",
        flush=True,
    )

    registry_bits = full_registry_bits(args.model)
    native_ids = np.arange(full_registry_size(args.model), dtype=np.int64)
    target_pool = None
    if args.method == "mrc" and args.target_workers > 1:
        target_pool = ProcessPoolExecutor(
            max_workers=args.target_workers, mp_context=get_context("spawn"))
    started = time.time()
    for trial_id in assigned_list:
        if trial_id in complete:
            continue
        trial_started = time.time()
        spk, local_t = schedule[trial_id]
        source = source_record(spk, local_t)
        shared_record = None
        if args.model in SHARED_MODELS:
            if args.shared_coalition_dir is None:
                raise RuntimeError(
                    f"{args.model} requires --shared-coalition-dir under the shared4 protocol")
            shared_record = shared_coalition_record(
                args.shared_coalition_dir, args.K, trial_id)
            if (shared_record["spk"] != spk
                    or int(shared_record["local_t"]) != local_t):
                raise RuntimeError(f"shared coalition schedule mismatch trial={trial_id}")
            coalition = [int(value) for value in shared_record["coalition_payloads"]]
            waveforms = [
                np.asarray(get_or_embed(args.model, spk, payload, local_t),
                           dtype=np.float32)
                for payload in coalition
            ]
            source_decoded = detect_many(args.model, waveforms, registry_bits)
            source_exact = sum(
                decoded_payload(hard) == payload
                for payload, (_, _, hard) in zip(coalition, source_decoded)
            )
            if source_exact != args.K:
                raise RuntimeError(
                    f"shared coalition source validation failed model={args.model} "
                    f"trial={trial_id}: {source_exact}/{args.K}")
            source_attempts = int(shared_record["candidate_count_screened"])
        else:
            coalition, waveforms, source_attempts = source_correct_coalition(
                args.model, spk, local_t, args.K, registry_bits)
        length = min(map(len, waveforms))
        waveforms = [waveform[:length] for waveform in waveforms]
        C = np.stack([
            int_to_bits(payload, NBITS[args.model]) for payload in coalition
        ])

        cache_path = None
        selected = None
        if shared_record is not None:
            cache_path = selection_cache_path(
                args.shared_coalition_dir, args.method, args.K, trial_id)
            selected = load_shared_selection(
                cache_path, args.method, args.K, coalition)
        if args.method == "mrc":
            if selected is None:
                selected = mrc_targets(C, coalition, args.model, spk, local_t,
                                       target_pool)
            selection_policy = "mrc_global_top10_over_full_native_registry"
            selection_n = full_registry_size(args.model) - args.K
        else:
            if selected is None:
                selected = pm_targets(C, coalition, args.model, spk, local_t,
                                      registry_bits)
            selection_policy = "pm_global_convex_distance_top10_over_full_native_registry"
            selection_n = full_registry_size(args.model) - args.K
        if cache_path is not None and not cache_path.exists():
            write_shared_selection(
                cache_path, args.method, args.K, coalition, selected)

        attacked_waveforms = [
            np.asarray(sum(weights[member] * waveforms[member]
                           for member in range(args.K)), dtype=np.float32)
            for _, _, weights, _, _ in selected
        ]
        decoded = detect_many(args.model, attacked_waveforms, registry_bits)
        reference = waveforms[0]
        trial_rows: list[dict] = []
        decoded_records = []
        for selected_item, attacked, decoded_item in zip(
                selected, attacked_waveforms, decoded):
            score, target, weights, success, hull_distance = selected_item
            native_scores, presence, hard = decoded_item
            identity, margin = restricted_top1_and_margin(
                native_scores, native_ids, target)
            decoded_records.append((
                score, target, weights, success, hull_distance, attacked,
                native_scores, presence, hard, identity, margin,
            ))
        hits_selected_10 = sum(
            int(identity == target)
            for _, target, _, _, _, _, _, _, _, identity, _ in decoded_records)

        for rank, ((score, target, weights, success, hull_distance, attacked,
                    native_scores, presence, hard, identity, margin)) in enumerate(
                       decoded_records, start=1):
            hard_array = (None if hard is None
                          else np.asarray(hard, dtype=np.int8))
            target_bits = int_to_bits(target, NBITS[args.model])
            nca = (float("nan") if hard_array is None else
                   float(max(np.mean(hard_array == member_bits)
                             for member_bits in C)))
            target_accuracy = (float("nan") if hard_array is None else
                               float(np.mean(hard_array == target_bits)))
            effective_k = 1.0 / float(np.sum(weights ** 2))
            trial_rows.append({
                "dataset": (shared_record["dataset"]
                            if shared_record is not None else DATASET_TAG),
                "model": args.model, "K": args.K,
                "method": args.method, "trial_id": trial_id, "spk": spk,
                "local_t": local_t, "clip_index": int(source["clip_index"]),
                "source_path": source["path"], "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "coalition_payloads": json.dumps(coalition),
                "source_exact_count": args.K,
                "source_payload_attempts": source_attempts,
                "target_rank": rank, "target": target,
                "target_selection_policy": selection_policy,
                "selection_N_registry": selection_n,
                "N_registry": full_registry_size(args.model),
                "selection_score": fmt(score),
                "hull_distance": fmt(hull_distance),
                "weights": json.dumps(np.asarray(weights).tolist()),
                "effective_K": fmt(effective_k),
                "max_weight": fmt(float(np.max(weights))),
                "solver_success": int(success),
                "decoded_identity": identity,
                "target_top1": int(identity == target),
                "target_margin": fmt(margin),
                "hits_selected_10": hits_selected_10,
                "tracing_failure": int(identity not in coalition),
                "NCA": fmt(nca), "presence": fmt(presence, 6),
                "decoded_bits": ("" if hard_array is None else
                                 json.dumps(hard_array.astype(int).tolist())),
                "target_bit_accuracy": fmt(target_accuracy),
                "quality_reference": "first_source_correct_coalition_copy",
                "PESQ": fmt(float(pesq_wb(reference, attacked)), 4),
                "STOI": fmt(float(stoi(reference, attacked)), 4),
                "SI_SDR": fmt(float(si_sdr(reference, attacked)), 2),
                "elapsed_sec": "",
            })
        elapsed = time.time() - trial_started
        for row in trial_rows:
            row["elapsed_sec"] = f"{elapsed:.3f}"
        rows.extend(trial_rows)
        complete.add(trial_id)
        rows.sort(key=lambda row: (int(row["trial_id"]), int(row["target_rank"])))
        atomic_csv(partial, rows)
        print(
            f"checkpoint method={args.method} model={args.model} K={args.K} "
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

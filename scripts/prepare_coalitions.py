#!/usr/bin/env python3
"""Build valid coalitions shared by the four 16-bit systems.

VoiceMark and WMCodec screen a deterministic 64-payload pool. Their common
valid candidates form a smaller candidate pool. AudioSeal and WavMark then
validate that pool, and the first k payloads correct in all four
systems become the shared coalition.  Selection never uses mixture outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from registry import (  # noqa: E402
    coalition_seed, full_registry_bits, get_or_embed, source_record,
    trial_schedule,
)
from watermarks import detect_many  # noqa: E402


MODELS = ("audioseal", "wavmark", "voicemark", "wmcodec")
KS = (5, 8)
DEFAULT_OUT = ROOT / "results" / "coalitions"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def payload_from_hard(hard) -> int | None:
    if hard is None:
        return None
    return int(sum(int(bit) << index for index, bit in enumerate(hard)))


def candidate_payloads(speaker: str, clip_slot: int, k: int, count: int) -> list[int]:
    rng = np.random.default_rng(coalition_seed(speaker, k, clip_slot))
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        payload = int(rng.integers(0, 1 << 16))
        if payload not in seen:
            seen.add(payload)
            values.append(payload)
    return values


def record_path(out: Path, section: str, k: int, trial_id: int,
                model: str | None = None) -> Path:
    if section == "coalition":
        return out / f"k{k}" / f"trial_{trial_id:03d}.json"
    if model is None:
        return out / "work" / section / f"k{k}" / f"trial_{trial_id:03d}.json"
    return out / "work" / section / model / f"k{k}" / f"trial_{trial_id:03d}.json"


def expansion_request_path(out: Path, k: int, trial_id: int) -> Path:
    return out / "work" / "incomplete" / f"k{k}" / f"trial_{trial_id:03d}.json"


def validate_model(args, schedule) -> None:
    registry = full_registry_bits(args.model)
    done = 0
    for trial_id, (speaker, clip_slot) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
            continue
        if (args.retry_incomplete
                and not expansion_request_path(
                    args.output_dir, args.k, trial_id).exists()):
            continue
        final = record_path(args.output_dir, "validation", args.k, trial_id,
                            args.model)
        if args.pool == "full":
            payloads = candidate_payloads(speaker, clip_slot, args.k,
                                          args.candidate_count)
        else:
            candidate_record = json.loads(record_path(
                args.output_dir, "candidates", args.k, trial_id).read_text())
            payloads = [int(value) for value in candidate_record["payloads"]]
        if final.exists():
            existing = json.loads(final.read_text(encoding="utf-8"))
            existing_payloads = [
                int(value) for value in existing.get("payloads", [])]
            pool_matches = existing.get("pool") == args.pool
            payloads_match = existing_payloads == payloads
            if args.pool == "full":
                payloads_match = (
                    len(existing_payloads) >= len(payloads)
                    and existing_payloads[:len(payloads)] == payloads)
            if pool_matches and payloads_match:
                done += 1
                continue
        waveforms = [get_or_embed(args.model, speaker, payload, clip_slot)
                     for payload in payloads]
        decoded = detect_many(args.model, waveforms, registry)
        results = []
        for payload, (_, watermark_score, hard) in zip(payloads, decoded):
            got = payload_from_hard(hard)
            results.append({
                "payload": payload,
                "decoded_payload": got,
                "valid": int(got == payload),
                "watermark_score": (
                    None if not np.isfinite(watermark_score)
                    else float(watermark_score)),
            })
        source = source_record(speaker, clip_slot)
        atomic_json(final, {
            "dataset": "collusion_300",
            "stage": "copy_validation",
            "model": args.model, "k": args.k, "trial_id": trial_id,
            "speaker": speaker, "clip_slot": clip_slot,
            "clip_index": int(source["clip_index"]),
            "source_path": source["path"], "pool": args.pool,
            "candidate_count": len(payloads), "payloads": payloads,
            "results": results,
            "valid_count": sum(item["valid"] for item in results),
        })
        done += 1
        print(f"validate model={args.model} k={args.k} shard={args.shard_id} "
              f"{done} trial={trial_id} valid={sum(x['valid'] for x in results)}/"
              f"{len(results)}", flush=True)


def screen_candidates(args, schedule) -> None:
    done = 0
    for trial_id, (speaker, clip_slot) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
            continue
        final = record_path(args.output_dir, "candidates", args.k, trial_id)
        if final.exists():
            done += 1
            continue
        records = [json.loads(record_path(
            args.output_dir, "validation", args.k, trial_id, model).read_text())
            for model in ("voicemark", "wmcodec")]
        payloads = records[0]["payloads"]
        if any(record["payloads"] != payloads for record in records[1:]):
            raise RuntimeError(f"trial={trial_id}: candidate sequence mismatch")
        exact_sets = [
            {int(item["payload"]) for item in record["results"] if item["valid"]}
            for record in records
        ]
        common = set.intersection(*exact_sets)
        selected = [int(payload) for payload in payloads if payload in common]
        need = args.k + args.reserve
        if len(selected) < need:
            message = (f"trial={trial_id}: only {len(selected)} common "
                       f"VoiceMark/WMCodec payloads; need {need}; "
                       "increase --candidate-count")
            if not args.skip_incomplete:
                raise RuntimeError(message)
            atomic_json(expansion_request_path(
                args.output_dir, args.k, trial_id), {
                    "k": args.k, "trial_id": trial_id, "speaker": speaker,
                    "clip_slot": clip_slot, "common": len(selected), "need": need,
                    "candidate_count": len(payloads),
                })
            print(f"defer {message}", flush=True)
            continue
        selected = selected[:need]
        source = source_record(speaker, clip_slot)
        atomic_json(final, {
            "dataset": "collusion_300",
            "stage": "candidate_screening", "k": args.k,
            "trial_id": trial_id, "speaker": speaker, "clip_slot": clip_slot,
            "clip_index": int(source["clip_index"]),
            "source_path": source["path"], "payloads": selected,
            "screened_candidates": len(payloads),
            "valid_in_voicemark_and_wmcodec": len(common),
        })
        done += 1
        print(f"candidates k={args.k} shard={args.shard_id} {done} "
              f"trial={trial_id} common={len(common)} keep={len(selected)}",
              flush=True)


def finalize(args, schedule) -> None:
    done = 0
    for trial_id, (speaker, clip_slot) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
            continue
        final = record_path(args.output_dir, "coalition", args.k, trial_id)
        if final.exists():
            done += 1
            continue
        candidate_record = json.loads(record_path(
            args.output_dir, "candidates", args.k, trial_id).read_text())
        ordered = [int(value) for value in candidate_record["payloads"]]
        validation_models = MODELS
        valid_by_model = {}
        for model in validation_models:
            record = json.loads(record_path(
                args.output_dir, "validation", args.k, trial_id, model).read_text())
            valid_by_model[model] = {
                int(item["payload"]) for item in record["results"] if item["valid"]
            }
        common = set.intersection(*(
            valid_by_model[model] for model in validation_models))
        coalition = [payload for payload in ordered if payload in common][:args.k]
        if len(coalition) != args.k:
            raise RuntimeError(
                f"trial={trial_id}: only {len(coalition)}/{args.k} payloads "
                "decode exactly in all four systems")
        source = source_record(speaker, clip_slot)
        atomic_json(final, {
            "dataset": "collusion_300",
            "selection_rule": "first payloads valid in all shared models",
            "shared_models": list(validation_models), "k": args.k,
            "trial_id": trial_id,
            "speaker": speaker,
            "clip_index": int(source["clip_index"]),
            "source_path": source["path"],
            "coalition_payloads": coalition,
            "valid_copy_count": {
                model: args.k for model in validation_models},
            "payloads_tested": candidate_record["screened_candidates"],
            "candidate_pool_size": len(ordered),
        })
        done += 1
        print(f"final k={args.k} shard={args.shard_id} {done} "
              f"trial={trial_id} coalition={coalition}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("validate", "screen", "finalize"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--pool", choices=("full", "candidates"),
                        default="full")
    parser.add_argument("--k", type=int, required=True, choices=KS)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=7)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--extra-candidates", dest="reserve", type=int, default=4)
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.n_trials != 300:
        parser.error("shared-coalition protocol requires exactly 300 trials")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= shard-id < num-shards")
    if args.stage == "validate" and args.model is None:
        parser.error("--model is required for validate")
    schedule = trial_schedule(args.n_trials)
    if args.stage == "validate":
        validate_model(args, schedule)
    elif args.stage == "screen":
        screen_candidates(args, schedule)
    else:
        finalize(args, schedule)


if __name__ == "__main__":
    main()

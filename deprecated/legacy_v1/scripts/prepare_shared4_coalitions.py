#!/usr/bin/env python3
"""Build source-correct coalitions shared by the four 16-bit systems.

VoiceMark and WMCodec screen a deterministic 64-payload pool.  Their common
valid candidates form a small preliminary pool.  AudioSeal and WavMark then
validate that preliminary pool, and the first K payloads correct in all four
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
    speaker_trial_index,
)
from watermarks import detect_many  # noqa: E402


MODELS = ("audioseal", "wavmark", "voicemark", "wmcodec")
KS = (5, 8)
DEFAULT_OUT = ROOT / "data" / "shared4_coalitions_300clips_20260904"


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


def candidate_payloads(spk: str, local_t: int, K: int, count: int) -> list[int]:
    rng = np.random.default_rng(coalition_seed(spk, K, local_t))
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        payload = int(rng.integers(0, 1 << 16))
        if payload not in seen:
            seen.add(payload)
            values.append(payload)
    return values


def record_path(out: Path, section: str, K: int, trial_id: int,
                model: str | None = None) -> Path:
    if model is None:
        return out / section / f"K{K}" / f"trial_{trial_id:03d}.json"
    return out / section / model / f"K{K}" / f"trial_{trial_id:03d}.json"


def expansion_request_path(out: Path, K: int, trial_id: int) -> Path:
    return out / "expansion_requests" / f"K{K}" / f"trial_{trial_id:03d}.json"


def validate_model(args, schedule) -> None:
    registry = full_registry_bits(args.model)
    done = 0
    for trial_id, (spk, local_t) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
            continue
        if (args.only_expansion_requests
                and not expansion_request_path(
                    args.output_dir, args.K, trial_id).exists()):
            continue
        final = record_path(args.output_dir, "validation", args.K, trial_id,
                            args.model)
        if args.pool == "full":
            payloads = candidate_payloads(spk, local_t, args.K,
                                          args.candidate_count)
        else:
            preliminary = json.loads(record_path(
                args.output_dir, "preliminary", args.K, trial_id).read_text())
            payloads = [int(value) for value in preliminary["payloads"]]
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
        waveforms = [get_or_embed(args.model, spk, payload, local_t)
                     for payload in payloads]
        decoded = detect_many(args.model, waveforms, registry)
        results = []
        for payload, (_, presence, hard) in zip(payloads, decoded):
            got = payload_from_hard(hard)
            results.append({
                "payload": payload,
                "decoded_payload": got,
                "exact": int(got == payload),
                "presence": (None if not np.isfinite(presence)
                             else float(presence)),
            })
        source = source_record(spk, local_t)
        atomic_json(final, {
            "dataset": "collusion_300_manifest_clip_indexed_v20",
            "stage": "shared4_source_validation",
            "model": args.model, "K": args.K, "trial_id": trial_id,
            "spk": spk, "local_t": local_t,
            "clip_index": int(source["clip_index"]),
            "source_path": source["path"], "pool": args.pool,
            "candidate_count": len(payloads), "payloads": payloads,
            "results": results,
            "exact_count": sum(item["exact"] for item in results),
        })
        done += 1
        print(f"validate model={args.model} K={args.K} shard={args.shard_id} "
              f"{done} trial={trial_id} exact={sum(x['exact'] for x in results)}/"
              f"{len(results)}", flush=True)


def preliminary(args, schedule) -> None:
    done = 0
    for trial_id, (spk, local_t) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
            continue
        final = record_path(args.output_dir, "preliminary", args.K, trial_id)
        if final.exists():
            done += 1
            continue
        records = [json.loads(record_path(
            args.output_dir, "validation", args.K, trial_id, model).read_text())
            for model in ("voicemark", "wmcodec")]
        payloads = records[0]["payloads"]
        if any(record["payloads"] != payloads for record in records[1:]):
            raise RuntimeError(f"trial={trial_id}: candidate sequence mismatch")
        exact_sets = [
            {int(item["payload"]) for item in record["results"] if item["exact"]}
            for record in records
        ]
        common = set.intersection(*exact_sets)
        selected = [int(payload) for payload in payloads if payload in common]
        need = args.K + args.reserve
        if len(selected) < need:
            message = (f"trial={trial_id}: only {len(selected)} common "
                       f"VoiceMark/WMCodec payloads; need {need}; "
                       "increase --candidate-count")
            if not args.defer_insufficient:
                raise RuntimeError(message)
            atomic_json(expansion_request_path(
                args.output_dir, args.K, trial_id), {
                    "K": args.K, "trial_id": trial_id, "spk": spk,
                    "local_t": local_t, "common": len(selected), "need": need,
                    "candidate_count": len(payloads),
                })
            print(f"defer {message}", flush=True)
            continue
        selected = selected[:need]
        source = source_record(spk, local_t)
        atomic_json(final, {
            "dataset": "collusion_300_manifest_clip_indexed_v20",
            "stage": "shared4_preliminary", "K": args.K,
            "trial_id": trial_id, "spk": spk, "local_t": local_t,
            "clip_index": int(source["clip_index"]),
            "source_path": source["path"], "payloads": selected,
            "screened_candidates": len(payloads),
            "joint_exact_voicemark_wmcodec": len(common),
        })
        done += 1
        print(f"preliminary K={args.K} shard={args.shard_id} {done} "
              f"trial={trial_id} common={len(common)} keep={len(selected)}",
              flush=True)


def finalize(args, schedule) -> None:
    done = 0
    for trial_id, (spk, local_t) in enumerate(schedule):
        if trial_id % args.num_shards != args.shard_id:
            continue
        final = record_path(args.output_dir, "final", args.K, trial_id)
        if final.exists():
            done += 1
            continue
        preliminary_record = json.loads(record_path(
            args.output_dir, "preliminary", args.K, trial_id).read_text())
        ordered = [int(value) for value in preliminary_record["payloads"]]
        validation_models = (MODELS if not args.defer_wavmark_validation
                             else tuple(model for model in MODELS
                                        if model != "wavmark"))
        valid_by_model = {}
        for model in validation_models:
            record = json.loads(record_path(
                args.output_dir, "validation", args.K, trial_id, model).read_text())
            valid_by_model[model] = {
                int(item["payload"]) for item in record["results"] if item["exact"]
            }
        common = set.intersection(*(
            valid_by_model[model] for model in validation_models))
        coalition = [payload for payload in ordered if payload in common][:args.K]
        if len(coalition) != args.K:
            raise RuntimeError(
                f"trial={trial_id}: only {len(coalition)}/{args.K} payloads "
                "decode exactly in all four systems")
        source = source_record(spk, local_t)
        atomic_json(final, {
            "dataset": ("collusion_300_shared3_source_correct_wavmark_pending_v21"
                        if args.defer_wavmark_validation
                        else "collusion_300_shared4_source_correct_v21"),
            "selection": (
                "first deterministic candidates decoding exactly in AudioSeal, VoiceMark, and WMCodec; WavMark validation deferred"
                if args.defer_wavmark_validation else
                "first deterministic candidates decoding exactly in all four 16-bit systems"),
            "models": list(validation_models), "K": args.K,
            "trial_id": trial_id,
            "spk": spk, "local_t": local_t,
            "clip_index": int(source["clip_index"]),
            "source_path": source["path"],
            "coalition_payloads": coalition,
            "source_exact_count_by_model": {
                model: args.K for model in validation_models},
            "pending_validation_models": (
                ["wavmark"] if args.defer_wavmark_validation else []),
            "candidate_count_screened": preliminary_record["screened_candidates"],
            "preliminary_pool_size": len(ordered),
        })
        done += 1
        print(f"final K={args.K} shard={args.shard_id} {done} "
              f"trial={trial_id} coalition={coalition}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("validate", "preliminary", "finalize"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--pool", choices=("full", "preliminary"),
                        default="full")
    parser.add_argument("--K", type=int, required=True, choices=KS)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=7)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--reserve", type=int, default=4)
    parser.add_argument("--defer-insufficient", action="store_true")
    parser.add_argument("--only-expansion-requests", action="store_true")
    parser.add_argument("--defer-wavmark-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.n_trials != 300:
        parser.error("shared-coalition protocol requires exactly 300 trials")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= shard-id < num-shards")
    if args.stage == "validate" and args.model is None:
        parser.error("--model is required for validate")
    schedule = speaker_trial_index(args.n_trials)
    if args.stage == "validate":
        validate_model(args, schedule)
    elif args.stage == "preliminary":
        preliminary(args, schedule)
    else:
        finalize(args, schedule)


if __name__ == "__main__":
    main()

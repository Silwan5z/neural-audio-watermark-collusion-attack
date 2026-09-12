#!/usr/bin/env python3
"""Validate K=8 source copies and replace invalid coalitions.

Existing all-correct trials are retained. A trial with any failed source copy is
replaced by a deterministic random coalition drawn from source payloads already
observed to decode correctly for the same speaker. If that pool is too small,
new random source payloads are tested until eight valid copies are available.
The mixture outcome never enters selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from registry import NBITS, int_to_bits, source_record  # noqa: E402
from watermarks import pesq_wb, resample_to, stoi  # noqa: E402
from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE, bits_to_int, decode_native, get_or_embed_native,
)

SOURCE = ROOT / "results" / "average" / "k8_candidates"
DEST = ROOT / "results" / "average" / "k8"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--max-replacements", type=int)
    p.add_argument("--source-dir", type=Path, default=SOURCE)
    p.add_argument("--dest-dir", type=Path, default=DEST)
    return p.parse_args()


def seed_for(model: str, speaker: str, clip_slot: int, trial_id: int) -> int:
    token = f"k8-valid-clip-indexed-v20|{model}|{speaker}|{clip_slot}|{trial_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def main():
    a = args(); model = a.model; d = NBITS[model]; sr = NATIVE_SAMPLE_RATE[model]
    src = a.source_dir / model; dst = a.dest_dir / model; dst.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted(src.glob("trial_*.json"))]
    if len(records) != 300:
        raise RuntimeError(f"{model}: expected 300 source trials, found {len(records)}")

    # Payload -> exact source-decode record, grouped by the actual utterance.
    pools: dict[tuple[str, int], dict[int, dict]] = {}
    used: dict[tuple[str, int], set[tuple[int, ...]]] = {}
    for row in records:
        key = (row["speaker"], int(row["clip_index"]) - 1)
        pool = pools.setdefault(key, {})
        used.setdefault(key, set()).add(tuple(sorted(row["coalition_payloads"])))
        for source in row["source_copies"]:
            if source["valid"]:
                pool[int(source["assigned_payload"])] = source

    replaced = 0
    for row in records:
        final = dst / f"trial_{int(row['trial_id']):03d}.json"
        if final.exists():
            continue
        if int(row["valid_copy_count"]) == 8:
            shutil.copy2(src / f"trial_{int(row['trial_id']):03d}.json", final)
            continue
        if a.max_replacements is not None and replaced >= a.max_replacements:
            continue

        speaker = row["speaker"]
        clip_slot = int(row["clip_index"]) - 1
        source_key = (speaker, clip_slot); pool = pools[source_key]
        rng = np.random.default_rng(seed_for(model, speaker, clip_slot, int(row["trial_id"])))
        # Nine verified payloads provide multiple distinct eight-member
        # coalitions for speakers whose original valid pool was very small.
        while len(pool) < 9:
            payload = int(rng.integers(0, 2 ** d))
            if payload in pool:
                continue
            wav, got_sr = get_or_embed_native(
                model, speaker, payload, clip_slot)
            if got_sr != sr:
                raise ValueError((got_sr, sr))
            hard, _, presence, extra = decode_native(model, wav, sr)
            decoded = None if hard is None else bits_to_int(hard)
            if decoded == payload:
                pool[payload] = {
                    "assigned_payload": payload, "decoded_payload": decoded,
                    "valid": 1,
                    "watermark_score": (
                        None if not np.isfinite(presence) else float(presence)),
                }

        # Draw a coalition from the observed-valid pool without using mixture outcomes.
        choices = np.array(sorted(pool), dtype=np.int64)
        for _ in range(10000):
            payloads = sorted(rng.choice(choices, size=8, replace=False).astype(int).tolist())
            key = tuple(payloads)
            if key not in used[source_key]:
                used[source_key].add(key); break
        else:
            raise RuntimeError(f"{model} {speaker}: unable to draw a new valid coalition")

        payload_bits = np.asarray([int_to_bits(v, d) for v in payloads], dtype=np.int8)
        members = []
        for payload in payloads:
            wav, got_sr = get_or_embed_native(
                model, speaker, payload, clip_slot)
            if got_sr != sr:
                raise ValueError((got_sr, sr))
            members.append(wav)
        n = min(map(len, members)); members = [wav[:n] for wav in members]
        mixed = np.mean(np.stack(members), axis=0, dtype=np.float64).astype(np.float32)
        hard, p1, presence, extra = decode_native(model, mixed, sr)
        if hard is None: raise RuntimeError(f"{model} trial {row['trial_id']}: absent mixture decode")

        decoded = bits_to_int(hard); counts = payload_bits.sum(axis=0).astype(int)
        majority = np.where(counts > 4, 1, np.where(counts < 4, 0, -1)); strict = majority >= 0
        unanimous = (counts == 0) | (counts == 8); agreement = (payload_bits == hard[None]).sum(axis=1)
        ref16 = resample_to(members[0], sr, 16000) if sr != 16000 else members[0]
        mix16 = resample_to(mixed, sr, 16000) if sr != 16000 else mixed
        replacement = dict(row)
        replacement.update({
            "coalition_payloads": payloads,
            "coalition_bits": payload_bits.astype(int).tolist(),
            "source_copies": [pool[p] for p in payloads], "valid_copy_count": 8,
            "clip_index": source_record(speaker, clip_slot)["clip_index"],
            "source_path": source_record(speaker, clip_slot)["path"],
            "decoded_payload": decoded, "decoded_bits": hard.astype(int).tolist(),
            "bit_probabilities": np.asarray(p1, dtype=float).tolist(),
            "watermark_score": None if not np.isfinite(presence) else float(presence),
            "escaped": int(decoded not in payloads),
            "ones_in_coalition": counts.tolist(),
            "strict_majority_bit_accuracy": float(np.mean(hard[strict] == majority[strict])),
            "unanimous_bit_accuracy": None if not unanimous.any() else float(np.mean(hard[unanimous] == majority[unanimous])),
            "all_strict_majority_bits": int(np.all(hard[strict] == majority[strict])),
            "closest_member_bits": int(agreement.max()),
            "closest_member_bit_accuracy": float(agreement.max() / d),
            "pesq": float(pesq_wb(ref16, mix16)),
            "stoi": float(stoi(ref16, mix16)),
        })
        tmp = final.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(replacement, ensure_ascii=False) + "\n")
        os.replace(tmp, final); replaced += 1
        print(f"{model}: replaced trial {row['trial_id']} ({replaced})", flush=True)

    print(json.dumps({"model": model, "files": len(list(dst.glob('trial_*.json'))),
                      "replacements_this_run": replaced, "output": str(dst)}))


if __name__ == "__main__":
    main()

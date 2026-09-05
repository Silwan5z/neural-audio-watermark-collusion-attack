#!/usr/bin/env python3
"""Complete 300 random K=8 coalitions from correctly decoded source copies.

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
from registry import (NBITS, clean_path_v19, full_registry_bits, get_or_embed,
                      int_to_bits, load_clean, source_record)  # noqa: E402
from watermarks import pesq_wb, resample_to, stoi  # noqa: E402
from run_k8_constructed_payload_case_native import NATIVE_SR, bits_to_int, decode_native, embed_native  # noqa: E402

SOURCE = ROOT / "data" / "k8_population_native_20260830" / "raw"
DEST = ROOT / "data" / "k8_population_source_correct_20260831" / "raw"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--max-replacements", type=int)
    p.add_argument("--source-dir", type=Path, default=SOURCE)
    p.add_argument("--dest-dir", type=Path, default=DEST)
    return p.parse_args()


def seed_for(model: str, speaker: str, local_t: int, trial_id: int) -> int:
    token = f"k8-valid-clip-indexed-v20|{model}|{speaker}|{local_t}|{trial_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def main():
    a = args(); model = a.model; d = NBITS[model]; sr = NATIVE_SR[model]
    src = a.source_dir / model; dst = a.dest_dir / model; dst.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted(src.glob("trial_*.json"))]
    if len(records) != 300:
        raise RuntimeError(f"{model}: expected 300 source trials, found {len(records)}")

    # Payload -> exact source-decode record, grouped by the actual utterance.
    pools: dict[tuple[str, int], dict[int, dict]] = {}
    used: dict[tuple[str, int], set[tuple[int, ...]]] = {}
    for row in records:
        key = (row["speaker"], int(row["local_trial"]))
        pool = pools.setdefault(key, {})
        used.setdefault(key, set()).add(tuple(sorted(row["coalition_payloads"])))
        for source in row["source_decodes"]:
            if source["exact"]:
                pool[int(source["target_payload"])] = source

    replaced = 0
    for row in records:
        final = dst / f"trial_{int(row['trial_id']):03d}.json"
        if final.exists():
            continue
        if int(row["source_exact_count"]) == 8:
            shutil.copy2(src / f"trial_{int(row['trial_id']):03d}.json", final)
            continue
        if a.max_replacements is not None and replaced >= a.max_replacements:
            continue

        speaker = row["speaker"]; local_t = int(row["local_trial"])
        source_key = (speaker, local_t); pool = pools[source_key]
        rng = np.random.default_rng(seed_for(model, speaker, local_t, int(row["trial_id"])))
        clean16 = np.asarray(load_clean(speaker, local_t), dtype=np.float32)

        # Nine verified payloads provide multiple distinct eight-member
        # coalitions for speakers whose original valid pool was very small.
        while len(pool) < 9:
            payload = int(rng.integers(0, 2 ** d))
            if payload in pool:
                continue
            bits = int_to_bits(payload, d)
            if sr == 16000:
                wav = np.asarray(get_or_embed(model, speaker, payload, local_t), dtype=np.float32)
            else:
                wav, got_sr = embed_native(model, clean16, bits.tolist())
                if got_sr != sr: raise ValueError((got_sr, sr))
            hard, _, presence, extra = decode_native(model, wav, sr)
            decoded = None if hard is None else bits_to_int(hard)
            if decoded == payload:
                pool[payload] = {"target_payload": payload, "decoded_payload": decoded,
                                 "exact": 1,
                                 "presence": None if not np.isfinite(presence) else float(presence),
                                 **({"decoded_digits": extra["decoded_digits"]}
                                    if "decoded_digits" in extra else {})}

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
        for bits, payload in zip(payload_bits, payloads):
            if sr == 16000:
                wav = np.asarray(get_or_embed(model, speaker, payload, local_t), dtype=np.float32)
            else:
                wav, got_sr = embed_native(model, clean16, bits.tolist())
                if got_sr != sr: raise ValueError((got_sr, sr))
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
            "selection": "random same-speaker coalition from previously verified source copies",
            "replaces_original_trial": int(row["trial_id"]),
            "coalition_payloads": payloads,
            "coalition_payload_bits": payload_bits.astype(int).tolist(),
            "source_decodes": [pool[p] for p in payloads], "source_exact_count": 8,
            "clip_index": source_record(speaker, local_t)["clip_index"],
            "source_path": str(clean_path_v19(speaker, local_t)),
            "decoded_identity": decoded, "decoded_hard_bits": hard.astype(int).tolist(),
            "soft_bit_probability": np.asarray(p1, dtype=float).tolist(),
            "presence": None if not np.isfinite(presence) else float(presence),
            "tracing_failure": int(decoded not in payloads), "ones_among_8": counts.tolist(),
            "strict_majority_consistency": float(np.mean(hard[strict] == majority[strict])),
            "unanimous_preservation": None if not unanimous.any() else float(np.mean(hard[unanimous] == majority[unanimous])),
            "exact_majority_payload": int(np.all(hard[strict] == majority[strict])),
            "NCA_bits": int(agreement.max()), "NCA": float(agreement.max() / d),
            "PESQ": float(pesq_wb(ref16, mix16)), "STOI": float(stoi(ref16, mix16)),
            **extra,
        })
        tmp = final.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(replacement, ensure_ascii=False) + "\n")
        os.replace(tmp, final); replaced += 1
        print(f"{model}: replaced trial {row['trial_id']} ({replaced})", flush=True)

    print(json.dumps({"model": model, "files": len(list(dst.glob('trial_*.json'))),
                      "replacements_this_run": replaced, "output": str(dst)}))


if __name__ == "__main__":
    main()

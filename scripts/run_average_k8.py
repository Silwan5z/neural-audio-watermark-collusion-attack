#!/usr/bin/env python3
"""Evaluate K=8 uniform averaging at each system's native sample rate.

Writes one JSON per trial and no audio. The deterministic coalition sampler is
identical to ``run_average.py``. The output also stores the per-bit values used
to analyze how coalition composition affects the decoded payload.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from registry import (NBITS, coalition_seed, full_registry_bits,
                      int_to_bits, sample_coalition, source_record,
                      trial_schedule)  # noqa: E402
from watermarks import get_wavmark, pesq_wb, resample_to, stoi  # noqa: E402
from native_audio import (  # noqa: E402
    NATIVE_SAMPLE_RATE, bits_to_int, decode_native, get_or_embed_native)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"))
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "average" / "k8_candidates")
    p.add_argument("--n-trials", type=int, default=300)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def wavmark_decode_all(signals):
    """One shared set of forwards for source hard decodes and mixture soft vote."""
    import torch
    from wavmark.utils import wm_add_util
    model = get_wavmark()
    start = np.asarray(wm_add_util.fix_pattern[:16], dtype=np.int8)
    windows, owners = [], []
    for owner, waveform in enumerate(signals):
        for position in range((len(waveform) - 16000) // 800):
            windows.append(waveform[position * 800:position * 800 + 16000])
            owners.append(owner)
    owners = np.asarray(owners)
    votes = [[] for _ in signals]
    batch = int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE", "600"))
    for offset in range(0, len(windows), batch):
        inputs = np.stack(windows[offset:offset + batch])
        with torch.no_grad():
            decoded = (model["model"].decode(
                torch.from_numpy(inputs).to(model["dev"])) >= 0.5
            ).int().cpu().numpy()
        owner_batch = owners[offset:offset + len(decoded)]
        valid = np.all(decoded[:, :16] == start[None], axis=1)
        for index in np.unique(owner_batch[valid]):
            votes[int(index)].extend(
                decoded[valid & (owner_batch == index), 16:32].tolist())
    output = []
    for vote in votes:
        if not vote:
            output.append((None, None, 0))
        else:
            probability = np.asarray(vote, float).mean(0)
            output.append(((probability >= 0.5).astype(np.int8),
                           probability, len(vote)))
    return output, len(windows)


def main():
    a = parse_args(); model = a.model; d = NBITS[model]; sr = NATIVE_SAMPLE_RATE[model]
    out = a.output_dir / model; out.mkdir(parents=True, exist_ok=True)
    registry = full_registry_bits(model)
    started = time.time(); done = 0
    for gi, (speaker, clip_slot) in enumerate(trial_schedule(a.n_trials)):
        if gi % a.num_shards != a.shard_id: continue
        final = out / f"trial_{gi:03d}.json"
        if final.exists():
            done += 1; continue
        rng = np.random.default_rng(coalition_seed(speaker, 8, clip_slot))
        payloads = sample_coalition(rng, model, 8)
        payload_bits = np.asarray([int_to_bits(v, d) for v in payloads], dtype=np.int8)
        manifest_source = source_record(speaker, clip_slot)
        members = []
        for payload in payloads:
            wav, got_sr = get_or_embed_native(
                model, speaker, payload, clip_slot)
            if got_sr != sr:
                raise ValueError((model, got_sr, sr))
            members.append(wav)
        n = min(map(len, members)); members = [w[:n] for w in members]
        mixed = np.mean(np.stack(members), axis=0, dtype=np.float64).astype(np.float32)

        source_copies = []
        if model == "wavmark":
            batch_decoded, _ = wavmark_decode_all(members + [mixed])
            for payload, (hard_i, _, valid_i) in zip(payloads, batch_decoded[:-1]):
                decoded_i = None if hard_i is None else bits_to_int(np.asarray(hard_i))
                source_copies.append({
                    "assigned_payload": payload, "decoded_payload": decoded_i,
                    "valid": int(decoded_i == payload),
                    "watermark_score": 1.0 if valid_i else 0.0,
                })
            hard,p1,valid = batch_decoded[-1]; presence=1.0 if valid else 0.0
        else:
            for payload, wav in zip(payloads, members):
                hard_i, _, presence_i, _ = decode_native(model, wav, sr)
                decoded_i = None if hard_i is None else bits_to_int(hard_i)
                source_copies.append({
                    "assigned_payload": payload,
                    "decoded_payload": decoded_i,
                    "valid": int(decoded_i == payload),
                    "watermark_score": (
                        None if not np.isfinite(presence_i) else float(presence_i)),
                })
            hard, p1, presence, _ = decode_native(model, mixed, sr)
        if hard is None: raise RuntimeError(f"{model} trial {gi}: absent mixture decode")
        decoded = bits_to_int(hard)
        counts = payload_bits.sum(axis=0).astype(int)
        majority = np.where(counts > 4, 1, np.where(counts < 4, 0, -1))
        strict = majority >= 0
        unanimous = (counts == 0) | (counts == 8)
        agreement = (payload_bits == hard[None]).sum(axis=1)
        ref16 = resample_to(members[0], sr, 16000) if sr != 16000 else members[0]
        mix16 = resample_to(mixed, sr, 16000) if sr != 16000 else mixed
        row = {
            "condition": "average",
            "model": model, "k": 8, "trial_id": gi, "speaker": speaker,
            "clip_index": manifest_source["clip_index"],
            "source_path": manifest_source["path"],
            "sample_rate": sr, "bit_count": d, "bit_order": "LSB-first",
            "coalition_seed": coalition_seed(speaker, 8, clip_slot),
            "coalition_payloads": payloads,
            "coalition_bits": payload_bits.astype(int).tolist(),
            "weights": [0.125] * 8, "mixing": "uniform",
            "source_copies": source_copies,
            "valid_copy_count": sum(x["valid"] for x in source_copies),
            "decoded_payload": decoded, "decoded_bits": hard.astype(int).tolist(),
            "bit_probabilities": np.asarray(p1, dtype=float).tolist(),
            "confidence_source": (
                "valid-window vote fraction" if model == "wavmark"
                else "native decoder bit marginal"),
            "watermark_score": None if not np.isfinite(presence) else float(presence),
            "escaped": int(decoded not in payloads),
            "ones_in_coalition": counts.tolist(),
            "strict_majority_bit_accuracy": float(np.mean(hard[strict] == majority[strict])),
            "unanimous_bit_accuracy": (None if not unanimous.any()
                                      else float(np.mean(hard[unanimous] == majority[unanimous]))),
            "all_strict_majority_bits": int(np.all(hard[strict] == majority[strict])),
            "closest_member_bits": int(agreement.max()),
            "closest_member_bit_accuracy": float(agreement.max() / d),
            "pesq": float(pesq_wb(ref16, mix16)),
            "stoi": float(stoi(ref16, mix16)),
        }
        tmp = final.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, final); done += 1
        if done % 10 == 0:
            print(f"{model}: {done}/{a.n_trials} ({time.time()-started:.0f}s)", flush=True)
    print(json.dumps({"model": model, "completed": done, "output": str(out),
                      "elapsed_seconds": time.time()-started}), flush=True)


if __name__ == "__main__": main()

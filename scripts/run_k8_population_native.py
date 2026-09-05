#!/usr/bin/env python3
"""Random-population K=8 uniform averaging at each system's native rate.

Writes one JSON per trial and no audio. The deterministic coalition sampler is
identical to scripts/attack.py, so this is a bit-level augmentation of the
existing K=8 population rather than a selected case study.
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
from registry import (NBITS, clean_path_v19, coalition_seed, full_registry_bits, get_or_embed,
                      int_to_bits, load_clean, sample_coalition, source_record,
                      speaker_trial_index)  # noqa: E402
from watermarks import get_wavmark, pesq_wb, resample_to, stoi  # noqa: E402
from run_k8_constructed_payload_case_native import (  # noqa: E402
    NATIVE_SR, bits_to_int, decode_native, embed_native, wavmark_vote_probability)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"))
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "k8_population_native_20260830")
    p.add_argument("--n-trials", type=int, default=300)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def wavmark_decode_all(signals):
    """One shared set of forwards for source hard decodes and mixture soft vote."""
    import torch
    from wavmark.utils import wm_add_util
    m=get_wavmark(); start=np.asarray(wm_add_util.fix_pattern[:16],dtype=np.int8)
    windows=[]; owners=[]
    for owner,wav in enumerate(signals):
        for pos in range((len(wav)-16000)//800):
            windows.append(wav[pos*800:pos*800+16000]); owners.append(owner)
    owners=np.asarray(owners); votes=[[] for _ in signals]; batch=int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE","600"))
    for off in range(0,len(windows),batch):
        x=np.stack(windows[off:off+batch])
        with torch.no_grad(): dec=(m["model"].decode(torch.from_numpy(x).to(m["dev"]))>=.5).int().cpu().numpy()
        own=owners[off:off+len(dec)]; keep=np.all(dec[:,:16]==start[None],axis=1)
        for idx in np.unique(own[keep]): votes[int(idx)].extend(dec[keep & (own==idx),16:32].tolist())
    out=[]
    for v in votes:
        if not v: out.append((None,None,0))
        else:
            p=np.asarray(v,float).mean(0); out.append(((p>=.5).astype(np.int8),p,len(v)))
    return out,len(windows)


def main():
    a = parse_args(); model = a.model; d = NBITS[model]; sr = NATIVE_SR[model]
    out = a.output_dir / "raw" / model; out.mkdir(parents=True, exist_ok=True)
    registry = full_registry_bits(model)
    started = time.time(); done = 0
    for gi, (spk, local_t) in enumerate(speaker_trial_index(a.n_trials)):
        if gi % a.num_shards != a.shard_id: continue
        final = out / f"trial_{gi:03d}.json"
        if final.exists():
            done += 1; continue
        rng = np.random.default_rng(coalition_seed(spk, 8, local_t))
        payloads = sample_coalition(rng, model, 8)
        payload_bits = np.asarray([int_to_bits(v, d) for v in payloads], dtype=np.int8)
        manifest_source = source_record(spk, local_t)
        clean16 = np.asarray(load_clean(spk, local_t), dtype=np.float32)
        members = []
        for row, payload in zip(payload_bits, payloads):
            if sr == 16000:
                wav = np.asarray(get_or_embed(model, spk, payload, local_t), dtype=np.float32)
            else:
                wav, got_sr = embed_native(model, clean16, row.tolist())
                if got_sr != sr: raise ValueError((model, got_sr, sr))
            members.append(wav)
        n = min(map(len, members)); members = [w[:n] for w in members]
        mixed = np.mean(np.stack(members), axis=0, dtype=np.float64).astype(np.float32)

        source = []
        if model == "wavmark":
            batch_decoded,total_windows = wavmark_decode_all(members + [mixed])
            for payload, (hard_i, _, valid_i) in zip(payloads, batch_decoded[:-1]):
                decoded_i = None if hard_i is None else bits_to_int(np.asarray(hard_i))
                source.append({"target_payload": payload, "decoded_payload": decoded_i,
                               "exact": int(decoded_i == payload),
                               "presence": 1.0 if valid_i else 0.0,
                               "valid_start_pattern_windows": valid_i})
            hard,p1,valid = batch_decoded[-1]; presence=1.0 if valid else 0.0
            extra = {"valid_start_pattern_windows": valid, "total_sliding_windows_all_signals": total_windows}
        else:
            for payload, wav in zip(payloads, members):
                hard_i, _, presence_i, extra_i = decode_native(model, wav, sr)
                decoded_i = None if hard_i is None else bits_to_int(hard_i)
                source.append({"target_payload": payload, "decoded_payload": decoded_i,
                               "exact": int(decoded_i == payload),
                               "presence": None if not np.isfinite(presence_i) else float(presence_i),
                               **({"decoded_digits": extra_i["decoded_digits"]}
                                  if "decoded_digits" in extra_i else {})})
            hard, p1, presence, extra = decode_native(model, mixed, sr)
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
            "experiment": "random-population native-rate K=8 uniform full-waveform averaging",
            "model": model, "K": 8, "trial_id": gi, "speaker": spk,
            "local_trial": local_t, "coalition_seed": coalition_seed(spk, 8, local_t),
            "clip_index": manifest_source["clip_index"],
            "source_path": str(clean_path_v19(spk, local_t)),
            "native_sample_rate": sr, "payload_nbits": d,
            "payload_bit_order": "LSB-first common-registry identity",
            "coalition_payloads": payloads, "coalition_payload_bits": payload_bits.astype(int).tolist(),
            "weights": [0.125] * 8, "mixing": "aligned equal full-waveform mean",
            "source_decodes": source, "source_exact_count": sum(x["exact"] for x in source),
            "decoded_identity": decoded, "decoded_hard_bits": hard.astype(int).tolist(),
            "soft_bit_probability": np.asarray(p1, dtype=float).tolist(),
            "soft_probability_kind": ("valid-window vote fraction (proxy)" if model == "wavmark"
                                      else "native decoder bit marginal"),
            "presence": None if not np.isfinite(presence) else float(presence),
            "tracing_failure": int(decoded not in payloads),
            "ones_among_8": counts.tolist(),
            "strict_majority_consistency": float(np.mean(hard[strict] == majority[strict])),
            "unanimous_preservation": (None if not unanimous.any()
                                      else float(np.mean(hard[unanimous] == majority[unanimous]))),
            "exact_majority_payload": int(np.all(hard[strict] == majority[strict])),
            "NCA_bits": int(agreement.max()), "NCA": float(agreement.max() / d),
            "PESQ": float(pesq_wb(ref16, mix16)), "STOI": float(stoi(ref16, mix16)),
            "quality_metric_rate": 16000,
            **extra,
        }
        tmp = final.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, final); done += 1
        if done % 10 == 0:
            print(f"{model}: {done}/{a.n_trials} ({time.time()-started:.0f}s)", flush=True)
    print(json.dumps({"model": model, "completed": done, "output": str(out),
                      "elapsed_seconds": time.time()-started}), flush=True)


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Run one pre-registered K=8 equal-mean case with controlled bit counts.

The eight payload rows are fixed. Across the 16 canonical LSB-first bit columns,
the number of ones is [0,1,2,3,4,4,5,6,7,8,4,4,4,4,4,4]. TimbreWM uses
the first ten columns and therefore covers every count from zero through eight,
with two balanced 4/4 columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registry import NBITS, full_registry_bits, load_clean  # noqa: E402
from watermarks import detect_many, embed, extract_evidence, get_wavmark, pesq_wb, stoi  # noqa: E402

MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
OUT = ROOT / "data" / "k8_constructed_payload_case_20260829"
SPK = "chinese:SSB0005"
SEED = 20260829
COUNTS16 = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 4, 4, 4, 4, 4, 4]

# Rows found by a fixed-seed constrained search. The first 10 columns already
# make all eight identities unique; full-16 minimum pairwise Hamming distance=6.
BITS16 = np.asarray([
    [0,0,0,1,1,1,1,1,1,1,0,1,0,1,1,0],
    [0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0],
    [0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1],
    [0,0,0,0,1,1,0,1,0,1,0,0,1,0,1,1],
    [0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0],
    [0,1,0,0,1,0,1,1,1,1,0,1,1,1,0,1],
    [0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0],
], dtype=np.int8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--speaker", default=SPK)
    p.add_argument("--design-only", action="store_true")
    return p.parse_args()


def bits_to_int(bits: np.ndarray) -> int:
    return int(sum(int(v) << i for i, v in enumerate(bits)))


def design_for(model: str) -> np.ndarray:
    d = NBITS[model]
    return BITS16[:, :d].copy()


def validate_design() -> dict:
    if BITS16.shape != (8, 16):
        raise ValueError(BITS16.shape)
    if BITS16.sum(axis=0).tolist() != COUNTS16:
        raise ValueError("column-count mismatch")
    rows10 = {tuple(r) for r in BITS16[:, :10]}
    rows16 = {tuple(r) for r in BITS16}
    if len(rows10) != 8 or len(rows16) != 8:
        raise ValueError("payload rows are not unique")
    d10 = [int(np.count_nonzero(BITS16[i,:10] != BITS16[j,:10]))
           for i in range(8) for j in range(i)]
    d16 = [int(np.count_nonzero(BITS16[i] != BITS16[j]))
           for i in range(8) for j in range(i)]
    return {
        "seed": SEED, "bit_order": "LSB-first", "K": 8,
        "ones_per_bit_16": COUNTS16,
        "ones_per_bit_10": COUNTS16[:10],
        "payloads_16": [bits_to_int(r) for r in BITS16],
        "payloads_10": [bits_to_int(r[:10]) for r in BITS16],
        "min_pairwise_hamming_16": min(d16),
        "min_pairwise_hamming_10": min(d10),
        "pairwise_hamming_16": d16,
        "pairwise_hamming_10": d10,
        "payload_bits_16": BITS16.astype(int).tolist(),
    }


def wavmark_probabilities(signals: list[np.ndarray]) -> list[dict]:
    """Official valid-window bit vote fractions; confidence proxy only."""
    import torch
    from wavmark.utils import wm_add_util

    m = get_wavmark()
    start = np.asarray(wm_add_util.fix_pattern[:16], dtype=np.int8)
    batch_size = int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE", "600"))
    all_windows, owners = [], []
    for owner, wav in enumerate(signals):
        total = (len(wav)-16000)//800
        for pos in range(total):
            all_windows.append(wav[pos*800:pos*800+16000])
            owners.append(owner)
    owners = np.asarray(owners, dtype=int)
    kept = [[] for _ in signals]
    for off in range(0, len(all_windows), batch_size):
        batch = np.stack(all_windows[off:off+batch_size])
        with torch.no_grad():
            dec = (m["model"].decode(torch.from_numpy(batch).to(m["dev"])) >= .5)
        dec = dec.int().cpu().numpy()
        own = owners[off:off+len(dec)]
        valid = np.all(dec[:, :16] == start[None, :], axis=1)
        for idx in np.unique(own[valid]):
            kept[int(idx)].extend(dec[valid & (own == idx), 16:32].tolist())
    out=[]
    for votes in kept:
        if not votes:
            out.append({"p_bit_1": None, "valid_windows": 0})
        else:
            arr=np.asarray(votes, dtype=float)
            out.append({"p_bit_1": arr.mean(axis=0).tolist(), "valid_windows": len(arr)})
    return out


def main() -> None:
    args = parse_args()
    design = validate_design()
    if args.design_only:
        print(json.dumps(design, indent=2))
        return

    model=args.model; d=NBITS[model]; bits=design_for(model)
    payloads=[bits_to_int(r) for r in bits]
    out=args.output_dir
    audio_dir=out/"audio"/model
    raw_dir=out/"raw"
    audio_dir.mkdir(parents=True, exist_ok=True); raw_dir.mkdir(parents=True, exist_ok=True)
    start=time.time()
    clean=np.asarray(load_clean(args.speaker), dtype=np.float32)
    sf.write(audio_dir/"clean.wav", clean, 16000, subtype="FLOAT")

    members=[]; member_paths=[]
    for idx,row in enumerate(bits,1):
        wav=np.asarray(embed(model, clean, row.tolist()), dtype=np.float32)
        if not np.isfinite(wav).all(): raise ValueError(f"nonfinite member {idx}")
        members.append(wav)
    n=min([len(clean)]+[len(w) for w in members])
    clean=clean[:n]; members=[w[:n] for w in members]
    for idx,wav in enumerate(members,1):
        p=audio_dir/f"colluder_{idx:02d}_payload_{payloads[idx-1]}.wav"
        sf.write(p,wav,16000,subtype="FLOAT"); member_paths.append(str(p))
    mixed=np.mean(np.stack(members),axis=0,dtype=np.float64).astype(np.float32)
    mix_path=audio_dir/"equal_mean_k8.wav"; sf.write(mix_path,mixed,16000,subtype="FLOAT")

    registry=full_registry_bits(model)
    decoded=detect_many(model,members+[mixed],registry)
    clean_results=[]
    for idx,(_,presence,hard) in enumerate(decoded[:-1]):
        hard_arr=None if hard is None else np.asarray(hard,dtype=np.int8)
        clean_results.append({
            "colluder_index":idx+1, "target_payload":payloads[idx],
            "decoded_payload":None if hard_arr is None else bits_to_int(hard_arr),
            "decoded_bits":None if hard_arr is None else hard_arr.astype(int).tolist(),
            "exact":int(hard_arr is not None and np.array_equal(hard_arr,bits[idx])),
            "presence":None if not np.isfinite(presence) else float(presence),
        })
    _,presence,hard=decoded[-1]
    if hard is None: raise RuntimeError(f"{model}: mixed audio has no valid decode")
    hard=np.asarray(hard,dtype=np.int8)

    confidence_kind="native bit posterior"
    extra={}
    if model=="wavmark":
        probs=wavmark_probabilities([mixed])[0]
        if probs["p_bit_1"] is None: raise RuntimeError("WavMark: no valid windows")
        p1=np.asarray(probs["p_bit_1"],dtype=float)
        confidence_kind="valid-window vote fraction (confidence proxy)"
        extra={"valid_start_pattern_windows":probs["valid_windows"]}
    else:
        evidence=np.asarray(extract_evidence(model,mixed),dtype=float)
        p1=(evidence+1.0)/2.0
    if p1.shape!=(d,) or not np.isfinite(p1).all(): raise ValueError(f"bad p1 {p1.shape}")

    counts=bits.sum(axis=0).astype(int)
    majority=np.where(counts>4,1,np.where(counts<4,0,-1))
    strict=np.where(majority>=0)[0]
    tie=np.where(majority<0)[0]
    departures=[int(i) for i in strict if hard[i]!=majority[i]]
    unanimous=[int(i) for i in range(d) if counts[i] in (0,8)]
    unanimous_violations=[int(i) for i in unanimous if hard[i] != int(counts[i]==8)]
    agreements=[int(np.count_nonzero(hard==row)) for row in bits]
    decoded_identity=bits_to_int(hard)
    bit_results=[]
    for j in range(d):
        bit_results.append({
            "bit_index_lsb_first":j,
            "colluder_bits":bits[:,j].astype(int).tolist(),
            "ones_among_8":int(counts[j]),
            "composition":f"{int(counts[j])}/8",
            "state":"tie_4_4" if counts[j]==4 else "unanimous_0" if counts[j]==0 else "unanimous_1" if counts[j]==8 else "strict_majority_1" if counts[j]>4 else "strict_majority_0",
            "p_bit_1":float(p1[j]),
            "native_decoded_bit":int(hard[j]),
            "confidence_native_decoded_bit":float(p1[j] if hard[j] else 1-p1[j]),
        })
    result={
        "experiment":"constructed K=8 payload-composition validation",
        "model":model,"K":8,"seed":SEED,"speaker":args.speaker,
        "mixing":"equal full-waveform mean","weights":[0.125]*8,
        "payload_nbits":d,"payload_bit_order":"LSB-first",
        "ones_per_bit":counts.tolist(),"payloads":payloads,
        "payload_bits":bits.astype(int).tolist(),"member_audio_paths":member_paths,
        "mixed_audio_path":str(mix_path),"clean_audio_path":str(audio_dir/"clean.wav"),
        "clean_member_decodes":clean_results,
        "clean_attribution_exact":sum(r["exact"] for r in clean_results),
        "decoded_identity":decoded_identity,"decoded_bits":hard.astype(int).tolist(),
        "presence":None if not np.isfinite(presence) else float(presence),
        "confidence_kind":confidence_kind,"bit_results":bit_results,
        "strict_majority_bits":strict.astype(int).tolist(),"tie_bits":tie.astype(int).tolist(),
        "strict_majority_departures":departures,
        "unanimous_bits":unanimous,"unanimous_violations":unanimous_violations,
        "agreement_bits_by_colluder":agreements,
        "NCA_bits":max(agreements),"NCA":max(agreements)/d,
        "tracing_failure":int(decoded_identity not in payloads),
        "PESQ_vs_first_copy":pesq_wb(members[0],mixed),
        "STOI_vs_first_copy":stoi(members[0],mixed),
        "elapsed_seconds":time.time()-start,
        **extra,
    }
    tmp=raw_dir/f"{model}.json.tmp"; final=raw_dir/f"{model}.json"
    tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    os.replace(tmp,final)
    print(json.dumps({"model":model,"output":str(final),"payloads":payloads,
                      "clean_exact":result["clean_attribution_exact"],
                      "decoded_identity":decoded_identity,"NCA":result["NCA"],
                      "TF":result["tracing_failure"],"elapsed":result["elapsed_seconds"]},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

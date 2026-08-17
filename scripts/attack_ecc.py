"""Evaluate one-bit-correcting ECC watermark payloads under attack.

16-bit models use shortened Hamming (16,11); TimbreWM uses (10,6).  Both
The default runs only ECC because the uncoded condition is identical to the
main ``attack.py`` experiment.  ``--schemes uncoded,ecc1`` remains available
only when a self-contained ablation CSV is explicitly required.  The ECC
registry contains only valid codewords, so top-1 identity decoding includes
the prescribed single-bit correction capability.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecc import decode_bits, ecc_codebook, encode_int, info_bits, int_to_bits  # noqa: E402
from registry import (NBITS, coalition_seed, full_registry_bits, get_or_embed,
                      sample_coalition, speaker_trial_index)  # noqa: E402
from watermarks import detect, detect_many, pesq_wb, si_sdr, stoi  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"


def fwp(wavs, K):
    best_e, pair = -1.0, (0, 1)
    for i in range(K):
        for j in range(i + 1, K):
            e = float(np.mean((wavs[i] - wavs[j]) ** 2))
            if e > best_e:
                best_e, pair = e, (i, j)
    a = np.zeros(K)
    a[list(pair)] = 0.5
    return a


def evaluate(model, y, registry_bits, coalition_ids, message_width, decoded=None):
    scores, _, hard = decoded if decoded is not None else detect(model, y.astype(np.float32), registry_bits)
    rank = np.argsort(scores)[::-1]
    coalition = set(coalition_ids)
    asr = int(int(rank[0]) not in coalition)
    r3 = int(not (set(rank[:3]) & coalition))
    r5 = int(not (set(rank[:5]) & coalition))
    acc = ""
    if hard is not None:
        # For ECC this explicitly applies its syndrome decoder; for raw mode
        # `hard` itself is the message representation.
        h = np.asarray(hard, dtype=np.int8)
        if len(h) != message_width:
            h, _ = decode_bits(h, len(h))
        msg_rows = np.stack([int_to_bits(v, message_width) for v in coalition_ids])
        acc = int((msg_rows == h[None, :]).sum(axis=1).max())
    return asr, r3, r5, acc


def run_scheme(model, K, trial_idx, scheme):
    d = NBITS[model]
    if scheme == "uncoded":
        k, registry = d, full_registry_bits(model)
    else:
        k, registry = info_bits(d), ecc_codebook(d)
    rows = []
    t0 = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        if scheme == "uncoded":
            message_ids = sample_coalition(rng, model, K)
            embedded_ids = message_ids
        else:
            message_ids = sorted(rng.choice(1 << k, size=K, replace=False).tolist())
            embedded_ids = [encode_int(msg, d) for msg in message_ids]
        wavs = [get_or_embed(model, spk, code) for code in embedded_ids]
        n = min(map(len, wavs))
        wavs = [w[:n] for w in wavs]
        ref = wavs[0]
        methods = (("mean", np.ones(K) / K), ("fwp", fwp(wavs, K)))
        outputs = [sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
                   for _, a in methods]
        decoded_outputs = detect_many(model, outputs, registry)
        for (method, a), y, decoded in zip(methods, outputs, decoded_outputs):
            asr, r3, r5, acc = evaluate(model, y, registry, message_ids, k, decoded)
            rows.append({
                "model": model, "K": K, "scheme": scheme,
                "code_bits": d, "info_bits": k, "codebook_size": len(registry),
                "spk": spk, "local_t": local_t, "gi": gi, "method": method,
                "ASR": asr, "R3_escape": r3, "R5_escape": r5,
                "ACC_near": acc, "ACC_near_norm": "" if acc == "" else f"{acc/k:.4f}",
                "PESQ": f"{pesq_wb(ref, y):.4f}", "STOI": f"{stoi(ref, y):.4f}",
                "SI_SDR": f"{si_sdr(ref, y):.2f}",
            })
        if (gi + 1) % 30 == 0:
            print(f"  {model} {scheme} K={K}: {gi + 1}/{len(trial_idx)} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--schemes", default="ecc1",
                    help="comma-separated subset of uncoded,ecc1 (default: ecc1)")
    args = ap.parse_args()
    if args.K < 2:
        raise ValueError("K must be at least two")
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    if not schemes or any(s not in {"uncoded", "ecc1"} for s in schemes):
        raise ValueError("--schemes must contain only uncoded and/or ecc1")
    trial_idx = speaker_trial_index(n_total=args.n_trials)
    rows = []
    for scheme in schemes:
        rows.extend(run_scheme(args.model, args.K, trial_idx, scheme))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"attack_ecc_{args.model}_K{args.K}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for scheme in schemes:
        for method in ("mean", "fwp"):
            vals = [r["ASR"] for r in rows if r["scheme"] == scheme and r["method"] == method]
            print(f"{args.model} K={args.K} {scheme:7s} {method}: ASR={np.mean(vals):.3f}")


if __name__ == "__main__":
    main()

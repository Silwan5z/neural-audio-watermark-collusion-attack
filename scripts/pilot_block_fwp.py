"""Pilot: blockwise FWP collusion with smooth 50/50 mixing.

For each 0.5 s block, enumerate every coalition pair, select the pair with
the largest blockwise waveform MSE, average it equally, and reconstruct blocks
with a 20 ms raised-cosine overlap-add transition.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (NBITS, coalition_seed, full_registry_bits, get_or_embed,
                      int_to_bits, sample_coalition, speaker_trial_index)
from watermarks import detect, detect_many, pesq_wb, si_sdr, stoi

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
SR = 16000


def block_fwp_smooth(wavs, block_seconds=0.5, fade_seconds=0.02,
                     num_blocks=None):
    """Per-block FWP: maximum-MSE pair, 50/50 mix, raised-cosine OLA."""
    n = min(map(len, wavs))
    wavs = np.stack([w[:n] for w in wavs]).astype(np.float32)
    fade = int(round(fade_seconds * SR))
    if num_blocks is None:
        block = int(round(block_seconds * SR))
        if n < block:
            return wavs.mean(axis=0)
        hop = block - fade
        starts = list(range(0, n - block + 1, hop))
        if starts[-1] != n - block:
            starts.append(n - block)
    else:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        # N blocks with N-1 overlap regions exactly tile the waveform:
        # N * block - (N-1) * fade == n.
        block = int(round((n + (num_blocks - 1) * fade) / num_blocks))
        hop = block - fade
        starts = [i * hop for i in range(num_blocks)]
        if starts[-1] + block != n:
            raise ValueError("cannot tile waveform exactly with this block count")
    accum = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, endpoint=True))
    for start in starts:
        end = start + block
        pieces = wavs[:, start:end]
        best_dist, pair = -np.inf, (0, 1)
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                dist = float(np.mean((pieces[i].astype(np.float64) -
                                      pieces[j].astype(np.float64)) ** 2))
                if dist > best_dist:
                    best_dist, pair = dist, (i, j)
        mixed = 0.5 * (pieces[pair[0]] + pieces[pair[1]])
        window = np.ones(block, dtype=np.float64)
        if start > 0:
            window[:fade] = ramp
        if end < n:
            window[-fade:] = ramp[::-1]
        accum[start:end] += mixed * window
        norm[start:end] += window
    return (accum / np.maximum(norm, 1e-12)).astype(np.float32)


def fwp(wavs):
    """FWP: farthest whole-waveform pair, mixed equally."""
    best_energy, pair = -1.0, (0, 1)
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            energy = float(np.mean((wavs[i] - wavs[j]) ** 2))
            if energy > best_energy:
                best_energy, pair = energy, (i, j)
    return 0.5 * (wavs[pair[0]] + wavs[pair[1]])


def utterance_eep(wavs):
    """Whole-utterance max/min-energy pair, mixed equally."""
    energy = np.array([np.mean(w.astype(np.float64) ** 2) for w in wavs])
    lo, hi = int(np.argmin(energy)), int(np.argmax(energy))
    return 0.5 * (wavs[lo] + wavs[hi])


def measure(model, y, coalition, registry, d, decoded=None):
    scores, _, hard = decoded if decoded is not None else detect(model, y.astype(np.float32), registry)
    rank = np.argsort(scores)[::-1]
    coll_set = set(coalition)
    asr = int(int(rank[0]) not in coll_set)
    r3 = int(not (set(rank[:3]) & coll_set))
    r5 = int(not (set(rank[:5]) & coll_set))
    nac = ""
    if hard is not None:
        coll_bits = np.stack([int_to_bits(ci, d) for ci in coalition])
        nac = float((coll_bits == hard[None, :]).sum(axis=1).max() / d)
    return asr, r3, r5, nac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--block_seconds", type=float, default=0.5)
    ap.add_argument("--num_blocks", type=int,
                    help="exact number of overlapping blocks; overrides block_seconds")
    ap.add_argument("--fade_seconds", type=float, default=0.02)
    args = ap.parse_args()
    if args.K < 3:
        raise ValueError("K must be at least 3 for a nontrivial pair selection")
    d = NBITS[args.model]
    registry = full_registry_bits(args.model)
    rows = []
    t0 = time.time()
    for gi, (spk, local_t) in enumerate(speaker_trial_index(args.n_trials)):
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(rng, args.model, args.K)
        wavs = [get_or_embed(args.model, spk, code) for code in coalition]
        n = min(map(len, wavs))
        wavs = [w[:n] for w in wavs]
        pair = rng.choice(args.K, size=2, replace=False)
        outputs = {
            "mean": np.mean(wavs, axis=0, dtype=np.float32),
            "fwp": fwp(wavs),
            "rp": 0.5 * (wavs[int(pair[0])] + wavs[int(pair[1])]),
            "eep": utterance_eep(wavs),
            "median": np.median(np.stack(wavs), axis=0).astype(np.float32),
            "minimum": np.min(np.stack(wavs), axis=0),
            "maximum": np.max(np.stack(wavs), axis=0),
            "block_fwp50_smooth": block_fwp_smooth(
                wavs, args.block_seconds, args.fade_seconds, args.num_blocks),
        }
        ref = wavs[0]
        output_items = list(outputs.items())
        decoded_outputs = detect_many(args.model, [y.astype(np.float32) for _, y in output_items], registry)
        for (method, y), decoded in zip(output_items, decoded_outputs):
            asr, r3, r5, nac = measure(args.model, y, coalition, registry, d, decoded)
            rows.append({
                "model": args.model, "K": args.K, "spk": spk, "local_t": local_t,
                "gi": gi, "method": method, "block_seconds": args.block_seconds,
                "fade_seconds": args.fade_seconds, "num_blocks": args.num_blocks or "",
                "selection_rule": "blockwise_max_pair_mse",
                "ASR": asr, "R3_escape": r3, "R5_escape": r5,
                "NAC": "" if nac == "" else f"{nac:.4f}",
                "PESQ": f"{pesq_wb(ref, y):.4f}", "STOI": f"{stoi(ref, y):.4f}",
                "SI_SDR": f"{si_sdr(ref, y):.2f}",
            })
        if (gi + 1) % 10 == 0:
            print(f"  {args.model} K={args.K}: {gi+1}/{args.n_trials} ({time.time()-t0:.0f}s)", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.num_blocks is None:
        block_tag = f"B{args.block_seconds:.1f}".replace(".", "p")
    else:
        block_tag = f"N{args.num_blocks}"
    out = RESULTS / f"pilot_block_fwp_{args.model}_K{args.K}_{block_tag}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n=== {args.model} K={args.K} block-FWP pilot (n={args.n_trials}) ===")
    for method in ("mean", "fwp", "rp", "eep", "median", "minimum", "maximum",
                   "block_fwp50_smooth"):
        rr = [r for r in rows if r["method"] == method]
        asr = np.mean([r["ASR"] for r in rr])
        nacs = [float(r["NAC"]) for r in rr if r["NAC"]]
        print(f"  {method:20s} ASR={asr:.3f} NAC={np.mean(nacs) if nacs else float('nan'):.3f}")


if __name__ == "__main__":
    main()

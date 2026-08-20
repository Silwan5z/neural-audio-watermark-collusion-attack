"""Relative temporal-misalignment sensitivity for mean and FWP (K=5).

For each signed condition, one coalition copy is advanced/delayed while all
others remain fixed.  The shifted member rotates deterministically across
trials to avoid privileging a particular coalition position.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (coalition_seed, full_registry_bits, get_or_embed,  # noqa: E402
                      sample_coalition, speaker_trial_index)
from watermarks import detect_many, pesq_wb, stoi  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
K = 5
SHIFTS_MS = [0, -10, 10, -20, 20, -50, 50]
FIELDS = [
    "model", "K", "trial_id", "spk", "local_t", "method", "shift_ms",
    "shifted_colluder_index", "ASR", "attribution_margin", "PESQ", "STOI",
]


def write_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    expected = len(SHIFTS_MS) * 2
    completed = {trial for trial, rr in grouped.items() if len(rr) == expected}
    return [r for r in rows if int(r["trial_id"]) in completed], completed


def shift_fixed_length(wav: np.ndarray, samples: int) -> np.ndarray:
    if samples == 0:
        return wav.copy()
    if abs(samples) >= len(wav):
        return np.zeros_like(wav)
    if samples > 0:  # delay
        return np.concatenate([np.zeros(samples, dtype=wav.dtype), wav[:-samples]])
    advance = -samples
    return np.concatenate([wav[advance:], np.zeros(advance, dtype=wav.dtype)])


def fwp(wavs: list[np.ndarray]) -> np.ndarray:
    best_pair, best_dist = (0, 1), -np.inf
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            dist = float(np.mean((wavs[i] - wavs[j]) ** 2))
            if dist > best_dist:
                best_pair, best_dist = (i, j), dist
    weights = np.zeros(len(wavs))
    weights[list(best_pair)] = 0.5
    return weights


def attack_metrics(scores: np.ndarray, coalition: list[int]) -> tuple[int, float]:
    coll = np.asarray(coalition, dtype=np.int64)
    top1 = int(np.argsort(scores, kind="stable")[::-1][0])
    coll_max = float(np.max(scores[coll]))
    mask = np.ones(len(scores), dtype=bool)
    mask[coll] = False
    non_max = float(np.max(scores[mask]))
    return int(top1 not in set(coalition)), non_max - coll_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"])
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"temporal_sensitivity_{args.model}_K5.csv"
    partial = RESULTS / f"temporal_sensitivity_{args.model}_K5.partial.csv"
    rows, completed = load_completed(partial)
    trials = speaker_trial_index(n_total=args.n_trials)
    registry = full_registry_bits(args.model)
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coalition = sample_coalition(rng, args.model, K)
        wavs = [get_or_embed(args.model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        reference = wavs[0]
        shifted_index = trial_id % K
        specs, signals = [], []
        for shift_ms in SHIFTS_MS:
            condition_wavs = [wav.copy() for wav in wavs]
            if shift_ms:
                samples = int(round(16000 * shift_ms / 1000))
                condition_wavs[shifted_index] = shift_fixed_length(
                    condition_wavs[shifted_index], samples)
            methods = [("mean", np.full(K, 1 / K)), ("fwp", fwp(condition_wavs))]
            for method, weights in methods:
                signal = sum(weights[i] * condition_wavs[i] for i in range(K)).astype(np.float32)
                specs.append((shift_ms, -1 if shift_ms == 0 else shifted_index, method))
                signals.append(signal)
        decoded = detect_many(args.model, signals, registry)
        for (shift_ms, member, method), signal, (scores, _, _) in zip(specs, signals, decoded):
            asr, margin = attack_metrics(scores, coalition)
            rows.append({
                "model": args.model, "K": K, "trial_id": trial_id, "spk": spk,
                "local_t": local_t, "method": method, "shift_ms": shift_ms,
                "shifted_colluder_index": member, "ASR": asr,
                "attribution_margin": f"{margin:.8f}",
                "PESQ": f"{pesq_wb(reference, signal):.4f}",
                "STOI": f"{stoi(reference, signal):.4f}",
            })
        if (trial_id + 1) % 10 == 0:
            write_atomic(partial, rows)
            print(f"{args.model} temporal: {trial_id + 1}/{len(trials)} "
                  f"({time.time() - start:.0f}s)", flush=True)

    write_atomic(partial, rows)
    write_atomic(out, rows)
    print(f"completed {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

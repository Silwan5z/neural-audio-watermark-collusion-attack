#!/usr/bin/env python3
"""Evidence-chain experiment on the collusion_300 registry.

This is the migrated counterpart of the old v19 script.  It uses the 100
``language:speaker_id`` entries and SHA-256 trial seeds in :mod:`registry`,
the registry's atomic embedding cache, and batched decoding through
``detect_many``.  Results stay in the full-suite staging location:
``results/evaluation/evidence_chain_{model}_K{K}.csv``.

``payload_farthest`` is deliberately retained as an evaluator-only oracle
comparator.  It is not a deployable blind method and is kept separate from the
PGR diagnostic rather than being renamed to a regular attack method.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blind_distance import dm_weights  # noqa: E402
from framing import tct  # noqa: E402
from registry import (  # noqa: E402
    CAP, NBITS, coalition_seed, full_registry_bits, get_or_embed, int_to_bits,
    sample_coalition, speaker_trial_index,
)
from watermarks import detect_many, pesq_wb, si_sdr, stoi  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
FIELDS = [
    "model", "K", "spk", "local_t", "gi", "method", "ASR", "presence",
    "S_C", "margin", "PESQ", "STOI", "SI_SDR", "rho_wave", "rho_det",
    "cb_margin", "cb_shuf_margin",
]
METHODS_PER_TRIAL = 9


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    """Atomically publish a whole checkpoint, never a partially-written CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    os.replace(tmp, path)


def load_completed_trials(path: Path) -> tuple[list[dict], set[int]]:
    """Keep only complete method sets, so a restart is deterministic and safe."""
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(int(row["gi"]) for row in rows)
    complete = {gi for gi, n in counts.items() if n == METHODS_PER_TRIAL}
    return [row for row in rows if int(row["gi"]) in complete], complete


def eep(wavs: list[np.ndarray]) -> np.ndarray:
    energies = np.asarray([np.mean(w ** 2) for w in wavs])
    a = np.zeros(len(wavs)); a[int(energies.argmax())] = a[int(energies.argmin())] = 0.5
    return a


def fwp(wavs: list[np.ndarray]) -> np.ndarray:
    best, pair = -np.inf, (0, 1)
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            value = float(np.mean((wavs[i] - wavs[j]) ** 2))
            if value > best:
                best, pair = value, (i, j)
    a = np.zeros(len(wavs)); a[pair[0]] = a[pair[1]] = 0.5
    return a


def payload_farthest(coll_ints: list[int], d: int) -> np.ndarray:
    bits = np.stack([int_to_bits(ci, d) for ci in coll_ints])
    dist = (bits[:, None] != bits[None, :]).sum(axis=2)
    i, j = np.unravel_index(np.argmax(np.triu(dist, 1)), dist.shape)
    a = np.zeros(len(coll_ints)); a[i] = a[j] = 0.5
    return a


def alignment_wave(wavs: list[np.ndarray], coll_ints: list[int], d: int) -> float:
    bits = np.stack([int_to_bits(ci, d) for ci in coll_ints])
    wave_dist, bit_dist = [], []
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            wave_dist.append(float(np.mean((wavs[i] - wavs[j]) ** 2)))
            bit_dist.append(float((bits[i] != bits[j]).sum()))
    return float(spearmanr(wave_dist, bit_dist)[0]) if len(wave_dist) >= 2 else float("nan")


def margin(scores: np.ndarray, coll_ints: list[int]) -> float:
    coll_set = set(coll_ints)
    return float(max(scores[i] for i in coll_set) - max(scores[i] for i in range(len(scores)) if i not in coll_set))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(NBITS))
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=300)
    args = ap.parse_args()
    if args.K < 2:
        raise ValueError("K must be at least 2")

    model, K, d = args.model, args.K, NBITS[args.model]
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)
    out_csv = RESULTS / f"evidence_chain_{model}_K{K}.csv"
    partial_csv = RESULTS / f"evidence_chain_{model}_K{K}.partial.csv"
    rows, completed = load_completed_trials(partial_csv)
    if completed:
        print(f"resuming {model} K={K}: {len(completed)}/{len(trial_idx)} trials", flush=True)

    start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        if gi in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(map(len, wavs)); wavs = [w[:n] for w in wavs]
        C = np.stack([int_to_bits(ci, d) for ci in coll_ints]).astype(float)

        random_i, random_j = rng.choice(K, size=2, replace=False)
        rp = np.zeros(K); rp[random_i] = rp[random_j] = 0.5
        # A non-member target makes tct an actual payload-aware framing control.
        target = int(rng.integers(0, 2 ** d))
        while target in coll_ints:
            target = int(rng.integers(0, 2 ** d))
        Gc = (C * 2 - 1) @ (C * 2 - 1).T
        permutation = rng.permutation(K)
        methods = {
            "single_copy": np.eye(K)[0], "mean": np.full(K, 1 / K), "rp": rp,
            "random_dirichlet": rng.dirichlet(np.ones(K)), "eep": eep(wavs),
            "fwp": fwp(wavs), "dm": dm_weights(wavs, CAP), "tct": tct(C, int_to_bits(target, d), CAP),
            "payload_farthest": payload_farthest(coll_ints, d),
        }
        # Shuffle the coalition-to-payload assignment while retaining the same target.
        tct_shuf = tct(C[permutation], int_to_bits(target, d), CAP)[np.argsort(permutation)]
        names = list(methods)
        signals = [sum(a[i] * wavs[i] for i in range(K)).astype(np.float32) for a in methods.values()]
        signals.extend([sum(methods["tct"][i] * wavs[i] for i in range(K)).astype(np.float32),
                        sum(tct_shuf[i] * wavs[i] for i in range(K)).astype(np.float32)])
        decoded = detect_many(model, signals, registry_bits)
        rho_wave = alignment_wave(wavs, coll_ints, d)
        mean_scores = decoded[names.index("mean")][0]
        detector_evidence = -(Gc @ np.full(K, 1 / K))
        coll_scores = np.asarray([mean_scores[ci] for ci in coll_ints])
        rho_det = (float(pearsonr(detector_evidence, coll_scores)[0])
                   if detector_evidence.std() > 1e-12 and coll_scores.std() > 1e-12 else float("nan"))
        tct_margin = margin(decoded[-2][0], coll_ints)
        shuf_margin = margin(decoded[-1][0], coll_ints)

        for name, y, (scores, presence, _) in zip(names, signals, decoded):
            top1 = int(np.argmax(scores))
            presence_value = "" if presence is None or not np.isfinite(presence) else f"{presence:.4f}"
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi, "method": name,
                "ASR": int(top1 not in set(coll_ints)), "presence": presence_value,
                "S_C": f"{max(scores[ci] for ci in coll_ints):.4f}",
                "margin": f"{margin(scores, coll_ints):.4f}",
                "PESQ": f"{pesq_wb(wavs[0], y):.4f}", "STOI": f"{stoi(wavs[0], y):.4f}",
                "SI_SDR": f"{si_sdr(wavs[0], y):.2f}", "rho_wave": f"{rho_wave:.4f}",
                "rho_det": f"{rho_det:.4f}", "cb_margin": f"{tct_margin:.4f}",
                "cb_shuf_margin": f"{shuf_margin:.4f}",
            })
        if (gi + 1) % 10 == 0:
            write_rows_atomic(partial_csv, rows)
            print(f"  {model} K={K}: {gi + 1}/{len(trial_idx)} checkpointed ({time.time() - start:.0f}s)", flush=True)

    write_rows_atomic(partial_csv, rows)
    write_rows_atomic(out_csv, rows)
    print(f"completed {model} K={K}: {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()

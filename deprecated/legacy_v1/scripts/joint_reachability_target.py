#!/usr/bin/env python3
"""TCT against targets selected jointly by d_hull and registered-bit support.

The primary selector is an equal-weight, scale-free rank fusion:
  fused = 0.5 * percentile(-d_hull) + 0.5 * percentile(R_bit).
For diagnosis the same candidate pool also emits d_hull-only and R_bit-only
selectors.  This script preserves the project's existing native top-1 rule.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from framing import convex_dist_batch_exact, tct  # noqa: E402
from registry import (CAP, NBITS, coalition_seed, full_registry_bits,
                      full_registry_size, get_or_embed, int_to_bits,
                      sample_coalition, speaker_trial_index)  # noqa: E402
from watermarks import detect_many  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
FIELDS = [
    "model", "K", "spk", "local_t", "trial_id", "selector", "target",
    "candidate_pool_size", "d_hull", "G_bit", "R_bit",
    "min_bit_support_count", "unsupported_bit_count", "d_quality_percentile",
    "rbit_quality_percentile", "fused_score", "target_hit", "target_margin",
]
SELECTORS = ("dhull", "rbit", "fused")


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["trial_id"])].append(row)
    completed = {
        trial for trial, rr in grouped.items()
        if len(rr) == len(SELECTORS) and {r["selector"] for r in rr} == set(SELECTORS)
    }
    return [row for row in rows if int(row["trial_id"]) in completed], completed


def exact_positive_cdf(K: int, L: int) -> dict[int, float]:
    """Exact CDF of product M_l under zero-truncated Binomial(K, 0.5)."""
    base = {m: math.comb(K, m) / (2 ** K - 1) for m in range(1, K + 1)}
    dist = {1: 1.0}
    for _ in range(L):
        nxt: dict[int, float] = defaultdict(float)
        for product, probability in dist.items():
            for support, q in base.items():
                nxt[product * support] += probability * q
        dist = dict(nxt)
    total = sum(dist.values())
    cumulative = 0.0
    out = {}
    for product in sorted(dist):
        cumulative += dist[product] / total
        out[product] = cumulative
    out[max(out)] = 1.0
    return out


def bit_support(C: np.ndarray, targets: np.ndarray, K: int,
                cdf: dict[int, float]) -> tuple[np.ndarray, ...]:
    supports = (targets[:, None, :] == C[None, :, :]).sum(axis=1).astype(np.int64)
    unsupported = (supports == 0).sum(axis=1)
    minimum = supports.min(axis=1)
    products = np.prod(supports, axis=1, dtype=np.int64)
    G = np.zeros(len(targets), dtype=float)
    R = np.zeros(len(targets), dtype=float)
    positive = unsupported == 0
    if np.any(positive):
        G[positive] = np.exp(np.log(supports[positive] / float(K)).mean(axis=1))
        R[positive] = np.asarray([cdf[int(product)] for product in products[positive]])
    return G, R, minimum, unsupported


def choose_indices(target_ids: np.ndarray, distances: np.ndarray,
                   rbit: np.ndarray) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    n = len(target_ids)
    if n < 2:
        d_quality = np.ones(n)
        r_quality = np.ones(n)
    else:
        d_quality = 1.0 - (rankdata(distances, method="average") - 1.0) / (n - 1.0)
        r_quality = (rankdata(rbit, method="average") - 1.0) / (n - 1.0)
    fused = 0.5 * d_quality + 0.5 * r_quality

    # Stable lexicographic tie-breaks: better primary score, then smaller hull
    # distance, then smaller identity.  They never inspect detector outcomes.
    dhull_idx = int(np.lexsort((target_ids, -rbit, distances))[0])
    rbit_idx = int(np.lexsort((target_ids, distances, -rbit))[0])
    fused_idx = int(np.lexsort((target_ids, distances, -fused))[0])
    return {"dhull": dhull_idx, "rbit": rbit_idx, "fused": fused_idx}, d_quality, r_quality, fused


def target_margin(scores: np.ndarray, target: int) -> float:
    return float(scores[target] - np.max(np.delete(scores, target)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(NBITS))
    ap.add_argument("--K", required=True, type=int, choices=[2, 3, 5, 8])
    ap.add_argument("--n_trials", default=300, type=int)
    ap.add_argument("--candidate_pool", default=2000, type=int)
    ap.add_argument("--trial_start", default=0, type=int)
    ap.add_argument("--trial_end", default=None, type=int)
    ap.add_argument("--output_tag", default="")
    args = ap.parse_args()

    trial_end = args.n_trials if args.trial_end is None else args.trial_end
    if not 0 <= args.trial_start < trial_end <= args.n_trials:
        ap.error("trial range must satisfy 0 <= trial_start < trial_end <= n_trials")
    if args.candidate_pool < 10:
        ap.error("candidate_pool must be at least 10")

    model, K, L = args.model, args.K, NBITS[args.model]
    registry_bits = full_registry_bits(model)
    registry_size = full_registry_size(model)
    trials = speaker_trial_index(n_total=args.n_trials)
    tag = f".{args.output_tag}" if args.output_tag else ""
    stem = f"joint_reachability_target_{model}_K{K}{tag}"
    out = RESULTS / f"{stem}.csv"
    partial = RESULTS / f"{stem}.partial.csv"
    rows, completed = load_completed(partial)
    calibration = exact_positive_cdf(K, L)
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id < args.trial_start or trial_id >= trial_end or trial_id in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coalition = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        C = np.stack([int_to_bits(identity, L) for identity in coalition]).astype(float)

        available = np.setdiff1d(
            np.arange(registry_size, dtype=np.int64), np.asarray(coalition, dtype=np.int64),
            assume_unique=False)
        candidate_rng = np.random.default_rng(coalition_seed(spk, K, local_t) + 999)
        candidate_ids = candidate_rng.choice(
            available, size=min(args.candidate_pool, len(available)), replace=False)
        candidate_bits = registry_bits[candidate_ids]
        distances = convex_dist_batch_exact(C, candidate_bits)
        G, R, minimum, unsupported = bit_support(C, candidate_bits, K, calibration)
        selected, d_quality, r_quality, fused = choose_indices(candidate_ids, distances, R)

        signals = []
        for selector in SELECTORS:
            idx = selected[selector]
            weights = tct(C, candidate_bits[idx], CAP)
            signals.append(sum(weights[i] * wavs[i] for i in range(K)).astype(np.float32))
        decoded = detect_many(model, signals, registry_bits)

        for selector, (_, (scores, _, _)) in zip(SELECTORS, zip(signals, decoded)):
            idx = selected[selector]
            target = int(candidate_ids[idx])
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t,
                "trial_id": trial_id, "selector": selector, "target": target,
                "candidate_pool_size": len(candidate_ids),
                "d_hull": f"{distances[idx]:.8f}", "G_bit": f"{G[idx]:.10f}",
                "R_bit": f"{R[idx]:.10f}",
                "min_bit_support_count": int(minimum[idx]),
                "unsupported_bit_count": int(unsupported[idx]),
                "d_quality_percentile": f"{d_quality[idx]:.10f}",
                "rbit_quality_percentile": f"{r_quality[idx]:.10f}",
                "fused_score": f"{fused[idx]:.10f}",
                "target_hit": int(np.argmax(scores) == target),
                "target_margin": f"{target_margin(scores, target):.8f}",
            })
        if (trial_id + 1) % 10 == 0:
            write_rows_atomic(partial, rows)
            print(f"{model} K={K}: trial={trial_id + 1}/{len(trials)} "
                  f"elapsed={time.time() - start:.0f}s", flush=True)

    write_rows_atomic(partial, rows)
    write_rows_atomic(out, rows)
    print(f"completed {model} K={K} range=[{args.trial_start},{trial_end}) "
          f"rows={len(rows)} -> {out}")


if __name__ == "__main__":
    main()

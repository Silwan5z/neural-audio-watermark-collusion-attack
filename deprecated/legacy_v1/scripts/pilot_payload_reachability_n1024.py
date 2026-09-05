#!/usr/bin/env python3
"""Pilot for detector-agnostic payload-reachability mixing.

For every non-colluder identity in the matched N=1024 registry, optimize a
bit-support log loss over the full simplex while enforcing a minimum effective
coalition size.  Rank candidates by that optimized payload-reachability loss,
select the best ten, and use each target's own optimized weights to mix the
waveforms.  Ordinary payload-hull distance is recorded only as a diagnostic.
Detector outputs are used only once, after the waveforms have been generated,
to evaluate target Hit@1 and Any@10.

This script deliberately does not call framing.tct().
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from framing import convex_dist_batch_exact, restricted_top1_and_margin  # noqa: E402
from registry import (  # noqa: E402
    NBITS,
    coalition_seed,
    full_registry_bits,
    full_registry_size,
    get_or_embed,
    int_to_bits,
    sample_coalition,
    speaker_trial_index,
)
from registry_size_control import active_registry  # noqa: E402
from watermarks import detect_many, detect_wavmark_many, get_wavmark  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
N_REGISTRY = 1024
N_TARGETS = 10
FIELDS = [
    "model", "K", "N_registry", "trial_id", "spk", "local_t",
    "coalition_ids", "candidate_pool_size", "target", "target_rank",
    "d_hull", "method",
    "payload_loss", "effective_K", "max_weight", "weights",
    "unsupported_bits", "solver_success", "loss_mode", "bit_threshold",
    "boundary_sharpness", "worst_bit_weight", "target_hit", "target_margin",
]


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
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    completed = {
        trial_id for trial_id, trial_rows in grouped.items()
        if len(trial_rows) == N_TARGETS
    }
    return [row for row in rows if int(row["trial_id"]) in completed], completed


def reachability_weights(C: np.ndarray, target: np.ndarray,
                         min_effective_k: float,
                         loss_mode: str = "log_support",
                         softmin_beta: float = 8.0,
                         bit_threshold: float = 0.6,
                         boundary_sharpness: float = 20.0,
                         worst_bit_weight: float = 1.0,
                         worst_temperature: float = 0.05,
                         eps: float = 1e-6) -> tuple[np.ndarray, float, bool, int]:
    """Optimize target-bit support without detector scores or TCT distance."""
    C = np.asarray(C, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    K = C.shape[0]
    match = (C == target[None, :]).astype(np.float64)
    unsupported = int(np.sum(match.sum(axis=0) == 0))

    def objective(weights: np.ndarray) -> float:
        support = match.T @ weights
        if loss_mode == "log_support":
            return float(-np.mean(np.log(support + eps)))
        if loss_mode == "softmin_margin":
            margins = support - 0.5
            return float(
                (logsumexp(-softmin_beta * margins) - np.log(len(margins)))
                / softmin_beta
            )
        per_bit = np.logaddexp(
            0.0, boundary_sharpness * (bit_threshold - support)
        ) / boundary_sharpness
        smooth_worst = worst_temperature * (
            logsumexp(per_bit / worst_temperature) - np.log(len(per_bit))
        )
        return float(np.mean(per_bit) + worst_bit_weight * smooth_worst)

    def gradient(weights: np.ndarray) -> np.ndarray:
        support = match.T @ weights
        if loss_mode == "log_support":
            return -np.mean(match / (support[None, :] + eps), axis=1)
        if loss_mode == "softmin_margin":
            margins = support - 0.5
            return match @ (-softmax(-softmin_beta * margins))
        per_bit = np.logaddexp(
            0.0, boundary_sharpness * (bit_threshold - support)
        ) / boundary_sharpness
        derivative = -expit(boundary_sharpness * (bit_threshold - support))
        bit_coefficients = (
            np.full(len(support), 1.0 / len(support))
            + worst_bit_weight * softmax(per_bit / worst_temperature)
        )
        return match @ (derivative * bit_coefficients)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0,
         "jac": lambda weights: np.ones_like(weights)},
        {"type": "ineq",
         "fun": lambda weights: 1.0 / min_effective_k - np.sum(weights ** 2),
         "jac": lambda weights: -2.0 * weights},
    ]
    initial = np.full(K, 1.0 / K, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * K,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    if weights.sum() <= 1e-12:
        weights = initial
        success = False
    else:
        weights /= weights.sum()
        effective_k = 1.0 / np.sum(weights ** 2)
        success = bool(result.success and effective_k + 1e-6 >= min_effective_k)
        if not success:
            weights = initial
    return weights, objective(weights), success, unsupported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(NBITS))
    parser.add_argument("--K", required=True, type=int, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", default=30, type=int)
    parser.add_argument("--min_effective_k", default=1.5, type=float)
    parser.add_argument(
        "--loss_mode",
        choices=["log_support", "boundary_margin", "softmin_margin"],
        default="log_support")
    parser.add_argument("--softmin_beta", default=8.0, type=float)
    parser.add_argument("--bit_threshold", default=0.6, type=float)
    parser.add_argument("--boundary_sharpness", default=20.0, type=float)
    parser.add_argument("--worst_bit_weight", default=1.0, type=float)
    parser.add_argument("--worst_temperature", default=0.05, type=float)
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()

    if not 1.0 <= args.min_effective_k <= args.K:
        parser.error("min_effective_k must lie between 1 and K")
    if not 0.5 <= args.bit_threshold < 1.0:
        parser.error("bit_threshold must lie in [0.5, 1.0)")
    if args.boundary_sharpness <= 0 or args.worst_temperature <= 0:
        parser.error("boundary_sharpness and worst_temperature must be positive")
    if args.softmin_beta <= 0:
        parser.error("softmin_beta must be positive")
    if args.worst_bit_weight < 0:
        parser.error("worst_bit_weight must be non-negative")
    full_size = full_registry_size(args.model)
    if N_REGISTRY > full_size:
        parser.error(f"matched registry N={N_REGISTRY} exceeds native size {full_size}")

    trials = speaker_trial_index(n_total=args.n_trials)
    registry_bits = full_registry_bits(args.model)
    tag = f".{args.output_tag}" if args.output_tag else ""
    stem = f"payload_reachability_N{N_REGISTRY}_{args.model}_K{args.K}{tag}"
    out = RESULTS / f"{stem}.csv"
    partial = RESULTS / f"{stem}.partial.csv"
    rows, completed = load_completed(partial)
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(rng, args.model, args.K)
        wavs = [get_or_embed(args.model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        C = np.stack([
            int_to_bits(identity, NBITS[args.model]) for identity in coalition
        ]).astype(np.float64)

        active_ids = active_registry(
            full_size, coalition, N_REGISTRY, spk, args.K, local_t)
        candidate_ids = active_ids[
            ~np.isin(active_ids, np.asarray(coalition, dtype=np.int64))
        ]
        candidate_bits = registry_bits[candidate_ids]
        distances = convex_dist_batch_exact(C, candidate_bits)
        candidate_solutions: list[tuple[np.ndarray, float, bool, int]] = []
        reachability_losses = np.empty(len(candidate_ids), dtype=np.float64)
        for candidate_index, target_bits in enumerate(candidate_bits):
            solution = reachability_weights(
                C, target_bits, args.min_effective_k,
                loss_mode=args.loss_mode,
                softmin_beta=args.softmin_beta,
                bit_threshold=args.bit_threshold,
                boundary_sharpness=args.boundary_sharpness,
                worst_bit_weight=args.worst_bit_weight,
                worst_temperature=args.worst_temperature)
            candidate_solutions.append(solution)
            reachability_losses[candidate_index] = solution[1]

        # The new method's target selector and waveform mixer use the same
        # optimized payload-reachability objective.  Hull distance appears only
        # as a deterministic secondary tie-break and a diagnostic output.
        selected = np.lexsort(
            (candidate_ids, distances, reachability_losses)
        )[:N_TARGETS]

        outputs: list[np.ndarray] = []
        metadata: list[tuple] = []
        for target_rank, candidate_index in enumerate(selected, start=1):
            target = int(candidate_ids[candidate_index])
            distance = float(distances[candidate_index])
            weights, loss, success, unsupported = candidate_solutions[candidate_index]
            output = sum(
                weights[i] * wavs[i] for i in range(args.K)
            ).astype(np.float32)
            outputs.append(output)
            metadata.append((
                int(target), target_rank, float(distance), weights,
                loss, success, unsupported,
            ))

        if args.model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(args.model, outputs, registry_bits)

        for meta, (scores, _, _) in zip(metadata, decoded):
            target, target_rank, distance, weights, loss, success, unsupported = meta
            top1, margin = restricted_top1_and_margin(scores, active_ids, target)
            effective_k = 1.0 / np.sum(weights ** 2)
            rows.append({
                "model": args.model,
                "K": args.K,
                "N_registry": N_REGISTRY,
                "trial_id": trial_id,
                "spk": spk,
                "local_t": local_t,
                "coalition_ids": json.dumps(coalition),
                "candidate_pool_size": len(candidate_ids),
                "target": target,
                "target_rank": target_rank,
                "d_hull": f"{distance:.8f}",
                "method": f"payload_reachability_{args.loss_mode}",
                "payload_loss": f"{loss:.10f}",
                "effective_K": f"{effective_k:.8f}",
                "max_weight": f"{np.max(weights):.8f}",
                "weights": json.dumps(weights.tolist()),
                "unsupported_bits": unsupported,
                "solver_success": int(success),
                "loss_mode": args.loss_mode,
                "bit_threshold": f"{args.bit_threshold:.6f}",
                "boundary_sharpness": f"{args.boundary_sharpness:.6f}",
                "worst_bit_weight": f"{args.worst_bit_weight:.6f}",
                "target_hit": int(top1 == target),
                "target_margin": f"{margin:.8f}",
            })

        write_rows_atomic(partial, rows)
        if (trial_id + 1) % 5 == 0:
            print(
                f"{args.model} K={args.K}: {trial_id + 1}/{len(trials)} trials "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    write_rows_atomic(partial, rows)
    write_rows_atomic(out, rows)

    hits = np.asarray([int(row["target_hit"]) for row in rows], dtype=np.int64)
    by_trial: dict[int, list[int]] = {}
    for row in rows:
        by_trial.setdefault(int(row["trial_id"]), []).append(int(row["target_hit"]))
    any_hits = np.asarray([int(any(values)) for values in by_trial.values()])
    effective = np.asarray([float(row["effective_K"]) for row in rows])
    maximum = np.asarray([float(row["max_weight"]) for row in rows])
    solver_ok = np.asarray([int(row["solver_success"]) for row in rows])
    print(f"completed {out} rows={len(rows)}", flush=True)
    print(f"single-target Hit@1: {100 * hits.mean():.1f}%", flush=True)
    print(f"Any@{N_TARGETS}: {100 * any_hits.mean():.1f}%", flush=True)
    print(f"mean effective K: {effective.mean():.3f}", flush=True)
    print(f"mean max weight: {maximum.mean():.3f}", flush=True)
    print(f"solver success: {100 * solver_ok.mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()

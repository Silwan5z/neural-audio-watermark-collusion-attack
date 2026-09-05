#!/usr/bin/env python3
"""Pilot for contribution-aware soft-min payload reachability.

The attack is detector agnostic.  It first screens every non-colluder target
using optimized soft-min bit margins.  It then computes exact payload-only
Shapley contributions for a short list, converts those contributions into
feasible mixing weights, re-ranks the targets with those actual weights, and
mixes the corresponding watermarked copies.  Detector outputs are requested
only after all attacked waveforms for a trial have been fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from framing import convex_dist_batch_exact, restricted_top1_and_margin  # noqa: E402
from pilot_dynamic_payload_contribution import exact_contributions  # noqa: E402
from pilot_payload_reachability_n1024 import reachability_weights  # noqa: E402
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
    "coalition_ids", "candidate_pool_size", "preselect_size",
    "target", "target_rank", "screen_rank", "d_hull", "method",
    "softmin_beta", "min_effective_ratio", "screen_loss", "final_loss",
    "weight_strategy", "dynamic_effective_floor",
    "shapley_contributions", "contribution_shares", "weights",
    "effective_contributors", "positive_contributors", "effective_K",
    "max_weight", "screen_solver_success", "projection_success",
    "target_hit", "target_margin",
]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    completed = {
        trial_id for trial_id, trial_rows in grouped.items()
        if len(trial_rows) == N_TARGETS
    }
    return [row for row in rows if int(row["trial_id"]) in completed], completed


def softmin_loss(C: np.ndarray, target: np.ndarray, weights: np.ndarray,
                 beta: float) -> float:
    match = (np.asarray(C) == np.asarray(target)[None, :]).astype(np.float64)
    margins = match.T @ np.asarray(weights, dtype=np.float64) - 0.5
    return float((logsumexp(-beta * margins) - np.log(len(margins))) / beta)


def subset_softmin_losses(C: np.ndarray, target: np.ndarray,
                          beta: float) -> dict[int, float]:
    K, n_bits = C.shape
    empty_margin = np.full(n_bits, -0.5, dtype=np.float64)
    losses = {
        0: float((logsumexp(-beta * empty_margin) - np.log(n_bits)) / beta)
    }
    for mask in range(1, 1 << K):
        members = [member for member in range(K) if mask & (1 << member)]
        _, loss, _, _ = reachability_weights(
            C[members], target, 1.0,
            loss_mode="softmin_margin", softmin_beta=beta)
        losses[mask] = float(loss)
    return losses


def contribution_shares(shapley: np.ndarray) -> tuple[np.ndarray, float, int]:
    positive = np.clip(np.asarray(shapley, dtype=np.float64), 0.0, None)
    if positive.sum() <= 1e-12:
        shares = np.full(len(positive), 1.0 / len(positive))
        return shares, float(len(positive)), 0
    shares = positive / positive.sum()
    effective = float(1.0 / np.sum(shares ** 2))
    return shares, effective, int(np.sum(shares > 1e-6))


def project_contribution_weights(shares: np.ndarray,
                                 min_effective_k: float) -> tuple[np.ndarray, bool]:
    shares = np.asarray(shares, dtype=np.float64)
    shares = np.clip(shares, 0.0, None)
    shares /= shares.sum()
    if np.sum(shares ** 2) <= 1.0 / min_effective_k + 1e-12:
        return shares, True

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * np.sum((weights - shares) ** 2))

    result = minimize(
        objective,
        np.full(len(shares), 1.0 / len(shares)),
        jac=lambda weights: weights - shares,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(shares),
        constraints=[
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0,
             "jac": lambda weights: np.ones_like(weights)},
            {"type": "ineq",
             "fun": lambda weights: 1.0 / min_effective_k - np.sum(weights ** 2),
             "jac": lambda weights: -2.0 * weights},
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= weights.sum()
    feasible = np.sum(weights ** 2) <= 1.0 / min_effective_k + 1e-6
    if not result.success or not feasible:
        return np.full(len(shares), 1.0 / len(shares)), False
    return weights, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(NBITS))
    parser.add_argument("--K", required=True, type=int, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", default=30, type=int)
    parser.add_argument("--softmin_beta", default=8.0, type=float)
    parser.add_argument("--min_effective_ratio", default=0.6, type=float)
    parser.add_argument("--preselect", default=50, type=int)
    parser.add_argument(
        "--weight_strategy",
        choices=["contribution_projection", "dynamic_floor"],
        default="dynamic_floor")
    parser.add_argument("--contribution_floor_ratio", default=0.6, type=float)
    parser.add_argument("--min_dynamic_effective_k", default=1.5, type=float)
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()

    if args.softmin_beta <= 0:
        parser.error("softmin_beta must be positive")
    if not 0 < args.min_effective_ratio <= 1:
        parser.error("min_effective_ratio must lie in (0, 1]")
    if args.preselect < N_TARGETS:
        parser.error(f"preselect must be at least {N_TARGETS}")
    if args.contribution_floor_ratio <= 0:
        parser.error("contribution_floor_ratio must be positive")
    if not 1 <= args.min_dynamic_effective_k <= args.K:
        parser.error("min_dynamic_effective_k must lie between 1 and K")
    full_size = full_registry_size(args.model)
    if N_REGISTRY > full_size:
        parser.error(f"matched registry N={N_REGISTRY} exceeds native size {full_size}")

    min_effective_k = max(1.0, args.min_effective_ratio * args.K)
    trials = speaker_trial_index(n_total=args.n_trials)
    registry_bits = full_registry_bits(args.model)
    tag = f".{args.output_tag}" if args.output_tag else ""
    stem = f"contribution_softmin_N{N_REGISTRY}_{args.model}_K{args.K}{tag}"
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

        screen_losses = np.empty(len(candidate_ids), dtype=np.float64)
        screen_success = np.empty(len(candidate_ids), dtype=bool)
        for candidate_index, target_bits in enumerate(candidate_bits):
            _, loss, success, _ = reachability_weights(
                C, target_bits, min_effective_k,
                loss_mode="softmin_margin", softmin_beta=args.softmin_beta)
            screen_losses[candidate_index] = loss
            screen_success[candidate_index] = success

        screen_order = np.lexsort((candidate_ids, distances, screen_losses))
        screen_ranks = np.empty(len(candidate_ids), dtype=np.int64)
        screen_ranks[screen_order] = np.arange(1, len(candidate_ids) + 1)
        shortlist = screen_order[:min(args.preselect, len(screen_order))]

        finalists: list[dict] = []
        for candidate_index in shortlist:
            target_bits = candidate_bits[candidate_index].astype(np.float64)
            losses = subset_softmin_losses(C, target_bits, args.softmin_beta)
            shapley = exact_contributions(losses, args.K)
            shares, effective_contributors, positive_contributors = (
                contribution_shares(shapley)
            )
            if args.weight_strategy == "dynamic_floor":
                dynamic_floor = float(np.clip(
                    args.contribution_floor_ratio * effective_contributors,
                    args.min_dynamic_effective_k,
                    min_effective_k,
                ))
                weights, final_loss, projection_success, _ = reachability_weights(
                    C, target_bits, dynamic_floor,
                    loss_mode="softmin_margin", softmin_beta=args.softmin_beta)
            else:
                dynamic_floor = min_effective_k
                weights, projection_success = project_contribution_weights(
                    shares, min_effective_k)
                final_loss = softmin_loss(
                    C, target_bits, weights, args.softmin_beta)
            finalists.append({
                "candidate_index": int(candidate_index),
                "final_loss": final_loss,
                "shapley": shapley,
                "shares": shares,
                "weights": weights,
                "dynamic_floor": dynamic_floor,
                "effective_contributors": effective_contributors,
                "positive_contributors": positive_contributors,
                "projection_success": projection_success,
            })

        finalists.sort(key=lambda item: (
            item["final_loss"],
            screen_losses[item["candidate_index"]],
            int(candidate_ids[item["candidate_index"]]),
        ))
        selected = finalists[:N_TARGETS]
        outputs = []
        for item in selected:
            outputs.append(sum(
                item["weights"][member] * wavs[member]
                for member in range(args.K)
            ).astype(np.float32))

        # This is the first detector call in the trial.  It is evaluation only.
        if args.model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(args.model, outputs, registry_bits)

        for target_rank, (item, (scores, _, _)) in enumerate(
                zip(selected, decoded), start=1):
            candidate_index = item["candidate_index"]
            target = int(candidate_ids[candidate_index])
            top1, margin = restricted_top1_and_margin(scores, active_ids, target)
            weights = item["weights"]
            rows.append({
                "model": args.model,
                "K": args.K,
                "N_registry": N_REGISTRY,
                "trial_id": trial_id,
                "spk": spk,
                "local_t": local_t,
                "coalition_ids": json.dumps(coalition),
                "candidate_pool_size": len(candidate_ids),
                "preselect_size": len(shortlist),
                "target": target,
                "target_rank": target_rank,
                "screen_rank": int(screen_ranks[candidate_index]),
                "d_hull": f"{distances[candidate_index]:.8f}",
                "method": f"contribution_softmin_{args.weight_strategy}",
                "softmin_beta": f"{args.softmin_beta:.6f}",
                "min_effective_ratio": f"{args.min_effective_ratio:.6f}",
                "screen_loss": f"{screen_losses[candidate_index]:.10f}",
                "final_loss": f"{item['final_loss']:.10f}",
                "weight_strategy": args.weight_strategy,
                "dynamic_effective_floor": f"{item['dynamic_floor']:.8f}",
                "shapley_contributions": json.dumps(item["shapley"].tolist()),
                "contribution_shares": json.dumps(item["shares"].tolist()),
                "weights": json.dumps(weights.tolist()),
                "effective_contributors": f"{item['effective_contributors']:.8f}",
                "positive_contributors": item["positive_contributors"],
                "effective_K": f"{1.0 / np.sum(weights ** 2):.8f}",
                "max_weight": f"{np.max(weights):.8f}",
                "screen_solver_success": int(screen_success[candidate_index]),
                "projection_success": int(item["projection_success"]),
                "target_hit": int(top1 == target),
                "target_margin": f"{margin:.8f}",
            })

        write_rows_atomic(partial, rows)
        print(
            f"{args.model} K={args.K}: {trial_id + 1}/{len(trials)} trials "
            f"({time.time() - start:.0f}s)",
            flush=True,
        )

    write_rows_atomic(partial, rows)
    write_rows_atomic(out, rows)
    hits = np.asarray([int(row["target_hit"]) for row in rows])
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(int(row["target_hit"]))
    any_hits = np.asarray([int(any(values)) for values in grouped.values()])
    rank_one = np.asarray([
        int(row["target_hit"]) for row in rows if int(row["target_rank"]) == 1
    ])
    effective = np.asarray([float(row["effective_K"]) for row in rows])
    contributors = np.asarray([float(row["effective_contributors"]) for row in rows])
    max_weights = np.asarray([float(row["max_weight"]) for row in rows])
    projection_ok = np.asarray([int(row["projection_success"]) for row in rows])
    print(f"completed {out} rows={len(rows)}", flush=True)
    print(f"single-target Hit@1: {100 * hits.mean():.1f}%", flush=True)
    print(f"Any@{N_TARGETS}: {100 * any_hits.mean():.1f}%", flush=True)
    print(f"rank-1 target Hit@1: {100 * rank_one.mean():.1f}%", flush=True)
    print(f"mean effective K: {effective.mean():.3f}", flush=True)
    print(f"mean effective contributors: {contributors.mean():.3f}", flush=True)
    print(f"mean max weight: {max_weights.mean():.3f}", flush=True)
    print(f"projection success: {100 * projection_ok.mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()

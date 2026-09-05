#!/usr/bin/env python3
"""Exact payload-only contribution analysis for selected framing targets.

For each coalition/target row produced by the boundary-reachability pilot,
enumerate every coalition subset and re-optimize the same detector-agnostic
payload loss.  Exact Shapley-style contributions and leave-one-out loss
increases are computed without audio, detector scores, or detector queries.
The existing target_hit column is copied only for post-hoc grouping.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pilot_payload_reachability_n1024 import reachability_weights  # noqa: E402
from registry import NBITS, int_to_bits  # noqa: E402


def boundary_loss_from_support(support: np.ndarray, bit_threshold: float,
                               boundary_sharpness: float,
                               worst_bit_weight: float,
                               worst_temperature: float) -> float:
    per_bit = np.logaddexp(
        0.0, boundary_sharpness * (bit_threshold - support)
    ) / boundary_sharpness
    smooth_worst = worst_temperature * (
        logsumexp(per_bit / worst_temperature) - np.log(len(per_bit))
    )
    return float(np.mean(per_bit) + worst_bit_weight * smooth_worst)


def subset_losses(C: np.ndarray, target: np.ndarray,
                  bit_threshold: float, boundary_sharpness: float,
                  worst_bit_weight: float,
                  worst_temperature: float) -> dict[int, float]:
    K, n_bits = C.shape
    losses = {
        0: boundary_loss_from_support(
            np.zeros(n_bits, dtype=np.float64), bit_threshold,
            boundary_sharpness, worst_bit_weight, worst_temperature)
    }
    for mask in range(1, 1 << K):
        members = [index for index in range(K) if mask & (1 << index)]
        _, loss, _, _ = reachability_weights(
            C[members], target, 1.0,
            loss_mode="boundary_margin",
            bit_threshold=bit_threshold,
            boundary_sharpness=boundary_sharpness,
            worst_bit_weight=worst_bit_weight,
            worst_temperature=worst_temperature)
        losses[mask] = float(loss)
    return losses


def exact_contributions(losses: dict[int, float], K: int) -> np.ndarray:
    """Return exact Shapley values for improvement in optimized payload loss."""
    values = {mask: -loss for mask, loss in losses.items()}
    contributions = np.zeros(K, dtype=np.float64)
    denominator = math.factorial(K)
    full_mask = (1 << K) - 1
    for member in range(K):
        member_bit = 1 << member
        others = full_mask ^ member_bit
        subset = others
        while True:
            size = subset.bit_count()
            coefficient = (
                math.factorial(size) * math.factorial(K - size - 1)
                / denominator
            )
            contributions[member] += coefficient * (
                values[subset | member_bit] - values[subset]
            )
            if subset == 0:
                break
            subset = (subset - 1) & others
    return contributions


def effective_count(values: np.ndarray) -> tuple[float, int, np.ndarray]:
    positive = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = positive.sum()
    if total <= 1e-12:
        return 0.0, 0, np.zeros_like(positive)
    normalized = positive / total
    effective = 1.0 / np.sum(normalized ** 2)
    positive_count = int(np.sum(normalized > 1e-6))
    return float(effective), positive_count, normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv)
    required = {
        "model", "K", "trial_id", "target", "target_rank",
        "coalition_ids", "weights", "target_hit", "bit_threshold",
        "boundary_sharpness", "worst_bit_weight",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        parser.error(f"input CSV is missing columns: {missing}")

    output_rows: list[dict] = []
    start = time.time()
    for row_index, row in source.iterrows():
        model = str(row["model"])
        coalition = [int(value) for value in json.loads(row["coalition_ids"])]
        K = int(row["K"])
        if len(coalition) != K:
            raise ValueError(f"row {row_index}: coalition length does not equal K")
        n_bits = NBITS[model]
        C = np.stack([int_to_bits(identity, n_bits) for identity in coalition]).astype(float)
        target = int(row["target"])
        target_bits = int_to_bits(target, n_bits).astype(float)
        bit_threshold = float(row["bit_threshold"])
        boundary_sharpness = float(row["boundary_sharpness"])
        worst_bit_weight = float(row["worst_bit_weight"])
        worst_temperature = 0.05

        losses = subset_losses(
            C, target_bits, bit_threshold, boundary_sharpness,
            worst_bit_weight, worst_temperature)
        shapley = exact_contributions(losses, K)
        effective, positive_count, normalized = effective_count(shapley)
        full_mask = (1 << K) - 1
        full_loss = losses[full_mask]
        leave_one_out = np.asarray([
            losses[full_mask ^ (1 << member)] - full_loss
            for member in range(K)
        ], dtype=np.float64)
        attack_weights = np.asarray(json.loads(row["weights"]), dtype=np.float64)

        output_rows.append({
            "model": model,
            "K": K,
            "trial_id": int(row["trial_id"]),
            "target": target,
            "target_rank": int(row["target_rank"]),
            "target_hit": int(row["target_hit"]),
            "coalition_ids": json.dumps(coalition),
            "attack_weights": json.dumps(attack_weights.tolist()),
            "shapley_contributions": json.dumps(shapley.tolist()),
            "normalized_contributions": json.dumps(normalized.tolist()),
            "leave_one_out_loss_increase": json.dumps(leave_one_out.tolist()),
            "effective_contributors": f"{effective:.8f}",
            "positive_contributors": positive_count,
            "empty_subset_loss": f"{losses[0]:.10f}",
            "full_subset_loss": f"{full_loss:.10f}",
            "shapley_efficiency_error": f"{abs(shapley.sum() - (losses[0] - full_loss)):.12e}",
        })
        if (row_index + 1) % 50 == 0:
            print(
                f"processed {row_index + 1}/{len(source)} targets "
                f"({time.time() - start:.0f}s)", flush=True)

    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    result = pd.DataFrame(output_rows)
    result["effective_contributors"] = pd.to_numeric(result["effective_contributors"])
    result["positive_contributors"] = pd.to_numeric(result["positive_contributors"])
    print(f"completed {destination} rows={len(result)}", flush=True)
    for label, group in [("all", result), ("hit", result[result.target_hit == 1]),
                         ("miss", result[result.target_hit == 0])]:
        if len(group) == 0:
            continue
        print(
            f"{label}: n={len(group)} "
            f"mean_effective_contributors={group.effective_contributors.mean():.3f} "
            f"mean_positive_contributors={group.positive_contributors.mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

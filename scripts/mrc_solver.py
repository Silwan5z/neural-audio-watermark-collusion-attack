"""Weight optimization used by the MRC target-selection experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registry import int_to_bits  # noqa: E402


def optimize_softmin(
    coalition_bits: np.ndarray,
    target_bits: np.ndarray,
    beta: float,
    minimum_effective_k: float | None,
    entropy_weight: float,
) -> tuple[np.ndarray, float, bool]:
    """Optimize mixture weights for one target payload."""
    coalition_size = coalition_bits.shape[0]
    sign = 2.0 * target_bits.astype(np.float64) - 1.0
    matrix = coalition_bits.astype(np.float64) * sign[None, :]
    offset = -0.5 * sign

    def margins(weights: np.ndarray) -> np.ndarray:
        return matrix.T @ weights + offset

    def negative_objective(weights: np.ndarray) -> float:
        value = -(1.0 / beta) * logsumexp(-beta * margins(weights))
        if entropy_weight > 0:
            clipped = np.clip(weights, 1e-12, 1.0)
            value += entropy_weight * float(-np.sum(clipped * np.log(clipped)))
        return -value

    def negative_gradient(weights: np.ndarray) -> np.ndarray:
        values = margins(weights)
        probability = np.exp(-beta * (values - values.max()))
        probability /= probability.sum()
        gradient = matrix @ probability
        if entropy_weight > 0:
            clipped = np.clip(weights, 1e-12, 1.0)
            gradient += entropy_weight * (-(np.log(clipped) + 1.0))
        return -gradient

    if (minimum_effective_k is not None
            and minimum_effective_k >= coalition_size - 1e-12):
        weights = np.full(coalition_size, 1.0 / coalition_size)
        score = float(-(1.0 / beta) * logsumexp(-beta * margins(weights)))
        return weights, score, True

    constraints = [{
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1.0,
        "jac": lambda weights: np.ones(coalition_size),
    }]
    if minimum_effective_k is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda weights: (
                1.0 / minimum_effective_k - np.sum(weights ** 2)),
            "jac": lambda weights: -2.0 * weights,
        })

    start = np.full(coalition_size, 1.0 / coalition_size)
    result = minimize(
        negative_objective,
        start,
        jac=negative_gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * coalition_size,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-14},
    )
    if not result.success:
        retry = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        retry = (retry / retry.sum()
                 if retry.sum() > 1e-8 else start)
        result = minimize(
            negative_objective,
            retry,
            jac=negative_gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * coalition_size,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights = weights / weights.sum() if weights.sum() > 1e-8 else start
    score = float(-(1.0 / beta) * logsumexp(-beta * margins(weights)))
    effective_k = 1.0 / float(np.sum(weights ** 2))
    feasible = (minimum_effective_k is None
                or effective_k + 1e-6 >= minimum_effective_k)
    return weights, score, bool(result.success and feasible)


def score_target_task(task):
    """Picklable worker entry point for an independent target solve."""
    coalition_bits, target, nbits, beta, minimum_effective_k, entropy_weight = task
    weights, score, success = optimize_softmin(
        coalition_bits,
        int_to_bits(target, nbits),
        beta,
        minimum_effective_k,
        entropy_weight,
    )
    return score, int(target), weights, success, np.nan

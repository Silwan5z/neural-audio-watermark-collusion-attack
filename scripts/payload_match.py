"""Payload-space utilities used by the Payload Match experiment."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def payload_match_weights(
    coalition_bits: np.ndarray,
    target_bits: np.ndarray,
    max_weight: float = 0.5,
) -> np.ndarray:
    """Find nonnegative mixture weights closest to one target payload."""
    coalition_size = coalition_bits.shape[0]

    def objective(weights: np.ndarray) -> float:
        error = coalition_bits.T @ weights - target_bits
        return float(np.sum(error**2))

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    result = minimize(
        objective,
        np.full(coalition_size, 1.0 / coalition_size),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * coalition_size,
        constraints=constraints,
        options={"maxiter": 800, "ftol": 1e-14},
    )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, max_weight)
    total = float(weights.sum())
    return (weights / total if total > 1e-8
            else np.full(coalition_size, 1.0 / coalition_size))


def payload_distances(
    coalition_bits: np.ndarray,
    target_bits: np.ndarray,
) -> np.ndarray:
    """Compute exact distances from many targets to the coalition convex hull."""
    coalition_bits = np.asarray(coalition_bits, dtype=np.float64)
    target_bits = np.asarray(target_bits, dtype=np.float64)
    coalition_size = coalition_bits.shape[0]
    target_count = len(target_bits)
    best_squared = np.full(target_count, np.inf, dtype=np.float64)

    # The closest point lies in a face of the convex hull. K is at most eight,
    # so all nonempty faces can be checked exactly and efficiently in batches.
    for mask in range(1, 1 << coalition_size):
        member_ids = [index for index in range(coalition_size)
                      if mask & (1 << index)]
        face = coalition_bits[member_ids]
        face_size = len(member_ids)
        kkt = np.empty((face_size + 1, face_size + 1), dtype=np.float64)
        kkt[:face_size, :face_size] = face @ face.T
        kkt[:face_size, face_size] = 1.0
        kkt[face_size, :face_size] = 1.0
        kkt[face_size, face_size] = 0.0
        right_hand_side = np.vstack((face @ target_bits.T,
                                     np.ones((1, target_count))))
        weights = np.linalg.lstsq(kkt, right_hand_side, rcond=None)[0][:face_size]
        valid = np.all(weights >= -1e-9, axis=0)
        if not np.any(valid):
            continue
        projection = weights.T @ face
        squared = np.sum((projection - target_bits) ** 2, axis=1)
        best_squared[valid] = np.minimum(best_squared[valid], squared[valid])
    return np.sqrt(best_squared)


def decoded_payload_and_margin(
    scores: np.ndarray,
    registry_ids: np.ndarray,
    target: int,
) -> tuple[int, float]:
    """Return the decoded payload and target margin in one registry."""
    registry_ids = np.asarray(registry_ids, dtype=np.int64)
    registry_scores = np.asarray(scores)[registry_ids]
    decoded = int(registry_ids[np.argsort(
        registry_scores, kind="stable")[::-1][0]])
    target_position = int(np.flatnonzero(registry_ids == target)[0])
    other_scores = np.delete(registry_scores, target_position)
    margin = float(registry_scores[target_position] - np.max(other_scores))
    return decoded, margin

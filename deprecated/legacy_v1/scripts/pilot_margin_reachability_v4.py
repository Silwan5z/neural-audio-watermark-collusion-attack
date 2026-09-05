#!/usr/bin/env python3
"""Pilot v4: joint bit-stability and registry-competition reachability.

The attack is detector agnostic. Every objective term is computed from the
colluder payloads, target payload, and active payload registry. Detector
outputs are requested only after the ten targets and waveforms are fixed.
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
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from framing import restricted_top1_and_margin  # noqa: E402
from pilot_payload_reachability_n1024 import reachability_weights  # noqa: E402
from registry import (  # noqa: E402
    NBITS, coalition_seed, full_registry_bits, full_registry_size,
    get_or_embed, int_to_bits, sample_coalition, speaker_trial_index,
)
from registry_size_control import active_registry  # noqa: E402
from watermarks import detect_many, detect_wavmark_many, get_wavmark  # noqa: E402

N_REGISTRY = 1024
N_TARGETS = 10
RESULTS = ROOT / "results" / "evaluation"
FIELDS = [
    "model", "K", "N_registry", "trial_id", "spk", "local_t",
    "coalition_ids", "candidate_pool_size", "method", "target",
    "target_rank", "payload_score", "bit_stability_score",
    "registry_margin_score", "weights", "effective_K", "max_weight",
    "solver_success", "target_hit", "target_margin", "beta_bit",
    "beta_competitor", "competitor_weight",
]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def payload_components(weights: np.ndarray, match: np.ndarray,
                       target_col: int, V_all: np.ndarray,
                       offset_all: np.ndarray, beta_bit: float,
                       beta_competitor: float) -> tuple[float, float]:
    support = match.T @ weights
    margins = support - 0.5
    bit_score = -float(
        (logsumexp(-beta_bit * margins) - np.log(len(margins))) / beta_bit
    )
    scores = V_all.T @ weights - offset_all
    rival_scores = np.delete(scores, target_col)
    smooth_rival = float(
        (logsumexp(beta_competitor * rival_scores)
         - np.log(len(rival_scores))) / beta_competitor
    )
    registry_margin = float(scores[target_col] - smooth_rival)
    return bit_score, registry_margin


def joint_weights(C: np.ndarray, target: np.ndarray, target_col: int,
                  V_all: np.ndarray, offset_all: np.ndarray,
                  min_effective_k: float, beta_bit: float,
                  beta_competitor: float,
                  competitor_weight: float) -> tuple[np.ndarray, float, float, float, bool]:
    C = np.asarray(C, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    K = len(C)
    match = (C == target[None, :]).astype(np.float64)
    rival_mask = np.ones(V_all.shape[1], dtype=bool)
    rival_mask[target_col] = False
    V_rivals = V_all[:, rival_mask]
    offset_rivals = offset_all[rival_mask]
    v_target = V_all[:, target_col]

    def objective(weights: np.ndarray) -> float:
        bit_score, registry_margin = payload_components(
            weights, match, target_col, V_all, offset_all,
            beta_bit, beta_competitor)
        return -(bit_score + competitor_weight * registry_margin)

    def gradient(weights: np.ndarray) -> np.ndarray:
        support = match.T @ weights
        bit_grad = match @ softmax(-beta_bit * (support - 0.5))
        rival_scores = V_rivals.T @ weights - offset_rivals
        registry_grad = v_target - V_rivals @ softmax(
            beta_competitor * rival_scores)
        return -(bit_grad + competitor_weight * registry_grad)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0,
         "jac": lambda weights: np.ones_like(weights)},
        {"type": "ineq",
         "fun": lambda weights: 1.0 / min_effective_k - np.sum(weights ** 2),
         "jac": lambda weights: -2.0 * weights},
    ]
    initial, _, _, _ = reachability_weights(
        C, target, min_effective_k,
        loss_mode="softmin_margin", softmin_beta=beta_bit)
    result = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * K, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    if weights.sum() <= 1e-12:
        weights = initial
        success = False
    else:
        weights /= weights.sum()
        success = bool(
            result.success
            and 1.0 / np.sum(weights ** 2) + 1e-6 >= min_effective_k
        )
        if not success:
            weights = initial
    bit_score, registry_margin = payload_components(
        weights, match, target_col, V_all, offset_all,
        beta_bit, beta_competitor)
    total = bit_score + competitor_weight * registry_margin
    return weights, total, bit_score, registry_margin, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="audioseal", choices=sorted(NBITS))
    parser.add_argument("--K", type=int, default=5, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--beta_bit", type=float, default=8.0)
    parser.add_argument("--beta_competitor", type=float, default=20.0)
    parser.add_argument("--competitor_weight", type=float, default=1.0)
    parser.add_argument("--min_effective_ratio", type=float, default=0.6)
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()
    if args.beta_bit <= 0 or args.beta_competitor <= 0:
        parser.error("temperatures must be positive")
    if args.competitor_weight < 0:
        parser.error("competitor_weight must be non-negative")

    model, K = args.model, args.K
    d = NBITS[model]
    full_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    min_effective_k = max(1.0, args.min_effective_ratio * K)
    trials = speaker_trial_index(n_total=args.n_trials)
    rows: list[dict] = []
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coalition = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        C = np.stack([int_to_bits(identity, d) for identity in coalition]).astype(float)
        active_ids = active_registry(
            full_size, coalition, N_REGISTRY, spk, K, local_t)
        candidate_ids = active_ids[
            ~np.isin(active_ids, np.asarray(coalition, dtype=np.int64))
        ]
        candidate_bits = registry_bits[candidate_ids]
        active_bits = registry_bits[active_ids].astype(float)
        signs = 2.0 * active_bits - 1.0
        V_all = C @ signs.T / d
        offset_all = 0.5 * signs.mean(axis=1)
        id_to_col = {int(identity): col for col, identity in enumerate(active_ids)}

        softmin_candidates = []
        joint_candidates = []
        for candidate_id, target_bits in zip(candidate_ids, candidate_bits):
            soft_weights, _, soft_ok, _ = reachability_weights(
                C, target_bits, min_effective_k,
                loss_mode="softmin_margin", softmin_beta=args.beta_bit)
            target_col = id_to_col[int(candidate_id)]
            match = (C == target_bits[None, :]).astype(float)
            soft_bit, soft_registry = payload_components(
                soft_weights, match, target_col, V_all, offset_all,
                args.beta_bit, args.beta_competitor)
            softmin_candidates.append((
                soft_bit, int(candidate_id), soft_weights, soft_bit,
                soft_registry, soft_ok))

            joint = joint_weights(
                C, target_bits, target_col, V_all, offset_all,
                min_effective_k, args.beta_bit, args.beta_competitor,
                args.competitor_weight)
            joint_w, total, bit_score, registry_margin, joint_ok = joint
            joint_candidates.append((
                total, int(candidate_id), joint_w, bit_score,
                registry_margin, joint_ok))

        softmin_candidates.sort(key=lambda item: (-item[0], item[1]))
        joint_candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = {
            "softmin_v2": softmin_candidates[:N_TARGETS],
            "joint_v4": joint_candidates[:N_TARGETS],
        }
        outputs = []
        metadata = []
        for method, candidates in selected.items():
            for target_rank, candidate in enumerate(candidates, start=1):
                total, target, weights, bit_score, registry_margin, ok = candidate
                outputs.append(sum(
                    weights[index] * wavs[index] for index in range(K)
                ).astype(np.float32))
                metadata.append((
                    method, target, target_rank, total, weights,
                    bit_score, registry_margin, ok))

        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for meta, (scores, _, _) in zip(metadata, decoded):
            method, target, target_rank, total, weights, bit_score, registry_margin, ok = meta
            top1, margin = restricted_top1_and_margin(scores, active_ids, target)
            rows.append({
                "model": model,
                "K": K,
                "N_registry": N_REGISTRY,
                "trial_id": trial_id,
                "spk": spk,
                "local_t": local_t,
                "coalition_ids": json.dumps(coalition),
                "candidate_pool_size": len(candidate_ids),
                "method": method,
                "target": target,
                "target_rank": target_rank,
                "payload_score": f"{total:.10f}",
                "bit_stability_score": f"{bit_score:.10f}",
                "registry_margin_score": f"{registry_margin:.10f}",
                "weights": json.dumps(weights.tolist()),
                "effective_K": f"{1.0 / np.sum(weights ** 2):.8f}",
                "max_weight": f"{np.max(weights):.8f}",
                "solver_success": int(ok),
                "target_hit": int(top1 == target),
                "target_margin": f"{margin:.8f}",
                "beta_bit": f"{args.beta_bit:.6f}",
                "beta_competitor": f"{args.beta_competitor:.6f}",
                "competitor_weight": f"{args.competitor_weight:.6f}",
            })

        print(
            f"{model} K={K}: {trial_id + 1}/{len(trials)} trials "
            f"({time.time() - start:.0f}s)", flush=True)

    tag = f".{args.output_tag}" if args.output_tag else ""
    output = RESULTS / f"joint_reachability_v4_{model}_K{K}{tag}.csv"
    write_rows_atomic(output, rows)
    print(f"completed {output} rows={len(rows)}", flush=True)
    for method in ["softmin_v2", "joint_v4"]:
        method_rows = [row for row in rows if row["method"] == method]
        hits = np.asarray([int(row["target_hit"]) for row in method_rows])
        grouped: dict[int, list[int]] = {}
        for row in method_rows:
            grouped.setdefault(int(row["trial_id"]), []).append(int(row["target_hit"]))
        any_hits = np.asarray([int(any(values)) for values in grouped.values()])
        rank_one = np.asarray([
            int(row["target_hit"]) for row in method_rows
            if int(row["target_rank"]) == 1
        ])
        effective = np.asarray([float(row["effective_K"]) for row in method_rows])
        print(
            f"{method}: single={100 * hits.mean():.1f}% "
            f"Any@10={100 * any_hits.mean():.1f}% "
            f"rank1={100 * rank_one.mean():.1f}% "
            f"mean_effective_K={effective.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()

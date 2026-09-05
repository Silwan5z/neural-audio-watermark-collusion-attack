#!/usr/bin/env python3
"""Pilot v6: detector-agnostic lower-tail bit-margin reachability."""
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
    "coalition_ids", "method", "target", "target_rank", "payload_score",
    "tail_fraction", "tail_bits", "weights", "effective_K", "max_weight",
    "solver_success", "target_hit", "target_margin",
]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def tail_margin_weights(C: np.ndarray, target: np.ndarray,
                        min_effective_k: float,
                        tail_fraction: float) -> tuple[np.ndarray, float, bool, int]:
    C = np.asarray(C, dtype=float)
    target = np.asarray(target, dtype=float)
    K, n_bits = C.shape
    match = (C == target[None, :]).astype(float)
    tail_bits = max(1, int(np.ceil(tail_fraction * n_bits)))

    def weakest(weights: np.ndarray) -> np.ndarray:
        margins = match.T @ weights - 0.5
        return np.argsort(margins, kind="stable")[:tail_bits]

    def objective(weights: np.ndarray) -> float:
        indices = weakest(weights)
        margins = match[:, indices].T @ weights - 0.5
        return -float(np.mean(margins))

    def gradient(weights: np.ndarray) -> np.ndarray:
        indices = weakest(weights)
        return -np.mean(match[:, indices], axis=1)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0,
         "jac": lambda weights: np.ones_like(weights)},
        {"type": "ineq",
         "fun": lambda weights: 1.0 / min_effective_k - np.sum(weights ** 2),
         "jac": lambda weights: -2.0 * weights},
    ]
    initial = np.full(K, 1.0 / K)
    result = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * K, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
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
    return weights, -objective(weights), success, tail_bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="audioseal", choices=sorted(NBITS))
    parser.add_argument("--K", type=int, default=5, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--beta_bit", type=float, default=8.0)
    parser.add_argument("--min_effective_ratio", type=float, default=0.6)
    parser.add_argument("--tail_fractions", default="0.25,0.5")
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()
    fractions = [float(value) for value in args.tail_fractions.split(",")]
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        parser.error("tail fractions must lie in (0, 1]")

    model, K = args.model, args.K
    d = NBITS[model]
    full_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    min_effective_k = max(1.0, args.min_effective_ratio * K)
    trials = speaker_trial_index(n_total=args.n_trials)
    method_names = ["softmin_v2"] + [f"tail_{value:g}_v6" for value in fractions]
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

        candidates: dict[str, list[tuple]] = {name: [] for name in method_names}
        for candidate_id, target_bits in zip(candidate_ids, candidate_bits):
            weights, loss, success, _ = reachability_weights(
                C, target_bits, min_effective_k,
                loss_mode="softmin_margin", softmin_beta=args.beta_bit)
            candidates["softmin_v2"].append((
                -loss, int(candidate_id), weights, success, 0.0, 0))
            for fraction in fractions:
                weights, score, success, tail_bits = tail_margin_weights(
                    C, target_bits, min_effective_k, fraction)
                candidates[f"tail_{fraction:g}_v6"].append((
                    score, int(candidate_id), weights, success,
                    fraction, tail_bits))

        outputs = []
        metadata = []
        for method in method_names:
            candidates[method].sort(key=lambda item: (-item[0], item[1]))
            for target_rank, candidate in enumerate(
                    candidates[method][:N_TARGETS], start=1):
                score, target, weights, success, fraction, tail_bits = candidate
                outputs.append(sum(
                    weights[member] * wavs[member] for member in range(K)
                ).astype(np.float32))
                metadata.append((
                    method, target, target_rank, score, weights,
                    success, fraction, tail_bits))

        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for meta, (scores, _, _) in zip(metadata, decoded):
            method, target, target_rank, score, weights, success, fraction, tail_bits = meta
            top1, margin = restricted_top1_and_margin(scores, active_ids, target)
            rows.append({
                "model": model,
                "K": K,
                "N_registry": N_REGISTRY,
                "trial_id": trial_id,
                "spk": spk,
                "local_t": local_t,
                "coalition_ids": json.dumps(coalition),
                "method": method,
                "target": target,
                "target_rank": target_rank,
                "payload_score": f"{score:.10f}",
                "tail_fraction": f"{fraction:.6f}",
                "tail_bits": tail_bits,
                "weights": json.dumps(weights.tolist()),
                "effective_K": f"{1.0 / np.sum(weights ** 2):.8f}",
                "max_weight": f"{np.max(weights):.8f}",
                "solver_success": int(success),
                "target_hit": int(top1 == target),
                "target_margin": f"{margin:.8f}",
            })
        print(
            f"{model} K={K}: {trial_id + 1}/{len(trials)} trials "
            f"({time.time() - start:.0f}s)", flush=True)

    tag = f".{args.output_tag}" if args.output_tag else ""
    output = RESULTS / f"tail_margin_v6_{model}_K{K}{tag}.csv"
    write_rows_atomic(output, rows)
    print(f"completed {output} rows={len(rows)}", flush=True)
    for method in method_names:
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
        solver = np.asarray([int(row["solver_success"]) for row in method_rows])
        print(
            f"{method}: single={100 * hits.mean():.1f}% "
            f"Any@10={100 * any_hits.mean():.1f}% "
            f"rank1={100 * rank_one.mean():.1f}% "
            f"solver={100 * solver.mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()

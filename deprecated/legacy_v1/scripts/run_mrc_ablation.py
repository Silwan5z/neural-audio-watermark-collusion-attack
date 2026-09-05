#!/usr/bin/env python3
"""Seven controlled ablations of MRC / Softmin-v2 with 10-trial checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from framing import convex_dist_batch_exact, restricted_top1_and_margin  # noqa: E402
from registry import (  # noqa: E402
    NBITS, coalition_seed, full_registry_bits, full_registry_size,
    get_or_embed, int_to_bits, sample_coalition, speaker_trial_index,
)
from registry_size_control import active_registry  # noqa: E402
from watermarks import detect_many, detect_wavmark_many, get_wavmark  # noqa: E402


RESULTS = ROOT / "results" / "evaluation"
N_REGISTRY = 1024
N_TARGETS = 10
CHECKPOINT_EVERY = 10
VARIANTS = {
    "beta5": {"beta": 5.0, "keff_frac": 0.6, "entropy": 0.0,
              "selection": "softmin", "attack": "optimized"},
    "beta10": {"beta": 10.0, "keff_frac": 0.6, "entropy": 0.0,
               "selection": "softmin", "attack": "optimized"},
    "no_keff_floor": {"beta": 8.0, "keff_frac": None, "entropy": 0.0,
                      "selection": "softmin", "attack": "optimized"},
    "uniform_floor": {"beta": 8.0, "keff_frac": 1.0, "entropy": 0.0,
                      "selection": "softmin", "attack": "optimized"},
    "entropy005": {"beta": 8.0, "keff_frac": None, "entropy": 0.05,
                   "selection": "softmin", "attack": "optimized"},
    "entropy005_keff_floor": {"beta": 8.0, "keff_frac": 0.6, "entropy": 0.05,
                              "selection": "softmin", "attack": "optimized"},
    "hull_targets": {"beta": 8.0, "keff_frac": 0.6, "entropy": 0.0,
                     "selection": "hull", "attack": "optimized"},
    "uniform_weights": {"beta": 8.0, "keff_frac": 0.6, "entropy": 0.0,
                        "selection": "softmin", "attack": "uniform"},
}
FIELDS = [
    "model", "K", "spk", "local_t", "gi", "N_registry", "variant",
    "target", "target_rank", "target_top1", "target_margin",
    "selection_policy", "attack_policy", "beta", "keff_frac", "entropy_gamma",
    "selection_score", "hull_distance", "selection_weights", "attack_weights",
    "effective_K", "max_weight", "solver_success",
]


def write_atomic(path: Path, rows: list[dict]) -> None:
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
        grouped.setdefault(int(row["gi"]), []).append(row)
    completed = {trial for trial, trial_rows in grouped.items()
                 if len(trial_rows) == N_TARGETS}
    return [row for row in rows if int(row["gi"]) in completed], completed


def optimize_softmin(C: np.ndarray, target: np.ndarray, beta: float,
                     keff_min: float | None, entropy_gamma: float
                     ) -> tuple[np.ndarray, float, bool]:
    K = C.shape[0]
    sign = 2.0 * target.astype(np.float64) - 1.0
    matrix = C.astype(np.float64) * sign[None, :]
    offset = -0.5 * sign

    def margins(weights: np.ndarray) -> np.ndarray:
        return matrix.T @ weights + offset

    def neg_objective(weights: np.ndarray) -> float:
        value = -(1.0 / beta) * logsumexp(-beta * margins(weights))
        if entropy_gamma > 0:
            clipped = np.clip(weights, 1e-12, 1.0)
            value += entropy_gamma * float(-np.sum(clipped * np.log(clipped)))
        return -value

    def neg_gradient(weights: np.ndarray) -> np.ndarray:
        values = margins(weights)
        probability = np.exp(-beta * (values - values.max()))
        probability /= probability.sum()
        gradient = matrix @ probability
        if entropy_gamma > 0:
            clipped = np.clip(weights, 1e-12, 1.0)
            gradient += entropy_gamma * (-(np.log(clipped) + 1.0))
        return -gradient

    # On the simplex, K_eff >= K has exactly one feasible point: uniform
    # weights.  Returning it directly avoids thousands of redundant SLSQP
    # solves in the uniform-floor ablation and makes the ten resulting attack
    # waveforms bit-identical, so the detector can evaluate them once.
    if keff_min is not None and keff_min >= K - 1e-12:
        weights = np.full(K, 1.0 / K)
        pure_score = float(
            -(1.0 / beta) * logsumexp(-beta * margins(weights)))
        return weights, pure_score, True

    constraints = [{
        "type": "eq", "fun": lambda weights: np.sum(weights) - 1.0,
        "jac": lambda weights: np.ones(K),
    }]
    if keff_min is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda weights: 1.0 / keff_min - np.sum(weights ** 2),
            "jac": lambda weights: -2.0 * weights,
        })
    result = minimize(
        neg_objective, np.full(K, 1.0 / K), jac=neg_gradient,
        method="SLSQP", bounds=[(0.0, 1.0)] * K, constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-14},
    )
    # SLSQP can occasionally stop at a feasible boundary point without
    # declaring success (typically a line-search status). Retry only those
    # cases from the feasible point it found; successful solves remain
    # bit-for-bit unchanged.
    if not result.success:
        retry_start = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        retry_start = (retry_start / retry_start.sum()
                       if retry_start.sum() > 1e-8
                       else np.full(K, 1.0 / K))
        result = minimize(
            neg_objective, retry_start, jac=neg_gradient,
            method="SLSQP", bounds=[(0.0, 1.0)] * K,
            constraints=constraints,
            # A tighter tolerance can report status 8 at an already converged
            # K_eff boundary point. 1e-10 preserves the objective/weights to
            # paper precision while returning a stable success status.
            options={"maxiter": 1000, "ftol": 1e-10},
        )
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights = weights / weights.sum() if weights.sum() > 1e-8 else np.full(K, 1.0 / K)
    pure_score = float(-(1.0 / beta) * logsumexp(-beta * margins(weights)))
    effective_k = 1.0 / float(np.sum(weights ** 2))
    feasible = keff_min is None or effective_k + 1e-6 >= keff_min
    return weights, pure_score, bool(result.success and feasible)


def score_target_task(task):
    """Picklable CPU-worker entry point for independent target solves."""
    C, target, nbits, beta, keff_min, entropy_gamma = task
    weights, score, success = optimize_softmin(
        C, int_to_bits(target, nbits), beta, keff_min, entropy_gamma)
    return score, int(target), weights, success, np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--model", required=True, choices=sorted(NBITS))
    parser.add_argument("--K", required=True, type=int, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", default=300, type=int)
    parser.add_argument("--trial_start", default=0, type=int)
    parser.add_argument("--trial_end", default=None, type=int)
    parser.add_argument("--output-suffix", default="",
                        help="optional isolated shard suffix; empty preserves canonical output")
    args = parser.parse_args()

    trial_end = args.n_trials if args.trial_end is None else args.trial_end
    if not 0 <= args.trial_start < trial_end <= args.n_trials:
        parser.error("require 0 <= trial_start < trial_end <= n_trials")
    config = VARIANTS[args.variant]
    model, K = args.model, args.K
    full_size = full_registry_size(model)
    if N_REGISTRY > full_size:
        parser.error(f"N=1024 exceeds native registry for {model}")
    registry_bits = full_registry_bits(model)
    trial_schedule = speaker_trial_index(n_total=args.n_trials)
    keff_min = (None if config["keff_frac"] is None
                else max(1.0, float(config["keff_frac"]) * K))

    # uniform_weights uses exactly the already-completed Softmin-v2 target
    # selection and weights.  Reuse that verified result instead of solving
    # the same 1,022 optimization problems again for every trial.
    reused_selection: dict[int, list[tuple[float, int, np.ndarray, bool, float]]] = {}
    if args.variant == "uniform_weights":
        source = RESULTS / f"margin_reachability_v2_{model}_K{K}.csv"
        if not source.exists():
            source = (ROOT / "data" / "tamper_softmin_v2_n1024" /
                      f"margin_reachability_v2_{model}_K{K}.csv")
        if not source.exists():
            raise FileNotFoundError(f"missing verified Softmin-v2 source: {source}")
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                reused_selection.setdefault(int(row["gi"]), []).append((
                    float(row["reachability_score"]), int(row["target"]),
                    np.asarray(json.loads(row["weights"]), dtype=np.float64),
                    bool(int(row["solver_success"])), np.nan))
        invalid = [gi for gi in range(args.n_trials)
                   if len(reused_selection.get(gi, [])) != N_TARGETS]
        if invalid:
            raise RuntimeError(
                f"incomplete Softmin-v2 reuse source; invalid trials={invalid[:10]}")

    target_workers = max(1, int(os.environ.get("MRC_TARGET_WORKERS", "1")))
    target_pool = None
    if (config["selection"] == "softmin" and not reused_selection
            and target_workers > 1 and keff_min != K):
        target_pool = ProcessPoolExecutor(
            max_workers=target_workers, mp_context=mp.get_context("spawn"))

    if args.output_suffix and not all(
            char.isalnum() or char in "_-" for char in args.output_suffix):
        parser.error("--output-suffix may contain only letters, digits, underscore, and hyphen")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    stem = f"mrc_ablation_{args.variant}_N1024_{model}_K{K}{suffix}"
    final = RESULTS / f"{stem}.csv"
    partial = RESULTS / f"{stem}.partial.csv"
    rows, completed = load_completed(partial)
    relevant_completed = sum(args.trial_start <= trial < trial_end for trial in completed)
    print(f"resume variant={args.variant} {model} K={K}: "
          f"{relevant_completed}/{trial_end - args.trial_start}", flush=True)
    started = time.time()
    new_since_checkpoint = 0

    for gi, (spk, local_t) in enumerate(trial_schedule):
        if gi < args.trial_start or gi >= trial_end or gi in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coalition = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        C = np.stack([int_to_bits(identity, NBITS[model]) for identity in coalition])
        active_ids = active_registry(full_size, coalition, N_REGISTRY, spk, K, local_t)
        candidate_ids = active_ids[
            ~np.isin(active_ids, np.asarray(coalition, dtype=np.int64))]

        selected: list[tuple[float, int, np.ndarray, bool, float]] = []
        if reused_selection:
            selected = reused_selection[gi]
        elif config["selection"] == "hull":
            candidate_bits = registry_bits[candidate_ids]
            distances = convex_dist_batch_exact(C, candidate_bits)
            order = np.lexsort((candidate_ids, distances))[:N_TARGETS]
            for index in order:
                target = int(candidate_ids[index])
                weights, score, success = optimize_softmin(
                    C, int_to_bits(target, NBITS[model]), float(config["beta"]),
                    keff_min, float(config["entropy"]))
                selected.append((score, target, weights, success, float(distances[index])))
        else:
            target_ids = candidate_ids.tolist()
            tasks = ((C, target, NBITS[model], float(config["beta"]),
                      keff_min, float(config["entropy"]))
                     for target in target_ids)
            if target_pool is None:
                scored = [score_target_task(task) for task in tasks]
            else:
                # Each target has an independent deterministic SLSQP problem;
                # separate processes bypass the SciPy/GIL serialization seen
                # with threads. map preserves the deterministic input order.
                scored = list(target_pool.map(
                    score_target_task, tasks, chunksize=16))
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected = scored[:N_TARGETS]

        attacks = []
        attack_weights = []
        for _, _, selection_weights, _, _ in selected:
            weights = (np.full(K, 1.0 / K) if config["attack"] == "uniform"
                       else selection_weights)
            attack_weights.append(weights)
            attacks.append(sum(weights[member] * wavs[member]
                               for member in range(K)).astype(np.float32))
        # Some ablations deliberately use the same attack weights for all ten
        # targets.  Decode each bit-identical waveform only once, then restore
        # the target order.  This changes no scores or attack semantics.
        unique_attacks: list[np.ndarray] = []
        attack_index: dict[bytes, int] = {}
        inverse: list[int] = []
        for attack in attacks:
            key = attack.tobytes()
            if key not in attack_index:
                attack_index[key] = len(unique_attacks)
                unique_attacks.append(attack)
            inverse.append(attack_index[key])
        unique_decoded = (
            detect_wavmark_many(get_wavmark(), unique_attacks, registry_bits)
            if model == "wavmark"
            else detect_many(model, unique_attacks, registry_bits))
        decoded = [unique_decoded[index] for index in inverse]

        for rank, ((score, target, selection_weights, success, hull_distance),
                   weights, (scores, _, _)) in enumerate(
                       zip(selected, attack_weights, decoded), start=1):
            top1, margin = restricted_top1_and_margin(scores, active_ids, target)
            effective_k = 1.0 / float(np.sum(weights ** 2))
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t,
                "gi": gi, "N_registry": N_REGISTRY, "variant": args.variant,
                "target": target, "target_rank": rank,
                "target_top1": int(top1 == target), "target_margin": f"{margin:.8f}",
                "selection_policy": config["selection"], "attack_policy": config["attack"],
                "beta": config["beta"],
                "keff_frac": "" if config["keff_frac"] is None else config["keff_frac"],
                "entropy_gamma": config["entropy"], "selection_score": f"{score:.8f}",
                "hull_distance": "" if np.isnan(hull_distance) else f"{hull_distance:.8f}",
                "selection_weights": json.dumps(selection_weights.tolist()),
                "attack_weights": json.dumps(weights.tolist()),
                "effective_K": f"{effective_k:.8f}", "max_weight": f"{weights.max():.8f}",
                "solver_success": int(success),
            })
        completed.add(gi)
        new_since_checkpoint += 1
        if new_since_checkpoint >= CHECKPOINT_EVERY:
            write_atomic(partial, rows)
            done = sum(args.trial_start <= trial < trial_end for trial in completed)
            print(f"checkpoint variant={args.variant} {model} K={K}: "
                  f"{done}/{trial_end - args.trial_start} ({time.time()-started:.0f}s)",
                  flush=True)
            new_since_checkpoint = 0

    if target_pool is not None:
        target_pool.shutdown(wait=True)
    write_atomic(partial, rows)
    # Publish a final only when this model-K cell covers all requested n_trials.
    all_counts = {trial: 0 for trial in range(args.n_trials)}
    for row in rows:
        all_counts[int(row["gi"])] += 1
    if all(count == N_TARGETS for count in all_counts.values()):
        write_atomic(final, rows)
        print(f"COMPLETE {final} rows={len(rows)}", flush=True)
    else:
        covered = sum(count == N_TARGETS for count in all_counts.values())
        print(f"PARTIAL {partial} complete_trials={covered}/{args.n_trials}", flush=True)


if __name__ == "__main__":
    main()

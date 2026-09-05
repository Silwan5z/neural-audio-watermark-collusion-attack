#!/usr/bin/env python3
"""DM restart-stability diagnostic (no detector evaluation required).

For each coalition, independently rebuild the multi-start SLSQP procedure used
by ``blind_distance.dm_weights``.  The final CSV has one row per (model, K)
with the paper-ready mean/median pairwise cosine statistics; the accompanying
detail CSV keeps each start's final objective for auditability.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blind_distance import dist_matrix, fwp  # noqa: E402
from registry import CAP, NBITS, coalition_seed, get_or_embed, sample_coalition, speaker_trial_index  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
SUMMARY_FIELDS = ["model", "K", "n_trials", "n_restarts", "starts_per_restart",
                  "restart_cosine_mean", "restart_cosine_median",
                  "all_start_cosine_mean", "all_start_cosine_median",
                  "restart_objective_std_mean", "detail_csv"]
DETAIL_FIELDS = ["model", "K", "spk", "local_t", "gi", "restart", "start_id",
                 "objective", "weights"]


def write_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def solve(D: np.ndarray, start: np.ndarray, cap: float) -> tuple[np.ndarray, float]:
    def objective(a):
        return -float(np.asarray(a) @ D @ np.asarray(a))
    res = minimize(objective, start, method="SLSQP", bounds=[(0, cap)] * len(start),
                   constraints=[{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}],
                   options={"maxiter": 800, "ftol": 1e-14})
    a = np.clip(res.x, 0, cap)
    a = a / a.sum() if a.sum() > 1e-8 else np.full(len(a), 1 / len(a))
    return a, -objective(a)


def one_restart(D: np.ndarray, wavs: list[np.ndarray], cap: float, seed: int,
                n_random_starts: int) -> list[tuple[np.ndarray, float]]:
    K = len(wavs)
    rng = np.random.default_rng(seed)
    starts = [np.full(K, 1 / K), fwp(wavs, K)]
    support = min(K, int(np.ceil(1 / cap)))
    for _ in range(n_random_starts):
        ids = rng.choice(K, size=support, replace=False)
        start = np.zeros(K); start[ids] = 1 / support
        starts.append(start)
    return [solve(D, start, cap) for start in starts]


def cosine_pairs(weights: list[np.ndarray]) -> list[float]:
    return [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            for a, b in itertools.combinations(weights, 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(NBITS))
    ap.add_argument("--K", required=True, type=int, choices=[5, 8])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_restarts", type=int, default=4)
    ap.add_argument("--n_random_starts", type=int, default=5)
    args = ap.parse_args()
    details: list[dict] = []
    restart_cosines: list[float] = []
    all_start_cosines: list[float] = []
    objective_stds: list[float] = []
    for gi, (spk, local_t) in enumerate(speaker_trial_index(n_total=args.n_trials)):
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(rng, args.model, args.K)
        wavs = [get_or_embed(args.model, spk, code) for code in coalition]
        n = min(map(len, wavs)); wavs = [w[:n] for w in wavs]
        D = dist_matrix(wavs)
        best_weights, all_weights, best_values = [], [], []
        for restart in range(args.n_restarts):
            seed = coalition_seed(spk, args.K, local_t) + 104729 * (restart + 1)
            converged = one_restart(D, wavs, CAP, seed, args.n_random_starts)
            weights, values = zip(*converged)
            best_index = int(np.argmax(values))
            best_weights.append(weights[best_index]); best_values.append(values[best_index])
            all_weights.extend(weights)
            for start_id, (weight, value) in enumerate(converged):
                details.append({"model": args.model, "K": args.K, "spk": spk, "local_t": local_t,
                                "gi": gi, "restart": restart, "start_id": start_id,
                                "objective": f"{value:.10f}",
                                "weights": ";".join(f"{v:.8f}" for v in weight)})
        restart_cosines.extend(cosine_pairs(best_weights))
        all_start_cosines.extend(cosine_pairs(all_weights))
        objective_stds.append(float(np.std(best_values)))
    detail_path = RESULTS / f"dm_restart_stability_detail_{args.model}_K{args.K}.csv"
    summary_path = RESULTS / f"dm_restart_stability_{args.model}_K{args.K}.csv"
    write_atomic(detail_path, details, DETAIL_FIELDS)
    summary = [{"model": args.model, "K": args.K, "n_trials": args.n_trials,
                "n_restarts": args.n_restarts, "starts_per_restart": 2 + args.n_random_starts,
                "restart_cosine_mean": f"{np.mean(restart_cosines):.6f}",
                "restart_cosine_median": f"{np.median(restart_cosines):.6f}",
                "all_start_cosine_mean": f"{np.mean(all_start_cosines):.6f}",
                "all_start_cosine_median": f"{np.median(all_start_cosines):.6f}",
                "restart_objective_std_mean": f"{np.mean(objective_stds):.10f}",
                "detail_csv": detail_path.name}]
    write_atomic(summary_path, summary, SUMMARY_FIELDS)


if __name__ == "__main__":
    main()

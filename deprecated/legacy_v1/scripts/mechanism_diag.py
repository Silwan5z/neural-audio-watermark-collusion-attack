#!/usr/bin/env python3
"""CPU-only payload-geometry diagnostic for the collusion_300 registry."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from framing import tct  # noqa: E402
from registry import CAP, NBITS, coalition_seed, int_to_bits, sample_coalition, speaker_trial_index  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
MODELS = sorted(NBITS)
KS = [2, 3, 5, 8]
FIELDS = ["model", "K", "method", "entropy_mean", "entropy_uniform_logK", "max_weight_mean",
          "max_weight_uniform_1overK", "l2_mean", "l2_uniform_1over_sqrtK", "n_trials"]
METHODS = ["mean", "bdb", "tct"]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path, n_trials: int) -> tuple[list[dict], set[tuple[str, int]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # A smoke-test checkpoint is intentionally reusable only for the same
    # sample size.  Do not let n=5 suppress a later n=300 full diagnostic.
    rows = [row for row in rows if int(row.get("n_trials", -1)) == n_trials]
    counts = Counter((row["model"], int(row["K"])) for row in rows)
    complete = {key for key, n in counts.items() if n == len(METHODS)}
    return [row for row in rows if (row["model"], int(row["K"])) in complete], complete


def entropy(a: np.ndarray) -> float:
    a = np.clip(a, 1e-12, 1.0)
    return float(-np.sum(a * np.log(a)))


def bdb(G: np.ndarray, cap: float = CAP) -> np.ndarray:
    K = G.shape[0]
    def objective(a):
        values = G @ np.asarray(a); maximum = values.max()
        return float(maximum + np.log(np.exp(values - maximum).sum()))
    res = minimize(objective, np.full(K, 1 / K), method="SLSQP", bounds=[(0, cap)] * K,
                   constraints=[{"type": "eq", "fun": lambda a: np.sum(a) - 1}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    a = np.clip(res.x, 0, cap)
    return a / a.sum() if a.sum() > 1e-8 else np.full(K, 1 / K)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[*MODELS, "all"], default="all")
    ap.add_argument("--K", type=int, choices=[*KS, 0], default=0, help="0 means all K")
    ap.add_argument("--n_trials", type=int, default=300)
    args = ap.parse_args()
    models = MODELS if args.model == "all" else [args.model]
    ks = KS if args.K == 0 else [args.K]
    out = RESULTS / "mechanism_diag.csv"
    partial = RESULTS / "mechanism_diag.partial.csv"
    rows, completed = load_completed(partial, args.n_trials)

    for model in models:
        d = NBITS[model]
        for K in ks:
            if (model, K) in completed:
                continue
            acc = {method: [] for method in METHODS}
            for spk, local_t in speaker_trial_index(n_total=args.n_trials):
                rng = np.random.default_rng(coalition_seed(spk, K, local_t))
                coalition = sample_coalition(rng, model, K)
                C = np.stack([int_to_bits(ci, d) for ci in coalition]).astype(float)
                G = (C * 2 - 1) @ (C * 2 - 1).T
                target = int(rng.integers(0, 2 ** d))
                while target in coalition:
                    target = int(rng.integers(0, 2 ** d))
                weights = {"mean": np.full(K, 1 / K), "bdb": bdb(G),
                           "tct": tct(C, int_to_bits(target, d), CAP)}
                for name, a in weights.items():
                    acc[name].append((entropy(a), a.max(), np.linalg.norm(a)))
            for method, values in acc.items():
                values = np.asarray(values)
                rows.append({"model": model, "K": K, "method": method,
                             "entropy_mean": f"{values[:, 0].mean():.4f}",
                             "entropy_uniform_logK": f"{np.log(K):.4f}",
                             "max_weight_mean": f"{values[:, 1].mean():.4f}",
                             "max_weight_uniform_1overK": f"{1 / K:.4f}",
                             "l2_mean": f"{values[:, 2].mean():.4f}",
                             "l2_uniform_1over_sqrtK": f"{1 / np.sqrt(K):.4f}",
                             "n_trials": args.n_trials})
            write_rows_atomic(partial, rows)
            print(f"checkpointed {model} K={K}", flush=True)
    write_rows_atomic(partial, rows); write_rows_atomic(out, rows)
    print(f"completed {len(rows)} diagnostic rows -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detector-aware random-search oracle on already evaluated TCT targets."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from framing import tct  # noqa: E402
from registry import (CAP, NBITS, coalition_seed, full_registry_bits, get_or_embed,
                      int_to_bits, sample_coalition)  # noqa: E402
from watermarks import detect  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
FIELDS = ["model", "K", "spk", "local_t", "gi", "target", "tct_hit", "tct_margin",
          "oracle_hit", "oracle_margin"]


def write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def margin_for(scores: np.ndarray, target: int) -> float:
    return float(scores[target] - np.delete(scores, target).max())


def sample_capped_dirichlet(rng: np.random.Generator, K: int, cap: float) -> np.ndarray:
    """Sample a true capped-simplex point; clipping then normalising can violate cap."""
    while True:
        a = rng.dirichlet(np.ones(K))
        if float(a.max()) <= cap + 1e-12:
            return a


def oracle_search(model: str, wavs: list[np.ndarray], registry_bits: np.ndarray, target: int,
                  K: int, cap: float, n_search: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    best_a, best_margin = np.full(K, 1 / K), -np.inf
    for _ in range(n_search):
        a = sample_capped_dirichlet(rng, K, cap)
        y = sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
        scores, _, _ = detect(model, y, registry_bits)
        score = margin_for(scores, target)
        if score > best_margin:
            best_a, best_margin = a, score
    return best_a, float(best_margin)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["voicemark", "wmcodec"])
    ap.add_argument("--K", required=True, type=int, choices=[5, 8])
    ap.add_argument("--n_trials", type=int, default=40, help="number of unique (coalition,target) contexts")
    ap.add_argument("--n_search", type=int, default=300)
    args = ap.parse_args()
    d = NBITS[args.model]
    registry_bits = full_registry_bits(args.model)
    source = RESULTS / f"tamper_{args.model}_K{args.K}.csv"
    with source.open(newline="") as f:
        source_rows = [r for r in csv.DictReader(f) if r.get("method") == "tct"]
    contexts, seen = [], set()
    for row in source_rows:
        key = (row["spk"], row["local_t"], row["gi"], row["target"])
        if key not in seen:
            seen.add(key); contexts.append(key)
        if len(contexts) == args.n_trials:
            break
    if len(contexts) != args.n_trials:
        raise RuntimeError(f"only found {len(contexts)} unique TCT contexts in {source}")
    out = RESULTS / f"detector_oracle_{args.model}_K{args.K}.csv"
    partial = RESULTS / f"detector_oracle_{args.model}_K{args.K}.partial.csv"
    rows: list[dict] = []
    done = set()
    if partial.exists():
        with partial.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {(r["spk"], r["local_t"], r["gi"], r["target"]) for r in rows}
    for index, (spk, local_t, gi_text, target_text) in enumerate(contexts):
        key = (spk, local_t, gi_text, target_text)
        if key in done:
            continue
        local_t_i, gi, target = int(local_t), int(gi_text), int(target_text)
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t_i))
        coalition = sample_coalition(rng, args.model, args.K)
        wavs = [get_or_embed(args.model, spk, code) for code in coalition]
        n = min(map(len, wavs)); wavs = [w[:n] for w in wavs]
        C = np.stack([int_to_bits(code, d) for code in coalition]).astype(float)
        a_tct = tct(C, int_to_bits(target, d), CAP)
        y_tct = sum(a_tct[i] * wavs[i] for i in range(args.K)).astype(np.float32)
        tct_scores, _, _ = detect(args.model, y_tct, registry_bits)
        tct_margin = margin_for(tct_scores, target)
        search_rng = np.random.default_rng(coalition_seed(spk, args.K, local_t_i) + 7919 * (target + 1))
        _, oracle_margin = oracle_search(args.model, wavs, registry_bits, target, args.K,
                                         CAP, args.n_search, search_rng)
        rows.append({"model": args.model, "K": args.K, "spk": spk, "local_t": local_t_i,
                     "gi": gi, "target": target, "tct_hit": int(tct_margin > 0),
                     "tct_margin": f"{tct_margin:.8f}", "oracle_hit": int(oracle_margin > 0),
                     "oracle_margin": f"{oracle_margin:.8f}"})
        if (index + 1) % 5 == 0:
            write_atomic(partial, rows)
    write_atomic(partial, rows)
    write_atomic(out, rows)


if __name__ == "__main__":
    main()

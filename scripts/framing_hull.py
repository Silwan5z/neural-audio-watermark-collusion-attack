#!/usr/bin/env python3
"""Convex-hull framing evidence on the collusion_300 registry."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from framing import N_CAND, convex_dist_batch_exact, tct  # noqa: E402
from registry import (CAP, NBITS, coalition_seed, full_registry_bits, full_registry_size,
                      get_or_embed, int_to_bits, sample_coalition, speaker_trial_index)  # noqa: E402
from watermarks import detect_many  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
FIELDS = ["model", "K", "spk", "local_t", "gi", "target", "method", "d_hull", "target_hit", "target_margin"]
ROWS_PER_TRIAL = 2 * N_CAND


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def load_completed_trials(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(int(row["gi"]) for row in rows)
    complete = {gi for gi, count in counts.items() if count == ROWS_PER_TRIAL}
    return [row for row in rows if int(row["gi"]) in complete], complete


def target_margin(scores: np.ndarray, target: int) -> float:
    return float(scores[target] - max(scores[i] for i in range(len(scores)) if i != target))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(NBITS))
    ap.add_argument("--K", required=True, type=int)
    ap.add_argument("--n_trials", default=300, type=int)
    args = ap.parse_args()
    model, K, d = args.model, args.K, NBITS[args.model]
    registry_bits, reg_size = full_registry_bits(model), full_registry_size(model)
    trials = speaker_trial_index(n_total=args.n_trials)
    out = RESULTS / f"framing_hull_{model}_K{K}.csv"
    partial = RESULTS / f"framing_hull_{model}_K{K}.partial.csv"
    rows, completed = load_completed_trials(partial)
    if completed:
        print(f"resuming {model} K={K}: {len(completed)}/{len(trials)} trials", flush=True)
    start = time.time()

    for gi, (spk, local_t) in enumerate(trials):
        if gi in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coalition = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coalition]
        n = min(map(len, wavs)); wavs = [w[:n] for w in wavs]
        C = np.stack([int_to_bits(ci, d) for ci in coalition]).astype(float)
        available = np.asarray([ci for ci in range(reg_size) if ci not in set(coalition)])
        candidate_rng = np.random.default_rng(coalition_seed(spk, K, local_t) + 999)
        sample_ids = candidate_rng.choice(available, size=min(2000, len(available)), replace=False)
        distances = convex_dist_batch_exact(C, registry_bits[sample_ids])
        order = np.argsort(distances)[:N_CAND]
        targets, d_hulls = sample_ids[order].tolist(), distances[order].tolist()
        mean_y = sum(w / K for w in wavs).astype(np.float32)
        tct_ys = [sum(a * w for a, w in zip(tct(C, int_to_bits(target, d), CAP), wavs)).astype(np.float32)
                  for target in targets]
        decoded = detect_many(model, [mean_y, *tct_ys], registry_bits)
        mean_scores = decoded[0][0]
        for target, d_hull, (scores, _, _) in zip(targets, d_hulls, decoded[1:]):
            rows.append({"model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
                         "target": target, "method": "mean", "d_hull": f"{d_hull:.6f}",
                         "target_hit": int(np.argmax(mean_scores) == target),
                         "target_margin": f"{target_margin(mean_scores, target):.6f}"})
            rows.append({"model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
                         "target": target, "method": "tct", "d_hull": f"{d_hull:.6f}",
                         "target_hit": int(np.argmax(scores) == target),
                         "target_margin": f"{target_margin(scores, target):.6f}"})
        if (gi + 1) % 10 == 0:
            write_rows_atomic(partial, rows)
            print(f"  {model} K={K}: {gi + 1}/{len(trials)} checkpointed ({time.time() - start:.0f}s)", flush=True)
    write_rows_atomic(partial, rows); write_rows_atomic(out, rows)
    print(f"completed {model} K={K}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

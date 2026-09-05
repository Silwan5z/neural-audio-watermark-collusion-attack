#!/usr/bin/env python3
"""Cross-system pilot for soft-min reachability and diversified attempt order."""
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
    "model", "K", "N_registry", "trial_id", "global_trial_id", "spk",
    "local_t", "coalition_ids", "target", "score_rank", "weight_diverse_rank",
    "payload_diverse_rank", "payload_score", "weights", "effective_K",
    "max_weight", "solver_success", "target_hit", "target_margin",
]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def farthest_order(representations: np.ndarray, count: int) -> list[int]:
    selected = [0]
    remaining = list(range(1, count))
    while remaining:
        scored = []
        for candidate in remaining:
            distance = min(
                float(np.linalg.norm(
                    representations[candidate] - representations[chosen]
                ))
                for chosen in selected
            )
            scored.append((distance, -candidate, candidate))
        chosen = max(scored)[2]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def summarize(rows: list[dict], rank_field: str) -> list[float]:
    curves = []
    trial_ids = sorted({int(row["trial_id"]) for row in rows})
    for trial_id in trial_ids:
        trial_rows = sorted(
            (row for row in rows if int(row["trial_id"]) == trial_id),
            key=lambda row: int(row[rank_field]),
        )
        hits = np.asarray([int(row["target_hit"]) for row in trial_rows])
        curves.append(np.maximum.accumulate(hits))
    return (100 * np.mean(np.stack(curves), axis=0)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(NBITS))
    parser.add_argument("--K", type=int, default=5, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--beta_bit", type=float, default=8.0)
    parser.add_argument("--min_effective_ratio", type=float, default=0.6)
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()

    model, K = args.model, args.K
    d = NBITS[model]
    full_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    min_effective_k = max(1.0, args.min_effective_ratio * K)
    all_trials = speaker_trial_index(n_total=args.trial_start + args.n_trials)
    trials = all_trials[args.trial_start:args.trial_start + args.n_trials]
    rows: list[dict] = []
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        global_trial_id = args.trial_start + trial_id
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

        candidates = []
        for candidate_id, target_bits in zip(candidate_ids, candidate_bits):
            weights, loss, success, _ = reachability_weights(
                C, target_bits, min_effective_k,
                loss_mode="softmin_margin", softmin_beta=args.beta_bit)
            candidates.append((
                -loss, int(candidate_id), weights, success,
                target_bits.astype(float)))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = candidates[:N_TARGETS]
        weights_matrix = np.stack([item[2] for item in selected])
        payload_matrix = np.stack([item[4] for item in selected])
        weight_order = farthest_order(weights_matrix, N_TARGETS)
        payload_order = farthest_order(payload_matrix, N_TARGETS)
        weight_ranks = np.empty(N_TARGETS, dtype=int)
        payload_ranks = np.empty(N_TARGETS, dtype=int)
        weight_ranks[weight_order] = np.arange(1, N_TARGETS + 1)
        payload_ranks[payload_order] = np.arange(1, N_TARGETS + 1)

        outputs = [sum(
            item[2][member] * wavs[member] for member in range(K)
        ).astype(np.float32) for item in selected]
        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for score_index, (item, (scores, _, _)) in enumerate(
                zip(selected, decoded)):
            score, target, weights, success, _ = item
            top1, margin = restricted_top1_and_margin(scores, active_ids, target)
            rows.append({
                "model": model,
                "K": K,
                "N_registry": N_REGISTRY,
                "trial_id": trial_id,
                "global_trial_id": global_trial_id,
                "spk": spk,
                "local_t": local_t,
                "coalition_ids": json.dumps(coalition),
                "target": target,
                "score_rank": score_index + 1,
                "weight_diverse_rank": int(weight_ranks[score_index]),
                "payload_diverse_rank": int(payload_ranks[score_index]),
                "payload_score": f"{score:.10f}",
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
    output = RESULTS / (
        f"softmin_diversity_v7_N{N_REGISTRY}_{model}_K{K}_"
        f"start{args.trial_start}{tag}.csv"
    )
    write_rows_atomic(output, rows)
    print(f"completed {output} rows={len(rows)}", flush=True)
    hits = np.asarray([int(row["target_hit"]) for row in rows])
    print(f"single-target Hit@1: {100 * hits.mean():.1f}%", flush=True)
    for rank_field in ["score_rank", "weight_diverse_rank", "payload_diverse_rank"]:
        curve = summarize(rows, rank_field)
        compact = ", ".join(
            f"{index + 1}:{value:.1f}" for index, value in enumerate(curve)
        )
        print(f"{rank_field} Any@n = {compact}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pilot v5: registry-aware target re-ranking with fixed soft-min weights.

All target scores use only colluder payloads and the active payload registry.
The detector is called only after every selected target and waveform is fixed.
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
    "target_rank", "softmin_rank", "competition_rank", "bit_score",
    "payload_competition_margin", "weights", "effective_K", "max_weight",
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


def bernoulli_competition_margin(C: np.ndarray, weights: np.ndarray,
                                 active_bits: np.ndarray,
                                 target_col: int) -> float:
    probability = np.clip(C.T @ weights, 1e-6, 1.0 - 1e-6)
    log_one = np.log(probability)
    log_zero = np.log1p(-probability)
    scores = np.mean(
        active_bits * log_one[None, :]
        + (1.0 - active_bits) * log_zero[None, :],
        axis=1,
    )
    target_score = scores[target_col]
    rival_score = np.max(np.delete(scores, target_col))
    return float(target_score - rival_score)


def ranks_descending(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((ids, -np.asarray(values)))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="audioseal", choices=sorted(NBITS))
    parser.add_argument("--K", type=int, default=5, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--beta_bit", type=float, default=8.0)
    parser.add_argument("--min_effective_ratio", type=float, default=0.6)
    parser.add_argument("--prefilter", type=int, default=50)
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()

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
        active_bits = registry_bits[active_ids].astype(float)
        candidate_ids = active_ids[
            ~np.isin(active_ids, np.asarray(coalition, dtype=np.int64))
        ]
        candidate_bits = registry_bits[candidate_ids]
        id_to_col = {int(identity): col for col, identity in enumerate(active_ids)}

        weights_all = []
        bit_scores = np.empty(len(candidate_ids), dtype=float)
        margins = np.empty(len(candidate_ids), dtype=float)
        solver_ok = np.empty(len(candidate_ids), dtype=bool)
        for index, (candidate_id, target_bits) in enumerate(
                zip(candidate_ids, candidate_bits)):
            weights, loss, success, _ = reachability_weights(
                C, target_bits, min_effective_k,
                loss_mode="softmin_margin", softmin_beta=args.beta_bit)
            weights_all.append(weights)
            bit_scores[index] = -loss
            margins[index] = bernoulli_competition_margin(
                C, weights, active_bits, id_to_col[int(candidate_id)])
            solver_ok[index] = success

        bit_ranks = ranks_descending(bit_scores, candidate_ids)
        competition_ranks = ranks_descending(margins, candidate_ids)
        soft_order = np.lexsort((candidate_ids, -bit_scores))
        shortlist = soft_order[:min(args.prefilter, len(soft_order))]
        filtered_order = shortlist[np.lexsort((
            candidate_ids[shortlist], -margins[shortlist]
        ))]
        borda_score = bit_ranks + competition_ranks
        borda_order = np.lexsort((candidate_ids, borda_score))

        selections = {
            "softmin_v2": soft_order[:N_TARGETS],
            "filtered_competition_v5": filtered_order[:N_TARGETS],
            "borda_joint_v5": borda_order[:N_TARGETS],
        }
        outputs = []
        metadata = []
        for method, selected in selections.items():
            for target_rank, candidate_index in enumerate(selected, start=1):
                weights = weights_all[int(candidate_index)]
                outputs.append(sum(
                    weights[member] * wavs[member] for member in range(K)
                ).astype(np.float32))
                metadata.append((method, target_rank, int(candidate_index), weights))

        if model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(model, outputs, registry_bits)

        for meta, (scores, _, _) in zip(metadata, decoded):
            method, target_rank, candidate_index, weights = meta
            target = int(candidate_ids[candidate_index])
            top1, detector_margin = restricted_top1_and_margin(
                scores, active_ids, target)
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
                "softmin_rank": int(bit_ranks[candidate_index]),
                "competition_rank": int(competition_ranks[candidate_index]),
                "bit_score": f"{bit_scores[candidate_index]:.10f}",
                "payload_competition_margin": f"{margins[candidate_index]:.10f}",
                "weights": json.dumps(weights.tolist()),
                "effective_K": f"{1.0 / np.sum(weights ** 2):.8f}",
                "max_weight": f"{np.max(weights):.8f}",
                "solver_success": int(solver_ok[candidate_index]),
                "target_hit": int(top1 == target),
                "target_margin": f"{detector_margin:.8f}",
            })

        print(
            f"{model} K={K}: {trial_id + 1}/{len(trials)} trials "
            f"({time.time() - start:.0f}s)", flush=True)

    tag = f".{args.output_tag}" if args.output_tag else ""
    output = RESULTS / f"payload_rerank_v5_{model}_K{K}{tag}.csv"
    write_rows_atomic(output, rows)
    print(f"completed {output} rows={len(rows)}", flush=True)
    for method in ["softmin_v2", "filtered_competition_v5", "borda_joint_v5"]:
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
        print(
            f"{method}: single={100 * hits.mean():.1f}% "
            f"Any@10={100 * any_hits.mean():.1f}% "
            f"rank1={100 * rank_one.mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate existing N=1024 softmin-v2 attacks against the native registry.

The target identities and mixture weights are read from the completed
``margin_reachability_v2_{model}_K{K}.csv`` files.  Thus N=1024 and native use
the same attacked waveform; only the attribution candidate registry changes.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from framing import restricted_top1_and_margin  # noqa: E402
from registry import (  # noqa: E402
    coalition_seed, full_registry_bits, full_registry_size, get_or_embed,
    sample_coalition,
)
from watermarks import detect_many, detect_wavmark_many, get_wavmark  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
N_CAND = 10
FIELDS = [
    "model", "K", "spk", "local_t", "gi", "selection_N_registry",
    "N_registry", "target", "method", "target_top1", "target_margin",
    "reachability_score", "weights", "effective_K", "solver_success",
]


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
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
    complete = {gi for gi, group in grouped.items() if len(group) == N_CAND}
    return [row for row in rows if int(row["gi"]) in complete], complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["audioseal", "wavmark", "voicemark", "wmcodec"])
    parser.add_argument("--K", type=int, required=True, choices=[2, 3, 5, 8])
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_end", type=int, default=300)
    parser.add_argument("--output_tag", default="")
    args = parser.parse_args()

    if not 0 <= args.trial_start < args.trial_end <= 300:
        parser.error("trial range must satisfy 0 <= start < end <= 300")

    source = RESULTS / f"margin_reachability_v2_{args.model}_K{args.K}.csv"
    if not source.exists():
        parser.error(f"missing completed N=1024 source: {source}")
    with source.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict]] = {}
    for row in source_rows:
        grouped.setdefault(int(row["gi"]), []).append(row)
    if len(grouped) != 300 or any(len(group) != N_CAND for group in grouped.values()):
        parser.error(f"source must contain 300 complete trials x {N_CAND} targets")
    if {int(row["N_registry"]) for row in source_rows} != {1024}:
        parser.error("source is not the expected N=1024 v2 result")

    full_size = full_registry_size(args.model)
    if full_size <= 1024:
        parser.error(f"{args.model} native registry is already {full_size}")
    registry_bits = full_registry_bits(args.model)
    native_ids = np.arange(full_size, dtype=np.int64)

    tag = f".{args.output_tag}" if args.output_tag else ""
    stem = f"margin_reachability_v2_native_{args.model}_K{args.K}{tag}"
    output = RESULTS / f"{stem}.csv"
    partial = RESULTS / f"{stem}.partial.csv"
    rows, completed = load_completed(partial)
    start = time.time()

    for gi in sorted(grouped):
        if gi < args.trial_start or gi >= args.trial_end:
            continue
        if gi in completed:
            continue
        trial_rows = grouped[gi]
        first = trial_rows[0]
        spk = first["spk"]
        local_t = int(first["local_t"])
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(rng, args.model, args.K)
        wavs = [get_or_embed(args.model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]

        weights = [np.asarray(json.loads(row["weights"]), dtype=np.float64)
                   for row in trial_rows]
        if any(weight.shape != (args.K,) for weight in weights):
            raise ValueError(f"invalid weight shape at trial {gi}")
        outputs = [
            sum(weight[index] * wavs[index] for index in range(args.K)).astype(np.float32)
            for weight in weights
        ]
        if args.model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), outputs, registry_bits)
        else:
            decoded = detect_many(args.model, outputs, registry_bits)

        for source_row, (scores, _, _) in zip(trial_rows, decoded):
            target = int(source_row["target"])
            top1, margin = restricted_top1_and_margin(scores, native_ids, target)
            rows.append({
                "model": args.model,
                "K": args.K,
                "spk": spk,
                "local_t": local_t,
                "gi": gi,
                "selection_N_registry": 1024,
                "N_registry": full_size,
                "target": target,
                "method": source_row["method"],
                "target_top1": int(top1 == target),
                "target_margin": f"{margin:.8f}",
                "reachability_score": source_row["reachability_score"],
                "weights": source_row["weights"],
                "effective_K": source_row["effective_K"],
                "solver_success": source_row["solver_success"],
            })
        write_rows_atomic(partial, rows)
        if (gi + 1) % 10 == 0:
            print(
                f"{stem}: {gi + 1}/300 elapsed={time.time() - start:.0f}s",
                flush=True,
            )

    write_rows_atomic(partial, rows)
    write_rows_atomic(output, rows)
    print(f"completed {output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split native tracing failures under partially occupied registries.

The analysis assumes exact payload lookup.  Every coalition member is
registered, while the remaining registered payloads are sampled uniformly
from the noncoalition payload space.  No watermark inference is required:
the expected registered-nonmember and unassigned rates follow from the
released native tracing-failure rates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
BIT_COUNTS = {
    "audioseal": 16,
    "wavmark": 16,
    "timbrewm": 10,
    "voicemark": 16,
    "wmcodec": 16,
}
OCCUPANCIES = (1, 10, 50, 100)


def split_escape(escape_pct: float, bit_count: int, k: int,
                 registry_size: int) -> tuple[float, float]:
    payload_count = 2 ** bit_count
    nonmember_count = payload_count - k
    registered_fraction = (registry_size - k) / nonmember_count
    unassigned_fraction = (payload_count - registry_size) / nonmember_count
    return (escape_pct * registered_fraction,
            escape_pct * unassigned_fraction)


def native_escape(model: str, k: int) -> tuple[int, float]:
    if k < 8:
        path = ROOT / "data" / "average" / f"k{k}" / f"{model}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        escaped = sum(int(row["escaped"]) for row in rows)
    else:
        paths = sorted((ROOT / "data" / "average" / "k8" / model)
                       .glob("trial_*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8"))
                for path in paths]
        escaped = sum(int(row["escaped"]) for row in rows)
    if len(rows) != 300:
        raise RuntimeError(f"{model} K={k}: expected 300 released trials")
    return len(rows), 100.0 * escaped / len(rows)


def aggregate(frame: pd.DataFrame, grouping: list[str], count_name: str
              ) -> pd.DataFrame:
    metrics = (
        "coalition_trace_pct", "registered_nonmember_pct", "unassigned_pct",
        "escape_pct", "ideal_coalition_trace_pct",
        "ideal_registered_nonmember_pct", "ideal_unassigned_pct",
        "ideal_escape_pct",
    )
    result = (frame.groupby(grouping, as_index=False)
              .agg(**{count_name: ("model", "size")},
                   **{metric: (metric, "mean") for metric in metrics}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "registry_occupancy")
    args = parser.parse_args()

    ideals = pd.read_csv(
        ROOT / "data" / "summary" / "ideal_tracing_failure.csv")
    ideal_map = {
        (int(row.bit_count), int(row.k)): float(row.mean_pct)
        for row in ideals.itertuples()
    }

    rows = []
    for model in MODELS:
        for k in (2, 3, 5, 8):
            bit_count = BIT_COUNTS[model]
            payload_count = 2 ** bit_count
            n_trials, escape_pct = native_escape(model, k)
            ideal_escape = ideal_map[(bit_count, k)]
            for occupancy in OCCUPANCIES:
                registry_size = max(k, round(payload_count * occupancy / 100))
                registered, unassigned = split_escape(
                    escape_pct, bit_count, k, registry_size)
                ideal_registered, ideal_unassigned = split_escape(
                    ideal_escape, bit_count, k, registry_size)
                rows.append({
                    "model": model,
                    "k": k,
                    "bit_count": bit_count,
                    "n_trials": n_trials,
                    "native_escape_pct": escape_pct,
                    "ideal_native_escape_pct": ideal_escape,
                    "occupancy_pct": occupancy,
                    "registry_size": registry_size,
                    "coalition_trace_pct": 100.0 - escape_pct,
                    "registered_nonmember_pct": registered,
                    "unassigned_pct": unassigned,
                    "escape_pct": escape_pct,
                    "ideal_coalition_trace_pct": 100.0 - ideal_escape,
                    "ideal_registered_nonmember_pct": ideal_registered,
                    "ideal_unassigned_pct": ideal_unassigned,
                    "ideal_escape_pct": ideal_escape,
                })

    detail = pd.DataFrame(rows).sort_values(
        ["model", "k", "occupancy_pct"]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "registry_occupancy_by_system_k.csv", index=False)

    all_k = aggregate(detail, ["occupancy_pct"], "cells")
    all_k.to_csv(
        args.output_dir / "registry_occupancy_all_k_average.csv", index=False)

    k2 = aggregate(
        detail[detail.k == 2], ["occupancy_pct"], "systems")
    k2.to_csv(
        args.output_dir / "registry_occupancy_k2_average.csv", index=False)

    print(k2.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

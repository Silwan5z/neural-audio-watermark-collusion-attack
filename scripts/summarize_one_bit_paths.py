#!/usr/bin/env python3
"""Reduce the full one-bit mixture paths to the per-trial evidence used in Fig. 3."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def summarize(path: Path, model: str) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            trial = int(row["trial_id"])
            item = grouped.setdefault(trial, {
                "model": model,
                "trial_id": trial,
                "base_payload": int(row["base_payload"]),
                "flipped_payload": int(row["flipped_payload"]),
                "decoded": set(),
                "n_points": 0,
            })
            item["decoded"].add(int(row["decoded_identity"]))
            item["n_points"] += 1
    if set(grouped) != set(range(300)):
        raise RuntimeError(f"{model}: expected trials 0--299, got {len(grouped)}")

    rows = []
    for trial in range(300):
        item = grouped[trial]
        endpoints = {item["base_payload"], item["flipped_payload"]}
        decoded = item.pop("decoded")
        other = sorted(decoded - endpoints)
        rows.append({
            **item,
            "direct": int(not other),
            "contains_other_payload": int(bool(other)),
            "n_decoded_payloads": len(decoded),
            "other_payload_count": len(other),
        })
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audioseal", type=Path,
        default=root / "data" / "mixture_path_k5_300clips_20260902" / "shards" /
        "mixture_path_audioseal_shard0of1.partial.csv")
    parser.add_argument(
        "--voicemark", type=Path,
        default=root / "data" / "mixture_path_k5_300clips_20260902" / "shards" /
        "mixture_path_voicemark_shard0of1.csv")
    parser.add_argument(
        "--output", type=Path,
        default=root / "data" / "one_bit" / "path_summary.csv")
    args = parser.parse_args()

    rows = summarize(args.audioseal, "audioseal") + summarize(args.voicemark, "voicemark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for model in ("audioseal", "voicemark"):
        subset = [row for row in rows if row["model"] == model]
        direct = sum(int(row["direct"]) for row in subset)
        print(f"{model}: direct={direct}, other={300 - direct}")


if __name__ == "__main__":
    main()

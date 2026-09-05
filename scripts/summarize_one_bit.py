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
                "decoded_payloads": set(),
                "weight_count": 0,
            })
            item["decoded_payloads"].add(int(row["decoded_payload"]))
            item["weight_count"] += 1
    if set(grouped) != set(range(300)):
        raise RuntimeError(f"{model}: expected trials 0--299, got {len(grouped)}")

    rows = []
    for trial in range(300):
        item = grouped[trial]
        endpoints = {item["base_payload"], item["flipped_payload"]}
        decoded = item.pop("decoded_payloads")
        other = sorted(decoded - endpoints)
        rows.append({
            **item,
            "direct_path": int(not other),
            "has_other_payload": int(bool(other)),
            "unique_payload_count": len(decoded),
            "other_payload_count": len(other),
        })
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audioseal", type=Path,
        default=root / "results" / "one_bit" / "paths" / "shards" /
        "audioseal.shard0of1.csv")
    parser.add_argument(
        "--voicemark", type=Path,
        default=root / "results" / "one_bit" / "paths" / "shards" /
        "voicemark.shard0of1.csv")
    parser.add_argument(
        "--output", type=Path,
        default=root / "data" / "one_bit" / "paths.csv")
    args = parser.parse_args()

    rows = summarize(args.audioseal, "audioseal") + summarize(args.voicemark, "voicemark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for model in ("audioseal", "voicemark"):
        subset = [row for row in rows if row["model"] == model]
        direct = sum(int(row["direct_path"]) for row in subset)
        print(f"{model}: direct={direct}, other={300 - direct}")


if __name__ == "__main__":
    main()

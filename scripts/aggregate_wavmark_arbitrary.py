"""Aggregate completed WavMark arbitrary-target TCT detail outputs."""
from __future__ import annotations

import csv
import os
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"
FIELDS = ["model", "K", "n_trials", "n_targets_per_trial", "per_target_hit_rate",
          "any_of_10_hit_rate"]


def main() -> None:
    summary = []
    for K in [2, 3, 5, 8]:
        path = EVAL / f"tamper_arbitrary_detail_wavmark_K{K}.csv"
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 3000:
            raise RuntimeError(f"{path.name}: expected 3000 rows, got {len(rows)}")
        grouped: dict[int, list[int]] = defaultdict(list)
        hits = []
        for row in rows:
            hit = int(row["target_hit"])
            hits.append(hit)
            grouped[int(row["trial_id"])].append(hit)
        if len(grouped) != 300 or {len(v) for v in grouped.values()} != {10}:
            raise RuntimeError(f"{path.name}: malformed trial/target grouping")
        summary.append({
            "model": "wavmark", "K": K, "n_trials": 300,
            "n_targets_per_trial": 10,
            "per_target_hit_rate": f"{sum(hits) / len(hits):.8f}",
            "any_of_10_hit_rate": f"{sum(max(v) for v in grouped.values()) / len(grouped):.8f}",
        })
    out = EVAL / "tamper_arbitrary_detail_wavmark_summary.csv"
    tmp = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    os.replace(tmp, out)
    print(f"wrote {out}: {len(summary)} rows")


if __name__ == "__main__":
    main()

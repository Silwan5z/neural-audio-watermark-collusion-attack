"""Merge single-copy top-10 tamper shards into canonical 300-trial CSVs."""
from __future__ import annotations

import csv
import os
import uuid
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"
ATTACK_MODELS = {
    "overwrite": ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"],
    "ifgsm": ["audioseal", "voicemark", "wmcodec", "timbrewm"],
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def merge(attack: str, model: str) -> None:
    stem = f"single_tamper_{attack}_{model}"
    paths = sorted(
        path for path in RESULTS.glob(f"{stem}.shard_*_*.csv")
        if not path.name.endswith(".partial.csv")
    )
    if not paths:
        raise FileNotFoundError(f"no completed shards for {attack}/{model}")

    merged: dict[tuple[int, int, int], dict[str, str]] = {}
    for path in paths:
        for row in read_rows(path):
            key = (int(row["trial_id"]), int(row["target_id"]),
                   int(row["N_registry"]))
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate {key} in {path}")
            merged[key] = row

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for (trial_id, _, _), row in merged.items():
        grouped[trial_id].append(row)
    expected_trials = set(range(300))
    if set(grouped) != expected_trials:
        missing = sorted(expected_trials - set(grouped))
        extra = sorted(set(grouped) - expected_trials)
        raise ValueError(
            f"{attack}/{model}: incomplete coverage; missing={missing}, extra={extra}")

    registries_per_trial = 1 if model == "timbrewm" else 2
    expected_rows = 10 * registries_per_trial
    for trial_id, trial_rows in grouped.items():
        if len(trial_rows) != expected_rows:
            raise ValueError(
                f"{attack}/{model} trial={trial_id}: "
                f"expected {expected_rows} rows, got {len(trial_rows)}")
        ranks = {int(row["candidate_rank"]) for row in trial_rows}
        if ranks != set(range(1, 11)):
            raise ValueError(
                f"{attack}/{model} trial={trial_id}: invalid target ranks {ranks}")
        for registry in {int(row["N_registry"]) for row in trial_rows}:
            registry_rows = [row for row in trial_rows
                             if int(row["N_registry"]) == registry]
            hits = sum(int(row["target_hit"]) for row in registry_rows)
            if any(int(row["successful_targets_in_top10"]) != hits
                   for row in registry_rows):
                raise ValueError(
                    f"{attack}/{model} trial={trial_id}: inconsistent top-10 hit count")

    rows = [row for trial_id in range(300)
            for row in sorted(
                grouped[trial_id],
                key=lambda item: (int(item["N_registry"]),
                                  int(item["candidate_rank"])))]
    output = RESULTS / f"{stem}.csv"
    write_atomic(output, rows)
    print(f"merged {attack}/{model}: rows={len(rows)} shards={len(paths)} -> {output}")


def main() -> None:
    for attack, models in ATTACK_MODELS.items():
        for model in models:
            merge(attack, model)


if __name__ == "__main__":
    main()

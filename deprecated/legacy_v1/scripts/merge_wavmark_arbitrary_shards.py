"""Atomically merge disjoint WavMark arbitrary-target detail shards."""
from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path

from wavmark_arbitrary_tct import FIELDS, N_CAND


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, choices=[2, 3, 5, 8])
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("parts", nargs="+")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    output = root / "results" / "evaluation" / f"tamper_arbitrary_detail_wavmark_K{args.K}.csv"
    rows_by_key: dict[tuple[int, int], dict] = {}
    for name in args.parts:
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        for row in read_rows(path):
            key = (int(row["trial_id"]), int(row["target_id"]))
            if key in rows_by_key:
                raise RuntimeError(f"duplicate row {key} while reading {path}")
            rows_by_key[key] = row

    rows = sorted(rows_by_key.values(), key=lambda r: (int(r["trial_id"]), int(r["target_id"])))
    trial_counts: dict[int, int] = {}
    for row in rows:
        trial = int(row["trial_id"])
        trial_counts[trial] = trial_counts.get(trial, 0) + 1
    expected_trials = set(range(args.n_trials))
    if set(trial_counts) != expected_trials:
        missing = sorted(expected_trials - set(trial_counts))
        raise RuntimeError(f"missing trials: {missing[:20]}")
    bad = {trial: count for trial, count in trial_counts.items() if count != N_CAND}
    if bad:
        raise RuntimeError(f"trials without exactly {N_CAND} targets: {bad}")

    write_atomic(output, rows)
    write_atomic(output.with_name(output.stem + ".partial.csv"), rows)
    print(f"merged {len(rows)} rows into {output}")


if __name__ == "__main__":
    main()

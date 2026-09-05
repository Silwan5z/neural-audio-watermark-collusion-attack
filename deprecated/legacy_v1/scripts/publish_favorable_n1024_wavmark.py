#!/usr/bin/env python3
"""Atomically publish the four completed WavMark favorable-N1024 cells."""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"
DATA = ROOT / "data"
KS = [2, 3, 5, 8]


def inspect(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    counts = Counter(int(row["gi"]) for row in rows)
    if len(rows) != 6000 or set(counts) != set(range(300)) or set(counts.values()) != {20}:
        raise ValueError(f"invalid completed cell {path}: rows={len(rows)} trials={len(counts)}")
    return len(rows), columns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    published = []
    target_dir = DATA / "tamper"
    target_dir.mkdir(parents=True, exist_ok=True)
    for k in KS:
        name = f"tamper_N1024_wavmark_K{k}.csv"
        source = RESULTS / name
        rows, columns = inspect(source)
        target = target_dir / name
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
        published.append({
            "category": "tamper", "file": name, "rows": str(rows),
            "columns": ";".join(columns), "bytes": str(target.stat().st_size),
            "sha256": sha256(target),
            "runtime_source": str(source.relative_to(ROOT)),
        })

    index = DATA / "INDEX.csv"
    with index.open(newline="") as handle:
        existing = list(csv.DictReader(handle))
    names = {row["file"] for row in published}
    combined = [row for row in existing if row["file"] not in names] + published
    combined.sort(key=lambda row: (row["category"], row["file"]))
    fields = ["category", "file", "rows", "columns", "bytes", "sha256", "runtime_source"]
    tmp_index = index.with_name(f".{index.name}.{uuid.uuid4().hex}.tmp")
    with tmp_index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    os.replace(tmp_index, index)
    print("PASS published WavMark favorable N=1024: K=2,3,5,8")


if __name__ == "__main__":
    main()

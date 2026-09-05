#!/usr/bin/env python3
"""Write checksums and row counts for the active paper data tree."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def row_count(path: Path) -> str:
    if path.suffix != ".csv":
        return ""
    with path.open("rb") as handle:
        return str(max(0, sum(1 for _ in handle) - 1))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    output = data / "manifest.csv"
    active = (
        "average", "coalitions", "targets", "targeted", "one_bit",
        "confidence", "summary",
    )
    files = [path for name in active for path in sorted((data / name).rglob("*"))
             if path.is_file()]
    rows = [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "rows": row_count(path),
        "sha256": digest(path),
    } for path in files]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "bytes", "rows", "sha256"),
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} files")


if __name__ == "__main__":
    main()

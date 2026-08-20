"""Atomically publish completed runtime CSVs into the tracked ``data/`` tree."""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "evaluation"
DESTINATION = ROOT / "data"


def category(name: str) -> str:
    ordered_prefixes = [
        ("attack_ecc_", "ecc"),
        ("attack_", "attack"),
        ("tamper_arbitrary_N1024_", "tamper_arbitrary_matched_n1024"),
        ("tamper_arbitrary_detail_", "tamper_arbitrary_detail"),
        ("tamper_arbitrary_", "tamper_arbitrary"),
        ("tamper_", "tamper"),
        ("framing_hull_", "framing_hull"),
        ("detector_oracle_", "detector_oracle"),
        ("registry_control_", "registry_control"),
        ("quality_presence_", "quality_presence"),
        ("temporal_sensitivity_", "temporal_sensitivity"),
        ("codec_sensitivity_", "codec_sensitivity"),
        ("evidence_chain_", "evidence_chain"),
        ("dm_restart_stability_", "dm_restart_stability"),
        ("pilot_", "pilot"),
        ("pulse_noise_", "pulse_noise"),
        ("baselines_", "baselines"),
        ("rp_", "rp"),
        ("eep_", "eep"),
        ("dm_", "dm"),
        ("bdb_", "bdb"),
        ("pgr_", "pgr"),
    ]
    if name == "mechanism_diag.csv":
        return "mechanism_diag"
    for prefix, group in ordered_prefixes:
        if name.startswith(prefix):
            return group
    return "other"


def inspect_csv(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            columns = next(reader)
        except StopIteration:
            return 0, []
        return sum(1 for _ in reader), columns


def publishable(path: Path) -> tuple[bool, int, list[str]]:
    if path.name.endswith(".partial.csv") or ".shard_" in path.name:
        return False, 0, []
    rows, columns = inspect_csv(path)
    # A one-trial smoke file uses the final filename.  Never publish it as a
    # completed matched-registry experiment.
    if path.name.startswith("tamper_arbitrary_N1024_") and rows != 6000:
        return False, rows, columns
    return rows > 0, rows, columns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    selected = []
    for path in sorted(SOURCE.glob("*.csv")):
        keep, rows, columns = publishable(path)
        if keep:
            selected.append((path, rows, columns))
    if not selected:
        raise RuntimeError(f"no completed CSV files found in {SOURCE}")

    token = uuid.uuid4().hex
    staging = ROOT / f".data.staging.{token}"
    backup = ROOT / f".data.backup.{token}"
    staging.mkdir()
    inventory = []
    counts: Counter[str] = Counter()
    for source, rows, columns in selected:
        group = category(source.name)
        target_dir = staging / group
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        counts[group] += 1
        inventory.append({
            "category": group,
            "file": source.name,
            "rows": rows,
            "columns": ";".join(columns),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "runtime_source": str(source.relative_to(ROOT)),
        })

    with (staging / "INDEX.csv").open("w", newline="") as f:
        fields = ["category", "file", "rows", "columns", "bytes", "sha256", "runtime_source"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")
    readme = [
        "# Published experiment data",
        "",
        f"Generated from completed runtime outputs at {generated_at}.",
        "",
        "`data/` is the canonical, version-controlled result tree. Runtime checkpoints and logs",
        "remain under the ignored local `results/` directory. Audio datasets, caches, and model",
        "weights are not included.",
        "",
        "Matched `N=1024` experiments retain native payload bits and restrict only the independently",
        "sampled active candidate registry; they do not truncate 16-bit payloads.",
        "",
        "## Categories",
        "",
    ]
    readme.extend(f"- `{group}/`: {counts[group]} CSV files" for group in sorted(counts))
    readme.extend([
        "",
        f"Total: {len(inventory)} completed CSV files.",
        "",
        "See `INDEX.csv` for row counts, schemas, byte sizes, runtime source paths, and SHA-256 checksums.",
    ])
    (staging / "README.md").write_text("\n".join(readme) + "\n")

    # The user explicitly requested replacing data/.  The swap is atomic and
    # the old tracked tree remains recoverable from Git history.
    if DESTINATION.exists():
        os.replace(DESTINATION, backup)
    try:
        os.replace(staging, DESTINATION)
    except Exception:
        if backup.exists() and not DESTINATION.exists():
            os.replace(backup, DESTINATION)
        raise
    if backup.exists():
        shutil.rmtree(backup)

    table = ["| Category | Files |", "|---|---:|"]
    table.extend(f"| `{group}` | {counts[group]} |" for group in sorted(counts))
    inventory_doc = [
        "# Experiment data inventory",
        "",
        f"Generated: {generated_at}.",
        "",
        "The canonical publication results are stored in `data/`. Local `results/` is runtime-only",
        "and is intentionally not tracked. The audio dataset under `dataset/collusion_300/` is also",
        "excluded from Git.",
        "",
        "The main protocol uses 100 bilingual speakers (50 English and 50 Chinese), three 10-second",
        "clips per speaker, K in {2,3,5,8}, and 300 deterministic trials unless a pilot or robustness",
        "experiment explicitly specifies otherwise.",
        "",
        "Matched-registry controls use N=1024 independently sampled native identities, include every",
        "coalition member, and preserve all native payload bits.",
        "",
        "## Published categories",
        "",
        *table,
        "",
        f"Total: {len(inventory)} completed CSV files.",
        "",
        "See `data/INDEX.csv` for the complete machine-readable inventory.",
    ]
    (ROOT / "DATA_INVENTORY.md").write_text("\n".join(inventory_doc) + "\n")
    print(f"published {len(inventory)} completed CSV files into {DESTINATION}")


if __name__ == "__main__":
    main()

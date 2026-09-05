#!/usr/bin/env python3
"""Append PESQ/STOI to a completed MRC-ablation CSV using cached copies."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from registry import CACHE_DIR, coalition_seed, sample_coalition  # noqa: E402
from watermarks import pesq_wb, stoi  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
CHECKPOINTS = ROOT / "results" / "quality_backfill" / "mrc_variants"
REFERENCE = "first_legitimate_watermarked_coalition_copy"
QUALITY_FIELDS = ["quality_reference", "PESQ", "STOI"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_atomic(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_cached(model: str, spk: str, identity: int) -> np.ndarray:
    path = CACHE_DIR / model / spk / f"{identity}.wav"
    if not path.exists():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000 or len(audio) < 16000 or not np.isfinite(audio).all():
        raise ValueError(f"invalid cached waveform: {path}")
    return np.asarray(audio, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model", required=True,
                        choices=["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"])
    parser.add_argument("--K", required=True, type=int, choices=[2, 3, 5, 8])
    args = parser.parse_args()
    source = RESULTS / f"mrc_ablation_{args.variant}_N1024_{args.model}_K{args.K}.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = read_rows(source)
    if len(rows) != 3000:
        raise ValueError(f"expected 3000 rows, got {len(rows)}: {source}")
    if all(all(row.get(field, "") for field in QUALITY_FIELDS) for row in rows):
        print(f"quality already complete: {source}", flush=True)
        return
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["gi"]), []).append(row)
    if set(grouped) != set(range(300)) or any(len(value) != 10 for value in grouped.values()):
        raise ValueError(f"invalid 300x10 trial layout: {source}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / f"{args.variant}_{args.model}_K{args.K}.csv"
    quality: dict[tuple[int, int], dict[str, str]] = {}
    if checkpoint.exists():
        quality = {(int(row["gi"]), int(row["target"])): row
                   for row in read_rows(checkpoint)}

    for gi in range(300):
        trial_rows = grouped[gi]
        if all((gi, int(row["target"])) in quality for row in trial_rows):
            continue
        first = trial_rows[0]
        spk, local_t = first["spk"], int(first["local_t"])
        rng = np.random.default_rng(coalition_seed(spk, args.K, local_t))
        coalition = sample_coalition(rng, args.model, args.K)
        waveforms = [load_cached(args.model, spk, identity) for identity in coalition]
        length = min(map(len, waveforms))
        waveforms = [waveform[:length] for waveform in waveforms]
        reference = waveforms[0]
        for row in trial_rows:
            target = int(row["target"])
            weights = np.asarray(json.loads(row["attack_weights"]), dtype=np.float64)
            if weights.shape != (args.K,) or not np.isfinite(weights).all():
                raise ValueError(f"invalid weights gi={gi}, target={target}")
            attacked = np.asarray(sum(weights[i] * waveforms[i] for i in range(args.K)),
                                  dtype=np.float32)
            quality[(gi, target)] = {
                "gi": str(gi), "target": str(target), "quality_reference": REFERENCE,
                "PESQ": f"{pesq_wb(reference, attacked):.4f}",
                "STOI": f"{stoi(reference, attacked):.4f}",
            }
        write_atomic(checkpoint, [quality[key] for key in sorted(quality)],
                     ["gi", "target", *QUALITY_FIELDS])
        if (gi + 1) % 10 == 0:
            print(f"quality {args.variant} {args.model} K={args.K}: {gi+1}/300", flush=True)

    if len(quality) != 3000:
        raise ValueError(f"incomplete quality checkpoint: {len(quality)}")
    base_fields = [field for field in rows[0] if field not in QUALITY_FIELDS]
    for row in rows:
        row.update({field: quality[(int(row["gi"]), int(row["target"]))][field]
                    for field in QUALITY_FIELDS})
    write_atomic(source, rows, [*base_fields, *QUALITY_FIELDS])
    checkpoint.unlink(missing_ok=True)
    print(f"QUALITY_COMPLETE {source}", flush=True)


if __name__ == "__main__":
    main()

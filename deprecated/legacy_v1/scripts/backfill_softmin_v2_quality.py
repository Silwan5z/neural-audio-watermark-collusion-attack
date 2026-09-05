"""Backfill PESQ/STOI/SI-SDR for softmin-v2 using a watermarked reference.

The N=1024 and native evaluations share the exact same attacked waveform, so
quality is reconstructed once from the N=1024 weights and copied to both CSVs.
The reference is the first legitimate watermarked coalition copy, matching the
paper's collusion-quality protocol.
"""
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

from registry import (  # noqa: E402
    CACHE_DIR, coalition_seed, sample_coalition,
)
from watermarks import pesq_wb, si_sdr, stoi  # noqa: E402


EVALUATION = ROOT / "results" / "evaluation"
CHECKPOINTS = ROOT / "results" / "quality_backfill"
MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
KS = [2, 3, 5, 8]
QUALITY_REFERENCE = "first_legitimate_watermarked_coalition_copy"
QUALITY_FIELDS = ["quality_reference", "PESQ", "STOI", "SI_SDR"]
CHECKPOINT_FIELDS = ["gi", "target", *QUALITY_FIELDS]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_atomic(path: Path, rows: list[dict[str, str]],
                 fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0])
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def load_checkpoint(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_rows(path)
    return {(int(row["gi"]), int(row["target"])): row for row in rows}


def load_cached_watermarked(model: str, spk: str, identity: int) -> np.ndarray:
    """Load only an existing cache entry; quality backfill must never use a GPU."""
    path = CACHE_DIR / model / spk / f"{identity}.wav"
    if not path.exists():
        raise FileNotFoundError(f"missing required watermarked cache: {path}")
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 16000 or len(audio) < 16000 or not np.isfinite(audio).all():
        raise ValueError(f"invalid required watermarked cache: {path}")
    return np.asarray(audio, dtype=np.float32)


def augment(path: Path,
            quality: dict[tuple[int, int], dict[str, str]]) -> None:
    rows = read_rows(path)
    if len(rows) != 3000:
        raise ValueError(f"expected 3000 rows in {path}, got {len(rows)}")
    base_fields = [field for field in rows[0] if field not in QUALITY_FIELDS]
    for row in rows:
        key = (int(row["gi"]), int(row["target"]))
        if key not in quality:
            raise ValueError(f"missing quality for {key} while augmenting {path}")
        row.update({field: quality[key][field] for field in QUALITY_FIELDS})
    write_atomic(path, rows, [*base_fields, *QUALITY_FIELDS])


def backfill(model: str, k: int) -> None:
    source = EVALUATION / f"margin_reachability_v2_{model}_K{k}.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    source_rows = read_rows(source)
    if len(source_rows) != 3000:
        raise ValueError(f"expected 3000 rows in {source}, got {len(source_rows)}")
    native = EVALUATION / f"margin_reachability_v2_native_{model}_K{k}.csv"
    source_has_quality = all(
        all(row.get(field, "") != "" for field in QUALITY_FIELDS)
        for row in source_rows
    )
    native_has_quality = False
    if native.exists():
        native_rows = read_rows(native)
        native_has_quality = len(native_rows) == 3000 and all(
            all(row.get(field, "") != "" for field in QUALITY_FIELDS)
            for row in native_rows
        )
    if source_has_quality and (model == "timbrewm" or native_has_quality):
        print(f"quality already complete {model} K={k}; skipping", flush=True)
        return
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault(int(row["gi"]), []).append(row)
    if set(grouped) != set(range(300)) or any(len(rows) != 10 for rows in grouped.values()):
        raise ValueError(f"invalid trial layout in {source}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS / f"softmin_v2_quality_{model}_K{k}.csv"
    quality = load_checkpoint(checkpoint_path)

    for gi in range(300):
        trial_rows = grouped[gi]
        if all((gi, int(row["target"])) in quality for row in trial_rows):
            continue
        first = trial_rows[0]
        spk = first["spk"]
        local_t = int(first["local_t"])
        rng = np.random.default_rng(coalition_seed(spk, k, local_t))
        coalition = sample_coalition(rng, model, k)
        wavs = [load_cached_watermarked(model, spk, identity)
                for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        reference = wavs[0]

        for row in trial_rows:
            target = int(row["target"])
            weights = np.asarray(json.loads(row["weights"]), dtype=np.float64)
            if weights.shape != (k,):
                raise ValueError(f"invalid K={k} weights for gi={gi}, target={target}")
            attacked = sum(weights[index] * wavs[index] for index in range(k))
            attacked = np.asarray(attacked, dtype=np.float32)
            quality[(gi, target)] = {
                "gi": str(gi),
                "target": str(target),
                "quality_reference": QUALITY_REFERENCE,
                "PESQ": f"{pesq_wb(reference, attacked):.4f}",
                "STOI": f"{stoi(reference, attacked):.4f}",
                "SI_SDR": f"{si_sdr(reference, attacked):.2f}",
            }
        write_atomic(
            checkpoint_path,
            [quality[key] for key in sorted(quality)],
            CHECKPOINT_FIELDS,
        )
        if (gi + 1) % 10 == 0:
            print(f"quality {model} K={k}: {gi + 1}/300", flush=True)

    if len(quality) != 3000:
        raise ValueError(f"quality checkpoint has {len(quality)} rows, expected 3000")
    if not source_has_quality:
        augment(source, quality)
    if native.exists():
        if not native_has_quality:
            augment(native, quality)
    if model == "timbrewm" or native.exists():
        checkpoint_path.unlink(missing_ok=True)
    print(f"quality complete {model} K={k}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--K", type=int, choices=KS)
    args = parser.parse_args()
    if (args.model is None) != (args.K is None):
        parser.error("--model and --K must be supplied together")
    if args.model is not None:
        backfill(args.model, args.K)
        return
    for model in MODELS:
        for k in KS:
            backfill(model, k)


if __name__ == "__main__":
    main()

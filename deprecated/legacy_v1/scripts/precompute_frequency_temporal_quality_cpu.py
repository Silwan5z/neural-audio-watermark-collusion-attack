#!/usr/bin/env python3
"""CPU-only, resumable PESQ/STOI cache for precomputed ablation WAVs."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_frequency_temporal_k5 import quality_job  # noqa: E402
from run_frequency_temporal_k5_shard import CONDITIONS, MODELS  # noqa: E402

FIELDS = ["model", "trial_id", "condition", "reference_audio_path",
          "output_audio_path", "PESQ", "STOI"]


def atomic(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path,
                   default=ROOT / "results" / "frequency_temporal_k5_20260828")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    partial = args.input_dir / "cpu_quality" / "frequency_temporal_quality.partial.csv"
    final = args.input_dir / "cpu_quality" / "frequency_temporal_quality.csv"
    source = final if final.exists() else partial
    rows = []
    if source.exists():
        with source.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    done = {(r["model"], int(r["trial_id"]), r["condition"]) for r in rows}
    specs = []
    for model in MODELS:
        for trial in range(300):
            marker = args.input_dir / "precomputed" / model / f"trial_{trial:03d}.json"
            if not marker.exists(): raise FileNotFoundError(marker)
            import json
            meta = json.loads(marker.read_text(encoding="utf-8"))
            ref = ROOT / "cache" / model / meta["spk"] / f"{meta['coalition_payloads'][0]}.wav"
            for condition in CONDITIONS:
                key = (model, trial, condition)
                if key in done: continue
                out = args.input_dir / "audio" / model / f"trial_{trial:03d}" / f"{condition}.wav"
                specs.append((key, str(ref), str(out)))
    jobs = [(i, ref, out) for i, (_, ref, out) in enumerate(specs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, (idx, pesq, st) in enumerate(pool.map(quality_job, jobs, chunksize=8), 1):
            (model, trial, condition), ref, out = specs[idx]
            rows.append({"model": model, "trial_id": trial, "condition": condition,
                         "reference_audio_path": ref, "output_audio_path": out,
                         "PESQ": "" if not np.isfinite(pesq) else f"{pesq:.8f}",
                         "STOI": "" if not np.isfinite(st) else f"{st:.8f}"})
            if count % 250 == 0:
                rows.sort(key=lambda r: (MODELS.index(r["model"]), int(r["trial_id"]),
                                         CONDITIONS.index(r["condition"])))
                atomic(partial, rows)
                print(f"quality checkpoint {len(rows)}/12000", flush=True)
    rows.sort(key=lambda r: (MODELS.index(r["model"]), int(r["trial_id"]),
                             CONDITIONS.index(r["condition"])))
    keys = {(r["model"], int(r["trial_id"]), r["condition"]) for r in rows}
    if len(rows) != 12000 or len(keys) != 12000:
        raise ValueError(f"quality cache incomplete: rows={len(rows)} keys={len(keys)}")
    atomic(partial, rows); atomic(final, rows)
    print(f"quality complete rows={len(rows)} output={final}")


if __name__ == "__main__":
    main()

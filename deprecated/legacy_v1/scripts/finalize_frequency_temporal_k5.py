#!/usr/bin/env python3
"""Merge shards, backfill PESQ/STOI on CPU, summarize, and plot."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from watermarks import pesq_wb, stoi  # noqa: E402
from run_frequency_temporal_k5_shard import CONDITIONS, FIELDS, MODELS  # noqa: E402

N_TRIALS = 300
N_SHARDS = 7
SEED = 20260828


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path,
                   default=ROOT / "results" / "frequency_temporal_k5_20260828")
    p.add_argument("--data-dir", type=Path,
                   default=ROOT / "data" / "frequency_temporal_k5_20260828")
    p.add_argument("--quality-workers", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=10000)
    return p.parse_args()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quality_job(item: tuple[int, str, str]) -> tuple[int, float, float]:
    idx, ref_path, out_path = item
    ref, sr0 = sf.read(ref_path, dtype="float32")
    out, sr1 = sf.read(out_path, dtype="float32")
    if sr0 != 16000 or sr1 != 16000:
        raise ValueError(f"unexpected sample rate: {sr0}, {sr1}")
    return idx, pesq_wb(ref, out), stoi(ref, out)


def cluster_indices(rows: list[dict], nboot: int):
    speakers = sorted({r["spk"] for r in rows})
    if len(speakers) != 100:
        raise ValueError(f"expected 100 speakers, found {len(speakers)}")
    by = {s: np.asarray([i for i, r in enumerate(rows) if r["spk"] == s]) for s in speakers}
    rng = np.random.default_rng(SEED)
    for _ in range(nboot):
        chosen = rng.integers(0, len(speakers), len(speakers))
        yield np.concatenate([by[speakers[i]] for i in chosen])


def summarize(all_rows: list[dict], nboot: int) -> list[dict]:
    out = []
    for model in MODELS:
        for condition in CONDITIONS:
            rr = [r for r in all_rows if r["model"] == model and r["condition"] == condition]
            rr.sort(key=lambda r: int(r["trial_id"]))
            if len(rr) != N_TRIALS:
                raise ValueError(f"{model}/{condition}: expected 300 rows, got {len(rr)}")
            boot_idx = list(cluster_indices(rr, nboot))
            for metric in ("tracing_failure", "NCA", "PESQ", "STOI"):
                vals = np.asarray([float(r[metric]) if r[metric] != "" else np.nan for r in rr])
                valid = np.isfinite(vals)
                boots = []
                for idx in boot_idx:
                    x = vals[idx]; x = x[np.isfinite(x)]
                    boots.append(float(np.mean(x)) if len(x) else np.nan)
                low, high = np.nanquantile(boots, [0.025, 0.975])
                out.append({
                    "model": model, "K": 5, "condition": condition,
                    "condition_family": rr[0]["condition_family"], "metric": metric,
                    "mean": float(np.nanmean(vals)), "ci95_low": float(low),
                    "ci95_high": float(high), "n_trials": len(rr),
                    "n_valid": int(valid.sum()), "n_speakers": 100,
                    "bootstrap_unit": "speaker", "n_bootstrap": nboot, "seed": SEED,
                })
    return out


def plot_summary(summary: list[dict], outdir: Path) -> None:
    mpl.rcParams.update({"font.size": 7.2, "axes.labelsize": 7.4, "axes.titlesize": 7.6,
                         "xtick.labelsize": 6.4, "ytick.labelsize": 6.6,
                         "legend.fontsize": 6.2, "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = {"freq_0_1k": "0–1", "freq_1_2k": "1–2", "freq_2_4k": "2–4",
              "freq_4_8k": "4–8", "speech_only": "Speech", "non_speech_only": "Non-speech",
              "full_waveform": "Full", "random_mask": "Random"}
    colors = dict(zip(MODELS, ["#2878B5", "#D95F59", "#59A14F", "#B07AA1", "#F28E2B"]))
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.2))
    panels = [("frequency", "tracing_failure", "Tracing failure"),
              ("frequency", "NCA", "NCA"),
              ("temporal", "tracing_failure", "Tracing failure"),
              ("temporal", "NCA", "NCA")]
    for ax, (family, metric, ylabel) in zip(axes.ravel(), panels):
        conds = CONDITIONS[:4] if family == "frequency" else CONDITIONS[4:]
        x = np.arange(len(conds))
        for mi, model in enumerate(MODELS):
            rows = [next(r for r in summary if r["model"] == model and
                         r["condition"] == c and r["metric"] == metric) for c in conds]
            offset = (mi - 2) * 0.075
            means = np.asarray([r["mean"] for r in rows])
            low = np.asarray([r["ci95_low"] for r in rows])
            high = np.asarray([r["ci95_high"] for r in rows])
            ax.errorbar(x + offset, means, yerr=[means-low, high-means], fmt="o", ms=3,
                        capsize=1.7, color=colors[model], label=model if ax is axes[0,0] else None)
        ax.set_xticks(x, [labels[c] for c in conds])
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2, linewidth=.5)
        ax.set_title(("Frequency-band averaging" if family == "frequency" else "Temporal-region averaging"),
                     loc="left", fontweight="bold")
    axes[0,0].legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"frequency_temporal_k5_summary.{ext}", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main():
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    merged_all = []
    for model in MODELS:
        rows = []
        for shard in range(N_SHARDS):
            p = args.input_dir / "shards" / f"frequency_temporal_k5_{model}_shard{shard}of{N_SHARDS}.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            rows.extend(read_csv(p))
        keys = [(int(r["trial_id"]), r["condition"]) for r in rows]
        if len(rows) != N_TRIALS * len(CONDITIONS) or len(keys) != len(set(keys)):
            raise ValueError(f"invalid merged cardinality for {model}: {len(rows)}")
        if {k for _, k in keys} != set(CONDITIONS):
            raise ValueError(f"missing conditions for {model}")
        for r in rows:
            if not Path(r["output_audio_path"]).exists():
                raise FileNotFoundError(r["output_audio_path"])
            if int(r["sample_rate"]) != 16000:
                raise ValueError("sample-rate mismatch")
        rows.sort(key=lambda r: (int(r["trial_id"]), CONDITIONS.index(r["condition"])))
        merged_all.extend(rows)

    quality_cache = args.input_dir / "cpu_quality" / "frequency_temporal_quality.csv"
    if quality_cache.exists():
        cached = read_csv(quality_cache)
        qmap = {(r["model"], int(r["trial_id"]), r["condition"]): r for r in cached}
        if len(cached) != 12000 or len(qmap) != 12000:
            raise ValueError(f"invalid CPU quality cache: {quality_cache}")
        for row in merged_all:
            q = qmap[(row["model"], int(row["trial_id"]), row["condition"])]
            if q["output_audio_path"] != row["output_audio_path"]:
                raise ValueError("quality-cache audio path mismatch")
            row["PESQ"], row["STOI"] = q["PESQ"], q["STOI"]
        print("reused complete CPU PESQ/STOI cache", flush=True)
    else:
        jobs = [(i, r["reference_audio_path"], r["output_audio_path"]) for i, r in enumerate(merged_all)]
        with ProcessPoolExecutor(max_workers=args.quality_workers) as pool:
            for count, (idx, pesq, st) in enumerate(pool.map(quality_job, jobs, chunksize=8), 1):
                merged_all[idx]["PESQ"] = "" if not np.isfinite(pesq) else f"{pesq:.8f}"
                merged_all[idx]["STOI"] = "" if not np.isfinite(st) else f"{st:.8f}"
                if count % 500 == 0:
                    print(f"quality {count}/{len(jobs)}", flush=True)

    for model in MODELS:
        rows = [r for r in merged_all if r["model"] == model]
        atomic_csv(args.data_dir / f"frequency_temporal_k5_{model}.csv", rows, FIELDS)
    summary = summarize(merged_all, args.bootstrap)
    atomic_csv(args.data_dir / "frequency_temporal_k5_summary.csv", summary, list(summary[0]))
    plot_summary(summary, args.data_dir)
    audit = {
        "K": 5, "systems": MODELS, "trials_per_system": N_TRIALS,
        "conditions": CONDITIONS, "attack": "uniform Mean collusion",
        "quality_reference": "first coalition watermarked copy",
        "audio_retained": True, "audio_root": str(args.input_dir / "audio"),
        "mask_root": str(args.input_dir / "masks"), "sample_rate": 16000,
        "frequency_processing": "STFT Hann n_fft=1024 hop=256; complementary raised-cosine low-pass differences; 100-Hz transitions",
        "temporal_processing": "librosa energy VAD top_db=35; Gaussian smoothing; random circular shift with equal mask mass",
        "bootstrap_seed": SEED, "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "speaker", "quality_workers": args.quality_workers,
    }
    (args.data_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"finalized {len(merged_all)} trial-condition rows -> {args.data_dir}")


if __name__ == "__main__":
    main()

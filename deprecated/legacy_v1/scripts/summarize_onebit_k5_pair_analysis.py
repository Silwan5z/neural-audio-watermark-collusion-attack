#!/usr/bin/env python3
"""Validate, bootstrap, summarize, and plot the K=5 one-bit experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from registry import load_clean  # noqa: E402

MODELS = ("audioseal", "voicemark")
DISPLAY = {"audioseal": "AudioSeal", "voicemark": "VoiceMark"}
COLORS = {"audioseal": "#2878B5", "voicemark": "#D95F59"}
SEED = 20260828
N_BOOT = 10000
N_TRIALS = 300
METRICS = (
    "waveform_residual_mse",
    "mel_residual_distance",
    "directional_evidence_change",
    "unchanged_bit_leakage",
    "decoded_hamming",
    "base_payload_correct",
    "flipped_payload_correct",
    "base_changed_bit_correct",
    "flipped_changed_bit_correct",
    "band_fraction_0_1k",
    "band_fraction_1_2k",
    "band_fraction_2_4k",
    "band_fraction_4_8k",
)
PAIR_KEYS = (
    "trial_id", "spk", "local_t", "coalition", "coalition_member_index",
    "base_payload", "flipped_payload", "flipped_bit",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path,
                   default=ROOT / "results" / "one_bit_k5_20260828")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "one_bit_k5_20260828" / "summary")
    p.add_argument("--bootstrap", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["trial_id"]))
    if len(rows) != N_TRIALS:
        raise ValueError(f"{path}: expected 300 rows, found {len(rows)}")
    ids = [int(r["trial_id"]) for r in rows]
    if ids != list(range(N_TRIALS)):
        raise ValueError(f"{path}: trial IDs are not exactly 0..299")
    if len({(r["spk"], int(r["local_t"])) for r in rows}) != N_TRIALS:
        raise ValueError(f"{path}: duplicate speaker/local_t trials")
    return rows


def values(rows: list[dict], metric: str) -> np.ndarray:
    x = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite values in {metric}")
    return x


def cluster_boot_indices(rows: list[dict], n_boot: int, rng: np.random.Generator):
    speakers = sorted({r["spk"] for r in rows})
    by_spk = {s: np.asarray([i for i, r in enumerate(rows) if r["spk"] == s])
              for s in speakers}
    if len(speakers) != 100 or set(map(len, by_spk.values())) != {3}:
        raise ValueError("expected 100 speakers with exactly three trials each")
    for _ in range(n_boot):
        chosen = rng.integers(0, len(speakers), size=len(speakers))
        yield np.concatenate([by_spk[speakers[i]] for i in chosen])


def ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(x, [0.025, 0.975]).tolist())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.labelsize": 7.6,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def errorbar(ax, x, mean, low, high, model, label=None):
    ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                ms=4.0, capsize=2.3, capthick=0.8, color=COLORS[model],
                markeredgecolor="white", markeredgewidth=0.55, label=label, zorder=3)


def aggregate_figure(outdir: Path, summary: list[dict], comparison: list[dict]) -> None:
    lookup = {(r["model"], r["metric"]): r for r in summary}
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.05))

    ax = axes[0]
    for x, model in enumerate(MODELS):
        r = lookup[(model, "log10_waveform_residual_mse")]
        errorbar(ax, x, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]), model)
    ax.set_xticks([0, 1], [DISPLAY[m] for m in MODELS])
    ax.set_ylabel(r"$\log_{10}$ waveform MSE")
    ax.set_title("(a) One-bit waveform change", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    ax = axes[1]
    band_metrics = ["band_fraction_0_1k", "band_fraction_1_2k",
                    "band_fraction_2_4k", "band_fraction_4_8k"]
    labels = ["0–1", "1–2", "2–4", "4–8"]
    x = np.arange(4)
    for offset, model in zip((-0.12, 0.12), MODELS):
        for j, metric in enumerate(band_metrics):
            r = lookup[(model, metric)]
            errorbar(ax, x[j] + offset, 100 * float(r["mean"]),
                     100 * float(r["ci_low"]), 100 * float(r["ci_high"]), model,
                     DISPLAY[model] if j == 0 else None)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Frequency band (kHz)")
    ax.set_ylabel("Delta energy (%)")
    ax.set_title("(b) Spectral allocation", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, loc="best")

    ax = axes[2]
    for x0, model in enumerate(MODELS):
        r = lookup[(model, "directional_evidence_change")]
        errorbar(ax, x0, float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]), model)
    ax.axhline(0, color="#777777", linewidth=0.65, linestyle="--", zorder=1)
    ax.set_xticks([0, 1], [DISPLAY[m] for m in MODELS])
    ax.set_ylabel("Directional decoder change")
    ax.set_title("(c) Decoder response", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    fig.tight_layout(w_pad=1.6)
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"onebit_audioseal_voicemark_comparison.{ext}",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav


def mel_case_figure(outdir: Path, data: dict[str, list[dict]]) -> list[dict]:
    """Plot representative (nearest-to-median), not outcome-selected, cases."""
    configure_plotting()
    selected = []
    maps = []
    for model in MODELS:
        rows = data[model]
        x = values(rows, "mel_residual_distance")
        idx = int(np.argmin(np.abs(x - np.median(x))))
        row = rows[idx]
        clean = load_clean(row["spk"])
        w0 = load_wav(row["base_path"])
        w1 = load_wav(row["flipped_path"])
        n = min(len(clean), len(w0), len(w1))
        delta = w1[:n] - w0[:n]
        mel = librosa.feature.melspectrogram(
            y=delta, sr=16000, n_fft=1024, win_length=1024, hop_length=256,
            n_mels=80, fmin=0.0, fmax=8000.0, power=2.0,
        )
        logmel = 10.0 * np.log10(mel + 1e-12)
        maps.append(logmel)
        selected.append({
            "model": model,
            "selection_rule": "nearest_to_system_median_mel_residual_distance",
            "trial_id": int(row["trial_id"]),
            "spk": row["spk"],
            "flipped_bit": int(row["flipped_bit"]),
            "mel_residual_distance": float(row["mel_residual_distance"]),
        })

    vmin = float(np.quantile(np.concatenate([m.ravel() for m in maps]), 0.02))
    vmax = float(np.quantile(np.concatenate([m.ravel() for m in maps]), 0.98))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.15), sharey=True)
    image = None
    for ax, model, mel, meta in zip(axes, MODELS, maps, selected):
        image = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                          vmin=vmin, vmax=vmax, extent=[0, 10, 0, 8])
        ax.set_title(f"{DISPLAY[model]} (trial {meta['trial_id']}, bit {meta['flipped_bit']})")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Mel frequency (kHz)")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label("One-bit residual energy (dB)")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.19, top=0.87, wspace=0.13)
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"onebit_mel_case_study.{ext}",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {m: read_rows(args.input_dir / f"onebit_k5_{m}_full.csv") for m in MODELS}

    for i in range(N_TRIALS):
        for key in PAIR_KEYS:
            if data["audioseal"][i][key] != data["voicemark"][i][key]:
                raise ValueError(f"cross-system pairing mismatch at trial={i} key={key}")
    for model, rows in data.items():
        if {int(r["K"]) for r in rows} != {5} or {r["model"] for r in rows} != {model}:
            raise ValueError(f"model/K mismatch for {model}")
        max_parseval = max(abs(float(r["band_energy_sum_error"])) for r in rows)
        max_mse = max(float(r["waveform_residual_mse"]) for r in rows)
        if max_parseval > max(1e-12, max_mse * 1e-6):
            raise ValueError(f"Parseval check failed for {model}: {max_parseval}")

    rng = np.random.default_rng(args.seed)
    boot_indices = list(cluster_boot_indices(data["audioseal"], args.bootstrap, rng))
    summary = []
    comparison = []
    metric_arrays = {m: {metric: values(data[m], metric) for metric in METRICS} for m in MODELS}
    for model in MODELS:
        metric_arrays[model]["log10_waveform_residual_mse"] = np.log10(
            np.maximum(metric_arrays[model]["waveform_residual_mse"], 1e-20)
        )
        metric_arrays[model]["log10_mel_residual_distance"] = np.log10(
            np.maximum(metric_arrays[model]["mel_residual_distance"], 1e-20)
        )

    all_metrics = list(METRICS) + ["log10_waveform_residual_mse", "log10_mel_residual_distance"]
    for metric in all_metrics:
        boot_by_model = {}
        for model in MODELS:
            x = metric_arrays[model][metric]
            boot = np.asarray([x[idx].mean() for idx in boot_indices])
            low, high = ci(boot)
            summary.append({
                "model": model, "K": 5, "metric": metric, "mean": float(x.mean()),
                "ci_low": low, "ci_high": high, "n_trials": N_TRIALS,
                "n_speakers": 100, "bootstrap_unit": "speaker", "n_bootstrap": args.bootstrap,
                "seed": args.seed,
            })
            boot_by_model[model] = boot
        a = metric_arrays["audioseal"][metric]
        v = metric_arrays["voicemark"][metric]
        diff = v - a
        boot_diff = boot_by_model["voicemark"] - boot_by_model["audioseal"]
        low, high = ci(boot_diff)
        comparison.append({
            "metric": metric, "K": 5, "audioseal_mean": float(a.mean()),
            "voicemark_mean": float(v.mean()), "voicemark_minus_audioseal": float(diff.mean()),
            "difference_ci_low": low, "difference_ci_high": high,
            "n_paired_trials": N_TRIALS, "n_speakers": 100,
            "bootstrap_unit": "speaker", "n_bootstrap": args.bootstrap, "seed": args.seed,
        })

    write_csv(args.output_dir / "onebit_k5_system_summary.csv", summary)
    write_csv(args.output_dir / "onebit_k5_paired_comparison.csv", comparison)
    aggregate_figure(args.output_dir, summary, comparison)
    selected = mel_case_figure(args.output_dir, data)
    with (args.output_dir / "mel_case_selection.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    audit = {
        "experiment": "K=5 one-bit payload perturbation",
        "systems": list(MODELS),
        "n_trials_per_system": N_TRIALS,
        "n_speakers": 100,
        "trials_per_speaker": 3,
        "schedule_seed": SEED,
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "speaker (three paired trials retained as a cluster)",
        "pairing_keys": list(PAIR_KEYS),
        "mel": {"sample_rate": 16000, "n_fft": 1024, "win_length": 1024,
                "hop_length": 256, "n_mels": 80, "fmin": 0, "fmax": 8000},
        "frequency_bands_hz": [list(b) for b in ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000))],
        "case_selection": "nearest trial to each system's median Mel residual distance",
        "input_files": [str(args.input_dir / f"onebit_k5_{m}_full.csv") for m in MODELS],
    }
    with (args.output_dir / "analysis_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"summary complete: {args.output_dir}")


if __name__ == "__main__":
    main()

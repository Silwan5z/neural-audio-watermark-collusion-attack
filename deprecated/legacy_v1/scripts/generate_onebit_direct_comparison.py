#!/usr/bin/env python3
"""Draw direct, low-abstraction one-bit AudioSeal/VoiceMark comparisons."""
from __future__ import annotations

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
from registry import clean_path_v19  # noqa: E402

SR = 16000
RESULT_ROOT = ROOT / "results" / "one_bit_k5_20260828"
OUT = RESULT_ROOT / "figure_candidates_direct_20260829"
SELECTIONS = {
    "clear_typical": 223,
    "representative_median": 136,
    "spectral_contrast": 99,
}
BLUE = "#0077BB"
ORANGE = "#EE7733"
RED = "#CC3311"
GREY = "#555555"
BANDS = ("0–1", "1–2", "2–4", "4–8")
BAND_COLORS = ("#88CCEE", "#44AA99", "#DDCC77", "#CC6677")


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def read_rows(model: str) -> dict[int, dict[str, str]]:
    path = RESULT_ROOT / f"onebit_k5_{model}_full.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {int(r["trial_id"]): r for r in csv.DictReader(f)}


def load(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.asarray(wav, dtype=np.float32)


def strongest_speech_window(clean: np.ndarray, seconds: float = 0.8) -> tuple[int, int]:
    n = min(int(seconds * SR), len(clean))
    hop = max(1, n // 10)
    candidates = []
    for start in range(0, max(1, len(clean) - n + 1), hop):
        candidates.append((float(np.mean(clean[start:start+n].astype(np.float64) ** 2)), start))
    start = max(candidates)[1]
    return start, start + n


def smooth_abs(x: np.ndarray, milliseconds: float = 8.0) -> np.ndarray:
    width = max(1, int(SR * milliseconds / 1000))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.abs(x).astype(np.float64), kernel, mode="same")


def prepare(rows: dict[str, dict[int, dict[str, str]]], trial: int) -> dict:
    loaded = {}
    for model in ("audioseal", "voicemark"):
        r = rows[model][trial]
        clean = load(str(clean_path_v19(r["spk"])))
        base = load(r["base_path"])
        flip = load(r["flipped_path"])
        n = min(len(clean), len(base), len(flip))
        clean, base, flip = clean[:n], base[:n], flip[:n]
        loaded[model] = {
            "row": r,
            "clean": clean,
            "base": base,
            "flip": flip,
            "delta": flip - base,
        }
    return loaded


def draw(label: str, trial: int, rows: dict[str, dict[int, dict[str, str]]]) -> dict:
    x = prepare(rows, trial)
    start, end = strongest_speech_window(x["audioseal"]["clean"])
    t = np.arange(end - start) / SR
    base_row = x["audioseal"]["row"]

    fig = plt.figure(figsize=(7.12, 3.45))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.88, wspace=0.50)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.22, 1.02))

    # (a) The source audio and both one-bit variants on the ordinary audio scale.
    ax = fig.add_subplot(gs[0, 0])
    clean = x["audioseal"]["clean"][start:end]
    scale = max(float(np.max(np.abs(clean))), 1e-8)
    traces = [
        (clean / scale + 2.4, GREY, "Clean"),
        (x["audioseal"]["base"][start:end] / scale + 1.2, BLUE, "AudioSeal: payload A"),
        (x["audioseal"]["flip"][start:end] / scale + 1.2, ORANGE, "AudioSeal: payload B"),
        (x["voicemark"]["base"][start:end] / scale, BLUE, "VoiceMark: payload A"),
        (x["voicemark"]["flip"][start:end] / scale, ORANGE, "VoiceMark: payload B"),
    ]
    for y, color, _ in traces:
        ax.plot(t, y, color=color, lw=0.65, alpha=0.92)
    ax.set_yticks([2.4, 1.2, 0.0])
    ax.set_yticklabels(["Clean", "AudioSeal\nA / B", "VoiceMark\nA / B"])
    ax.set_ylim(-0.72, 3.12)
    ax.set_xlabel("Selected speech window (s)")
    ax.set_title("(a) Only one payload bit changes")
    ax.text(0.5, 0.02, f"payload bit {base_row['flipped_bit']} flipped",
            transform=ax.transAxes, ha="center", va="bottom", color=RED,
            fontsize=8.2, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3EE", ec=RED, lw=0.8))
    ax.legend(handles=[
        mpl.lines.Line2D([], [], color=BLUE, lw=1.5, label="payload A"),
        mpl.lines.Line2D([], [], color=ORANGE, lw=1.5, label="payload B"),
    ], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))

    # (b) Subtraction makes the magnitude contrast explicit on one true scale.
    ax = fig.add_subplot(gs[0, 1])
    env_a = smooth_abs(x["audioseal"]["delta"][start:end])
    env_v = smooth_abs(x["voicemark"]["delta"][start:end])
    ymax = max(float(np.max(env_v)), float(np.max(env_a)), 1e-9)
    ax.fill_between(t, 0, env_v, color=ORANGE, alpha=0.78, linewidth=0,
                    label="VoiceMark")
    ax.plot(t, env_v, color=RED, lw=0.7)
    ax.plot(t, env_a, color=BLUE, lw=1.1, label="AudioSeal")
    ax.set_ylim(0, 1.12 * ymax)
    ax.set_xlabel("Selected speech window (s)")
    ax.set_ylabel(r"Smoothed $|x_B-x_A|$")
    ax.set_title("(b) Subtract the two versions")
    ax.text(0.10, 0.43, "VoiceMark residual", transform=ax.transAxes,
            ha="left", va="center", color=RED, fontsize=8.3, weight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))
    ax.annotate("AudioSeal is nearly flat\non the same scale",
                xy=(t[int(np.argmax(env_a))], float(np.max(env_a))),
                xytext=(0.05, 0.82), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1),
                color=BLUE, fontsize=8, weight="bold")
    threshold = float(np.quantile(env_v, 0.90))
    mask = env_v >= threshold
    ax.fill_between(t, 0, env_v, where=mask, color=RED, alpha=0.82,
                    linewidth=0, label="largest differences")
    ax.text(0.98, 0.05, "red = largest 10%",
            transform=ax.transAxes, ha="right", va="bottom", color=RED,
            fontsize=7.5, weight="bold")

    # (c) Quantify magnitude and make frequency allocation readable as percentages.
    axb = fig.add_subplot(gs[0, 2])
    mse_a = float(x["audioseal"]["row"]["waveform_residual_mse"])
    mse_v = float(x["voicemark"]["row"]["waveform_residual_mse"])
    mel_a = float(x["audioseal"]["row"]["mel_residual_distance"])
    mel_v = float(x["voicemark"]["row"]["mel_residual_distance"])
    mse_ratio = mse_v / mse_a
    mel_ratio = mel_v / mel_a
    axb.set_title("(c) VoiceMark changes much more")
    axb.set_xlim(0, 100)
    axb.set_ylim(-0.62, 3.62)
    axb.text(50, 3.18, f"{mse_ratio:.0f}×", ha="center", va="center",
             fontsize=24, color=RED, weight="bold")
    axb.text(50, 2.74, "larger waveform residual", ha="center",
             va="center", fontsize=9.2)
    axb.text(50, 2.30, f"Mel residual: {mel_ratio:.1f}× larger",
             ha="center", va="center", fontsize=9.2, color="#8B3A2B", weight="bold",
             bbox=dict(boxstyle="round,pad=0.28", fc="#FFF3EE", ec="#DD9A87"))
    axb.text(50, 1.76, "Residual energy by frequency", ha="center",
             va="center", fontsize=9, weight="bold")
    left = np.zeros(2)
    for band, color in zip(("0_1k", "1_2k", "2_4k", "4_8k"), BAND_COLORS):
        vals = np.array([
            100 * float(x["audioseal"]["row"][f"band_fraction_{band}"]),
            100 * float(x["voicemark"]["row"][f"band_fraction_{band}"]),
        ])
        bars = axb.barh([1.14, 0.40], vals, left=left, height=0.42, color=color,
                        edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, vals):
            if value >= 10:
                axb.text(bar.get_x() + bar.get_width()/2,
                         bar.get_y() + bar.get_height()/2,
                         f"{value:.0f}%", ha="center", va="center", fontsize=7.2,
                         color="#222222", weight="bold")
        left += vals
    axb.set_yticks([1.14, 0.40], ["AudioSeal", "VoiceMark"])
    axb.set_xticks([0, 50, 100])
    axb.set_xlabel("One-bit residual energy (%)")
    axb.legend(BANDS, title="Frequency (kHz)", ncol=2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.52),
               columnspacing=0.8, handlelength=1.2)
    axb.spines["left"].set_visible(False)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"onebit_direct_{label}_trial{trial:03d}.{ext}")
    plt.close(fig)
    return {
        "label": label,
        "trial_id": trial,
        "speaker": base_row["spk"],
        "flipped_bit": int(base_row["flipped_bit"]),
        "waveform_mse_ratio_voicemark_over_audioseal": mse_ratio,
        "mel_distance_ratio_voicemark_over_audioseal": mel_ratio,
        "audioseal_band_percent": [100 * float(x["audioseal"]["row"][f"band_fraction_{b}"])
                                   for b in ("0_1k", "1_2k", "2_4k", "4_8k")],
        "voicemark_band_percent": [100 * float(x["voicemark"]["row"][f"band_fraction_{b}"])
                                   for b in ("0_1k", "1_2k", "2_4k", "4_8k")],
    }


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {model: read_rows(model) for model in ("audioseal", "voicemark")}
    audit = []
    for label, trial in SELECTIONS.items():
        for model in rows:
            r = rows[model][trial]
            assert int(r["base_payload_correct"]) == 1
            assert int(r["flipped_payload_correct"]) == 1
        audit.append(draw(label, trial, rows))
    with (OUT / "direct_figure_audit.json").open("w", encoding="utf-8") as f:
        json.dump({"selection_pool": 269, "candidates": audit}, f, indent=2)
    print(f"generated {len(audit)} direct-comparison figures in {OUT}")


if __name__ == "__main__":
    main()

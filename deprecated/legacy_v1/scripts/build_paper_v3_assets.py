#!/usr/bin/env python3
"""Build revised paper assets from existing trial-level results only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "main_text_revision_20260830" / "k8_population"
OUT = ROOT / "data" / "paper_v3_revision_20260830" / "k8_source_valid"
FIG = ROOT / "paper_rewriting_output" / "final_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}
COLOR = {
    "audioseal": "#2878B5", "wavmark": "#54A24B", "timbrewm": "#B279A2",
    "voicemark": "#D95F59", "wmcodec": "#F2A541",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.7,
    "axes.labelsize": 8.1,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": .65,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def speaker_ratio_band(df: pd.DataFrame, value: str, seed: int, n_boot: int = 10000):
    grouped = df.groupby("speaker")[value].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(grouped), len(grouped))
        draws[i] = sums[idx].sum() / counts[idx].sum()
    return float(df[value].mean()), *np.quantile(draws, [.025, .975]).tolist()


def build_source_valid_k8() -> None:
    trials = pd.read_csv(SOURCE / "k8_population_trials.csv")
    bits = pd.read_csv(SOURCE / "k8_population_bits.csv")
    valid_trials = trials[trials.source_exact_count == 8].copy()
    valid_keys = valid_trials[["model", "trial_id"]]
    valid_bits = bits.merge(valid_keys, on=["model", "trial_id"], how="inner")
    valid_trials.to_csv(OUT / "k8_source_valid_trials.csv", index=False)
    valid_bits.to_csv(OUT / "k8_source_valid_bits.csv", index=False)

    curve_rows = []
    summary_rows = []
    for mi, model in enumerate(MODELS):
        bm = valid_bits[valid_bits.model == model]
        tm = valid_trials[valid_trials.model == model]
        for count in range(9):
            z = bm[bm.ones_among_8 == count]
            mean, low, high = speaker_ratio_band(z, "decoded_bit", 20263000 + mi * 20 + count)
            curve_rows.append({
                "model": model,
                "ones_among_8": count,
                "n_bit_observations": len(z),
                "decoded_one_rate": mean,
                "speaker_band_low": low,
                "speaker_band_high": high,
            })

        strict = bm[bm.strict == 1]
        unanimous = bm[bm.unanimous == 1]
        complete = strict.groupby(["speaker", "trial_id"], as_index=False).strict_correct.min()
        summary_rows.append({
            "model": model,
            "source_valid_trials": len(tm),
            "speakers_represented": tm.speaker.nunique(),
            "strict_majority_agreement": strict.strict_correct.mean(),
            "unanimous_preserved": unanimous.unanimous_correct.mean(),
            "complete_majority_match": complete.strict_correct.mean(),
            "tracing_failure": tm.tracing_failure.mean(),
        })

    curves = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    curves.to_csv(OUT / "k8_source_valid_composition_by_count.csv", index=False)
    summary.to_csv(OUT / "k8_source_valid_system_summary.csv", index=False)
    (OUT / "k8_source_valid_analysis.json").write_text(json.dumps({
        "selection_rule": "retain a trial only when all eight source copies decode as assigned",
        "model_rerun": False,
        "source": str(SOURCE),
        "trials_retained": {r["model"]: int(r["source_valid_trials"]) for r in summary_rows},
        "figure_band": "2.5th to 97.5th percentiles from speaker-level resampling",
        "bootstrap_replicates": 10000,
        "metrics": [
            "strict-majority agreement", "unanimous preservation",
            "complete-majority match", "tracing failure",
        ],
    }, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.68), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        z = curves[curves.model == model].sort_values("ones_among_8")
        x = z.ones_among_8.to_numpy(float)
        ax.axvspan(3.68, 4.32, color="#F1D3AE", alpha=.55, lw=0)
        ax.axhline(.5, color="#6E6E6E", ls="--", lw=.65)
        ax.fill_between(x, z.speaker_band_low.to_numpy(float),
                        z.speaker_band_high.to_numpy(float),
                        color=COLOR[model], alpha=.19, lw=0)
        ax.plot(x, z.decoded_one_rate.to_numpy(float), color=COLOR[model],
                marker="o", ms=2.7, lw=1.25)
        ax.set_title(LABEL[model], fontweight="bold")
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xlim(-.25, 8.25)
        ax.set_ylim(-.03, 1.03)
        ax.set_xlabel("Ones among 8")
        ax.grid(axis="y", color="#E5E5E5", lw=.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Decoded-one rate $q(c)$")
    fig.subplots_adjust(left=.08, right=.995, bottom=.26, top=.88, wspace=.16)
    save(fig, "fig_k8_population_composition")


def rounded_box(ax, xy, width, height, text, face, edge="#3A3A3A", size=7.4, bold=False):
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           facecolor=face, edgecolor=edge, linewidth=.8)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=size, fontweight="bold" if bold else "normal", linespacing=1.1)


def arrow(ax, start, end, color="#565656", width=.75):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=width, color=color))


def waveform(ax, x0, x1, y0, amplitude, color, phase=0.0, width=.75):
    x = np.linspace(x0, x1, 150)
    env = .5 + .5 * np.sin(np.linspace(0, 3 * np.pi, len(x))) ** 2
    y = y0 + amplitude * env * (np.sin(np.linspace(phase, phase + 18 * np.pi, len(x)))
                                + .25 * np.sin(np.linspace(0, 39 * np.pi, len(x))))
    ax.plot(x, y, color=color, lw=width, solid_capstyle="round")


def build_scenario() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.00))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([.505, .505], [.07, .93], color="#D7D7D7", lw=.7)

    # Left: a provider creates personalized copies; one copy traces directly.
    ax.text(.25, .94, "(a) Direct decoding", ha="center", va="top", fontsize=8.5, fontweight="bold")
    waveform(ax, .025, .145, .47, .055, "#555555")
    ax.text(.085, .31, "original audio", ha="center", fontsize=6.7, color="#555555")
    rounded_box(ax, (.17, .37), .105, .24, "Watermark\nembedder", "#E7F1F8", bold=True)
    arrow(ax, (.145, .47), (.17, .49))
    users = [("A", "#2E9D70", .72, 0.0), ("B", "#4D74B8", .50, .8), ("C", "#895DA5", .28, 1.5)]
    for name, color, yy, phase in users:
        arrow(ax, (.275, .49), (.31, yy))
        ax.add_patch(plt.Circle((.328, yy), .018, facecolor=color, edgecolor="white", lw=.7))
        ax.text(.328, yy, name, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
        waveform(ax, .35, .455, yy, .038, color, phase)
    ax.axvspan(.395, .417, ymin=.21, ymax=.79, color="#EED9B8", alpha=.55, lw=0)
    ax.text(.406, .84, "payload-sensitive\nsegment", ha="center", fontsize=6.5, color="#6A5231")
    rounded_box(ax, (.39, .09), .075, .12, "Decoder", "#E9EEF5", bold=True, size=6.9)
    arrow(ax, (.405, .70), (.425, .21))
    ax.add_patch(plt.Circle((.482, .15), .021, facecolor="#2E9D70", edgecolor="white", lw=.7))
    ax.text(.482, .15, "A", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.text(.482, .09, "traced to A", ha="center", fontsize=6.7, color="#2E6F52")
    arrow(ax, (.465, .15), (.462, .15))

    # Right: the same copies are averaged; common content survives but identity leaves the set.
    ax.text(.75, .94, "(b) Multi-copy averaging", ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.text(.615, .84, "same content, different payloads", ha="center", fontsize=6.9, color="#444444")
    for name, color, yy, phase in users:
        ax.add_patch(plt.Circle((.54, yy), .018, facecolor=color, edgecolor="white", lw=.7))
        ax.text(.54, yy, name, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
        waveform(ax, .565, .68, yy, .038, color, phase)
        arrow(ax, (.68, yy), (.718, .50))
    ax.add_patch(plt.Circle((.746, .50), .036, facecolor="#F0B34D", edgecolor="#A86D16", lw=.8))
    ax.text(.746, .50, "Mean", ha="center", va="center", fontsize=6.6, fontweight="bold")
    waveform(ax, .79, .88, .50, .045, "#D08A25", .4, 1.0)
    arrow(ax, (.782, .50), (.79, .50))
    rounded_box(ax, (.90, .39), .075, .22, "Decoder", "#E9EEF5", bold=True, size=7.0)
    arrow(ax, (.88, .50), (.90, .50))
    ax.add_patch(plt.Circle((.987, .50), .019, facecolor="#C74C46", edgecolor="white", lw=.7))
    ax.text(.987, .50, "D", ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    ax.text(.93, .24, "decoded payload\nnot in {A, B, C}", ha="center", va="center",
            fontsize=7.0, color="#9C2F2A", fontweight="bold")
    ax.text(.75, .12, "shared speech remains; participant identity is lost", ha="center",
            fontsize=7.0, color="#4B4B4B")

    fig.subplots_adjust(left=.01, right=.995, bottom=.03, top=.98)
    save(fig, "fig_scenario")


def main() -> None:
    build_source_valid_k8()
    build_scenario()
    print(OUT)


if __name__ == "__main__":
    main()

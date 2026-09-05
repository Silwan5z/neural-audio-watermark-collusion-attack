#!/usr/bin/env python3
"""Build the single-column K=8 minimum-bit-confidence figure."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data" / "confidence"
SUMMARY = ROOT / "data" / "summary"
FIGURES = ROOT / "paper" / "figures"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABELS = {
    "audioseal": "AudioSeal",
    "wavmark": "WavMark",
    "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark",
    "wmcodec": "WMCodec",
}
CONDITIONS = ["benign_copy", "uniform_collusion", "successful_mrc"]
STYLE = {
    "benign_copy": ("Valid", "#176C9C", "o", 0.20),
    "uniform_collusion": ("Uniform", "#7C8790", "s", 0.00),
    "successful_mrc": ("MRC", "#D9822B", "D", -0.20),
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.8,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.72,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def main() -> None:
    frames = []
    for model in MODELS:
        path = ANALYSIS / f"min_bit_confidence_k8_{model}.csv"
        data = pd.read_csv(path)
        if set(data["model"]) != {model}:
            raise RuntimeError(f"unexpected model values in {path}")
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)

    summary_rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            values = data.loc[
                data.model.eq(model) & data.condition.eq(condition),
                "min_bit_confidence",
            ].to_numpy(float)
            if len(values) == 0:
                continue
            q05, q25, q50, q75, q95 = np.quantile(values, [.05, .25, .50, .75, .95])
            summary_rows.append({
                "model": model,
                "condition": condition,
                "n": len(values),
                "q05": f"{q05:.6f}",
                "q25": f"{q25:.6f}",
                "median": f"{q50:.6f}",
                "q75": f"{q75:.6f}",
                "q95": f"{q95:.6f}",
            })

    summary_path = SUMMARY / "confidence_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=summary_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(3.50, 2.95))
    centers = np.arange(len(MODELS))[::-1]

    for index, center in enumerate(centers):
        if index % 2 == 0:
            ax.axhspan(center - 0.42, center + 0.42, color="#F5F6F7", lw=0, zorder=0)
        ax.text(0.392, center + 0.34, LABELS[MODELS[index]],
                ha="left", va="center", fontsize=8.4,
                fontweight="semibold", color="#20262B", zorder=5)

    for condition in CONDITIONS:
        label, color, marker, offset = STYLE[condition]
        first = True
        for model, center in zip(MODELS, centers):
            row = summary[(summary.model == model) & (summary.condition == condition)].iloc[0]
            y = center + offset
            q05, q25, q50, q75, q95 = [
                float(row[key]) for key in ("q05", "q25", "median", "q75", "q95")
            ]
            ax.hlines(y, q05, q95, color=color, lw=0.9, alpha=0.75, zorder=2)
            ax.hlines(y, q25, q75, color=color, lw=3.5, zorder=3)
            ax.plot(q50, y, marker=marker, ms=4.5, color=color, markeredgecolor="white",
                    markeredgewidth=0.55, linestyle="none", zorder=4,
                    label=label if first else None)
            if condition == "successful_mrc" and model in {"voicemark", "wmcodec"}:
                ax.text(0.985, y + 0.105, f"{int(row['n'])} trials",
                        ha="right", va="center", fontsize=7.2, color=color)
            first = False

    ax.set_yticks([])
    ax.set_xlim(0.38, 1.01)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.48, len(MODELS) - 0.52)
    ax.set_xlabel("Minimum bit confidence")
    ax.grid(axis="x", color="#DDE1E4", lw=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=2.2, pad=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=3,
              frameon=False, handlelength=1.1, columnspacing=0.9)
    fig.subplots_adjust(left=0.045, right=0.998, bottom=0.16, top=0.885)

    for extension in ("pdf", "svg"):
        fig.savefig(FIGURES / f"fig4.{extension}",
                    dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the K=8 composition figure from final source-correct JSON records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data" / "summary"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABELS = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.8,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "axes.linewidth": 0.72,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def summarize_model(raw_dir: Path, model: str, bootstrap: int = 5000) -> list[dict]:
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((raw_dir / model).glob("trial_*.json"))]
    if len(records) != 300:
        raise RuntimeError(f"expected 300 final records for {model}, got {len(records)}")
    speakers = sorted({record["speaker"] for record in records})
    if len(speakers) != 100:
        raise RuntimeError(f"expected 100 speakers for {model}, got {len(speakers)}")
    index = {speaker: position for position, speaker in enumerate(speakers)}
    ones = np.zeros((100, 9), dtype=np.float64)
    totals = np.zeros((100, 9), dtype=np.float64)
    for record in records:
        speaker = index[record["speaker"]]
        for support, decoded in zip(record["ones_among_8"], record["decoded_hard_bits"]):
            support = int(support)
            ones[speaker, support] += int(decoded)
            totals[speaker, support] += 1

    rng = np.random.default_rng(20260905 + MODELS.index(model))
    weights = rng.multinomial(100, np.full(100, 0.01), size=bootstrap)
    boot_ones = weights @ ones
    boot_totals = weights @ totals
    rows = []
    for support in range(9):
        valid = boot_totals[:, support] > 0
        rates = boot_ones[valid, support] / boot_totals[valid, support]
        rows.append({
            "model": model,
            "ones_among_8": support,
            "n": int(totals[:, support].sum()),
            "decoded_one_rate": float(ones[:, support].sum() / totals[:, support].sum()),
            "speaker_bootstrap_low": float(np.quantile(rates, 0.025)),
            "speaker_bootstrap_high": float(np.quantile(rates, 0.975)),
            "n_speakers": int(np.count_nonzero(totals[:, support])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "k8" / "raw")
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = [row for model in MODELS for row in summarize_model(args.raw_dir, model)]
    summary_path = ANALYSIS / "k8_composition.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(5, 1, figsize=(3.50, 2.75), sharex=True, sharey=True)
    blue, gray, grid = "#176C9C", "#9AA2A8", "#E1E4E6"
    for ax, model in zip(axes, MODELS):
        model_rows = [row for row in rows if row["model"] == model]
        x = np.asarray([row["ones_among_8"] for row in model_rows], dtype=float)
        y = np.asarray([row["decoded_one_rate"] for row in model_rows], dtype=float)
        low = np.asarray([row["speaker_bootstrap_low"] for row in model_rows], dtype=float)
        high = np.asarray([row["speaker_bootstrap_high"] for row in model_rows], dtype=float)
        ax.axvspan(3.72, 4.28, color="#F1F2F3", lw=0, zorder=0)
        ax.plot([0, 8], [0, 1], color=gray, lw=0.55, ls=(0, (2.5, 2.2)), zorder=1)
        ax.errorbar(x, y, yerr=[y - low, high - y], fmt="o", ls="none",
                    ms=3.1, mfc="white", mec=blue, mew=0.85, color=blue,
                    elinewidth=0.7, capsize=1.2, capthick=0.65, zorder=3)
        ax.text(0.015, 0.78, LABELS[model], transform=ax.transAxes,
                ha="left", va="center", fontsize=8.6, fontweight="semibold")
        ax.set_xlim(-0.25, 8.25)
        ax.set_ylim(-0.06, 1.06)
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.grid(axis="y", color=grid, lw=0.42)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=1.8, pad=1.4)
    # A single shared y scale is enough; repeating the same three labels in all
    # five rows adds clutter without information.
    for index, ax in enumerate(axes):
        ax.tick_params(axis="y", left=(index == 2), labelleft=(index == 2))
    axes[-1].set_xticks(range(9))
    axes[-1].set_xlabel("Members with bit 1 (out of 8)")
    # The caption defines the vertical quantity.  Keeping only the shared tick
    # labels here removes the external side title and gives the data panels the
    # maximum useful single-column width.
    fig.subplots_adjust(left=0.045, right=0.998, bottom=0.15, top=0.995, hspace=0.11)

    for extension in ("pdf", "svg"):
        fig.savefig(FIGURES / f"fig2.{extension}", dpi=600,
                    facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

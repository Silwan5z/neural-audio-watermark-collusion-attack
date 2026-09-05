#!/usr/bin/env python3
"""Build the final one-bit mixture-path figure from frozen analysis data."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "summary" / "path_representatives.csv"
OUT = ROOT / "outputs" / "figures"

INK = "#20262B"
AXIS = "#596168"
GRID = "#E1E5E8"
BLUE = "#176C9C"
ORANGE = "#D9822B"
PURPLE = "#6F5C99"
STATE_COLORS = {0: "#A9CDE0", 1: "#F0BE86", 2: "#C3B8DD"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.8,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.4,
    "axes.linewidth": 0.72,
    "xtick.major.width": 0.70,
    "ytick.major.width": 0.70,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK, pad=1.5)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.42)


def path_panel(ax: plt.Axes, data: pd.DataFrame, model: str, title: str,
               letter: str) -> None:
    z = data[data.model == model].sort_values("lambda").copy()
    if z.empty:
        raise RuntimeError(f"no representative path for {model}")
    lam = z["lambda"].to_numpy(float)
    response = z.response.to_numpy(float)
    decoded = z.decoded_identity.to_numpy(int)
    payload_a = int(z.iloc[0].base_payload)
    payload_b = int(z.iloc[0].flipped_payload)
    state = np.where(decoded == payload_a, 0, np.where(decoded == payload_b, 2, 1))

    ax.axhline(0.5, color="#9EA6AC", lw=0.58, ls=(0, (3, 2)), zorder=1)
    ax.plot(lam, response, color=BLUE, marker="o", ms=3.4, lw=1.55,
            mfc="white", mec=BLUE, mew=0.9, zorder=3)

    # The thin strip reports the decoded complete payload; the curve reports
    # only the one bit on which the valid endpoints differ.
    for lo, hi, value in zip(lam[:-1], lam[1:], state[:-1]):
        ax.axvspan(lo, hi, ymin=0.015, ymax=0.095,
                   color=STATE_COLORS[int(value)], lw=0, zorder=0)
    run_start = 0
    names = {0: "A", 1: "Other", 2: "B"}
    for index in range(1, len(state) + 1):
        if index == len(state) or state[index] != state[run_start]:
            left = lam[run_start]
            right = lam[index] if index < len(lam) else 1.0
            if right - left >= 0.13:
                ax.text((left + right) / 2, 0.047, names[int(state[run_start])],
                        ha="center", va="center", fontsize=8.0, color=INK)
            run_start = index

    ax.text(0.0, 1.025, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=INK)
    ax.text(0.105, 1.025, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.0, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    clean_axis(ax)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    fig = plt.figure(figsize=(3.50, 2.85), facecolor="white")
    grid = fig.add_gridspec(2, 1, hspace=0.72)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])

    path_panel(ax_a, data, "audioseal", "AudioSeal", "a")
    path_panel(ax_b, data, "voicemark", "VoiceMark", "b")
    ax_a.tick_params(labelbottom=False)
    ax_b.set_xlabel(r"Weight on copy B, $\lambda$")
    fig.text(0.004, 0.53, "B-bit support",
             ha="center", va="center", rotation=90, fontsize=9.0, color=INK)
    fig.subplots_adjust(left=0.075, right=0.998, bottom=0.14, top=0.95)
    for extension in ("pdf", "svg"):
        fig.savefig(OUT / f"fig3.{extension}", dpi=600,
                    facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main()

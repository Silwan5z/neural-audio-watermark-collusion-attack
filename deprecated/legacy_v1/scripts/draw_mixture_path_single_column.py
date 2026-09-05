#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "main_text_revision_20260830" / "mixture_path"
OUT = ROOT / "paper_rewriting_output" / "final_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {"audioseal": "#3B6EA8", "voicemark": "#D9544D"}
LAB = {"audioseal": "AudioSeal", "voicemark": "VoiceMark"}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.6,
    "axes.labelsize": 8.1,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "pdf.fonttype": 42,
})

traj = pd.read_csv(DATA / "mixture_path_aggregate_trajectory.csv")
cnt = pd.read_csv(DATA / "mixture_path_transition_counts.csv").set_index("model")
fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.38), gridspec_kw={"height_ratios": [1.12, 1]})

ax = axes[0]
for m in ("audioseal", "voicemark"):
    z = traj[traj.model == m].sort_values("lambda_value")
    x = z.lambda_value.to_numpy(float)
    ax.fill_between(x, z.p10.to_numpy(float), z.p90.to_numpy(float), color=COL[m], alpha=.18, lw=0)
    ax.plot(x, z["mean"].to_numpy(float), color=COL[m], lw=1.45, label=LAB[m])
ax.axhline(.5, color="#777777", ls="--", lw=.7)
ax.set(xlim=(-.02, 1.02), ylim=(0, 1.03), xlabel=r"Interpolation $\lambda$", ylabel="Response toward flipped endpoint")
ax.set_title("(a) Target-bit trajectory (P10--P90)", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, loc="lower center", fontsize=7)
ax.grid(axis="y", color="#E4E4E4", lw=.45)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
x = np.arange(2)
single = [cnt.loc[m, "identity_single_transition"] for m in ("audioseal", "voicemark")]
multiple = [cnt.loc[m, "identity_multiple_transition"] for m in ("audioseal", "voicemark")]
ax.bar(x, single, color="#4C78A8", width=.55, label="Single")
ax.bar(x, multiple, bottom=single, color="#E45756", width=.55, label="Multiple")
for i, (s, u) in enumerate(zip(single, multiple)):
    if s > 15: ax.text(i, s/2, str(int(s)), ha="center", va="center", color="white", fontweight="bold")
    if u > 15: ax.text(i, s+u/2, str(int(u)), ha="center", va="center", color="white", fontweight="bold")
ax.set(xticks=x, xticklabels=["AudioSeal", "VoiceMark"], ylim=(0, 320), ylabel="Trials (of 300)")
ax.set_title("(b) Decoded-identity transitions", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=7)
ax.grid(axis="y", color="#E4E4E4", lw=.45)
ax.spines[["top", "right"]].set_visible(False)

fig.subplots_adjust(left=.18, right=.98, top=.96, bottom=.11, hspace=.55)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_controlled_onebit_interpolation_single.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
plt.close(fig)

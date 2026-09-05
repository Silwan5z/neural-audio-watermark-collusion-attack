#!/usr/bin/env python3
"""Draw population and representative controlled one-bit interpolation paths."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "main_text_revision_20260830" / "mixture_path"
RAW = ROOT / "data" / "mixture_path_k5_adaptive_20260829"
OUT = ROOT / "paper_v4_revision_20260831" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {"audioseal": "#3B6EA8", "voicemark": "#D9544D"}
LAB = {"audioseal": "AudioSeal", "voicemark": "VoiceMark"}
MODELS = ("audioseal", "voicemark")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.9, "axes.titlesize": 8.6, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def oriented_path(row: pd.Series) -> float:
    probs = json.loads(row.bit_probabilities)
    bit = int(row.flipped_bit)
    flipped_value = (int(row.flipped_payload) >> bit) & 1
    p = float(probs[bit])
    return p if flipped_value else 1.0 - p


traj = pd.read_csv(DATA / "mixture_path_aggregate_trajectory.csv")
cnt = pd.read_csv(DATA / "mixture_path_transition_counts.csv").set_index("model")
summary = pd.read_csv(RAW / "mixture_path_trial_summary.csv")
points = {model: pd.read_csv(RAW / f"mixture_path_{model}_points.csv") for model in MODELS}

# Among endpoint-correct paths with the requested discrete behavior, select the
# path closest to its system's aggregate target trajectory on the coarse grid.
selected = {}
for model in MODELS:
    sm = summary[summary.model == model].copy()
    if model == "audioseal":
        candidates = sm[(sm.target_bit_monotonic == 1) & (sm.identity_transition_count == 1)]
    else:
        candidates = sm[sm.identity_transition_count >= 2]
    aggregate = traj[traj.model == model].set_index("lambda_value")["mean"]
    scores = []
    for trial_id in candidates.trial_id.astype(int):
        z = points[model][(points[model].trial_id == trial_id) &
                          (points[model].sampling_stage == "coarse")].copy()
        if len(z) != 11:
            continue
        z = z.sort_values("lambda")
        if int(z.iloc[0].decoded_identity) != int(z.iloc[0].base_payload):
            continue
        if int(z.iloc[-1].decoded_identity) != int(z.iloc[-1].flipped_payload):
            continue
        response = z.apply(oriented_path, axis=1).to_numpy(float)
        target = aggregate.loc[z["lambda"].to_numpy(float)].to_numpy(float)
        scores.append((float(np.mean((response - target) ** 2)), trial_id))
    selected[model] = min(scores)[1]

fig = plt.figure(figsize=(7.16, 3.72))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, .92], hspace=.55, wspace=.75)
axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]

ax = axes[0]
for model in MODELS:
    z = traj[traj.model == model].sort_values("lambda_value")
    x = z.lambda_value.to_numpy(float)
    ax.fill_between(x, z.p10.to_numpy(float), z.p90.to_numpy(float), color=COL[model], alpha=.18, lw=0)
    ax.plot(x, z["mean"].to_numpy(float), color=COL[model], lw=1.45,
            marker="o", ms=2.2, label=LAB[model])
ax.axhline(.5, color="#777777", ls="--", lw=.7)
ax.set(xlim=(-.02, 1.02), ylim=(0, 1.03), xlabel=r"Interpolation weight $\lambda$",
       ylabel="Response toward bit-1 endpoint")
ax.set_title("(a) Target-bit response", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=6.8)

ax = axes[1]
x = np.arange(2); width = .34
mono = [cnt.loc[m, "target_monotonic"] for m in MODELS]
back = [cnt.loc[m, "target_back_and_forth"] for m in MODELS]
bars = [ax.bar(x - width / 2, mono, width=width, color="#4C78A8", label="Monotonic"),
        ax.bar(x + width / 2, back, width=width, color="#E45756", label="Back-and-forth")]
for group in bars:
    for bar in group:
        value = int(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, max(value, 2) + 5, str(value),
                ha="center", va="bottom", fontweight="bold", fontsize=7)
ax.set(xticks=x, xticklabels=[LAB[m] for m in MODELS], ylim=(0, 315), ylabel="Paths (of 300)")
ax.set_title("(b) Target-bit paths", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=6.4, loc="upper right")

ax = axes[2]
single = [cnt.loc[m, "identity_single_transition"] for m in MODELS]
multiple = [cnt.loc[m, "identity_multiple_transition"] for m in MODELS]
ax.bar(x, single, color="#4C78A8", width=.56, label="Single")
ax.bar(x, multiple, bottom=single, color="#E45756", width=.56, label="Multiple")
for i, (one, many) in enumerate(zip(single, multiple)):
    if one >= 18: ax.text(i, one / 2, str(int(one)), ha="center", va="center", color="white", fontweight="bold")
    if many >= 18: ax.text(i, one + many / 2, str(int(many)), ha="center", va="center", color="white", fontweight="bold")
ax.set(xticks=x, xticklabels=[LAB[m] for m in MODELS], ylim=(0, 315), ylabel="Paths (of 300)")
ax.set_title("(c) Identity transitions", loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="upper center")

for col, model in ((slice(0, 3), "audioseal"), (slice(3, 6), "voicemark")):
    ax = fig.add_subplot(gs[1, col])
    z = points[model][points[model].trial_id == selected[model]].copy().sort_values("lambda")
    response = z.apply(oriented_path, axis=1).to_numpy(float)
    lam = z["lambda"].to_numpy(float)
    base = int(z.iloc[0].base_payload); flipped = int(z.iloc[0].flipped_payload)
    decoded = z.decoded_identity.to_numpy(int)
    intermediate_ids = list(dict.fromkeys(v for v in decoded if v not in (base, flipped)))
    intermediate_colors = ["#F2B134", "#D9912B", "#B97835", "#8E6C8A"]
    color_for = {base: "#4C78A8", flipped: "#E45756"}
    color_for.update({value: intermediate_colors[i % len(intermediate_colors)]
                      for i, value in enumerate(intermediate_ids)})
    ax.plot(lam, response, color=COL[model], lw=1.5, marker="o", ms=2.5)
    ax.axhline(.5, color="#777777", ls="--", lw=.7)
    for lo, hi, value in zip(lam[:-1], lam[1:], decoded[:-1]):
        ax.axvspan(lo, hi, ymin=.00, ymax=.08, color=color_for[int(value)], lw=0)
    ax.set(xlim=(-.01, 1.01), ylim=(-.02, 1.03), xlabel=r"Interpolation weight $\lambda$",
           ylabel="Target-bit response")
    transitions = int(summary[(summary.model == model) &
                              (summary.trial_id == selected[model])].identity_transition_count.iloc[0])
    letter = "d" if model == "audioseal" else "e"
    plural = "s" if transitions != 1 else ""
    ax.set_title(f"({letter}) {LAB[model]} example: {transitions} identity transition{plural}",
                 loc="left", fontweight="bold")
    ax.text(.01, .11, "identity path", transform=ax.transAxes, fontsize=6.8, color="#444444")

for ax in fig.axes:
    ax.grid(axis="y", color="#E4E4E4", lw=.42, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

fig.legend(handles=[Patch(color="#4C78A8", label="bit-0 endpoint"),
                    Patch(color="#F2B134", label="intermediate identity"),
                    Patch(color="#E45756", label="bit-1 endpoint")],
           frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.005), fontsize=6.9)
fig.subplots_adjust(left=.075, right=.995, top=.96, bottom=.13)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_controlled_onebit_interpolation_wide.{ext}", dpi=350,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
(OUT / "mixture_path_representative_selection.json").write_text(
    json.dumps({"selection_rule": "closest coarse target trajectory to the system aggregate among endpoint-correct paths with the requested identity-transition behavior",
                "selected_trial": selected}, indent=2) + "\n")

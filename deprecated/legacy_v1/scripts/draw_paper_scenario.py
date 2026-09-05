#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_rewriting_output" / "final_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.45, 1.62))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def box(x, y, w, h, text, face, edge="#333333", bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       facecolor=face, edgecolor=edge, linewidth=0.9)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=8, fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.9, color="#555555"))

box(.02, .60, .20, .25, "Shared\ncontent $x$", "#EEEEEE", bold=True)
box(.30, .69, .20, .20, "$y_1=x+w_1$", "#F4D7A5")
box(.30, .45, .20, .20, "$y_2=x+w_2$", "#CDE8D7")
box(.30, .21, .20, .20, "$y_K=x+w_K$", "#D8E3F4")
arrow(.22, .725, .30, .79); arrow(.22, .725, .30, .55); arrow(.22, .725, .30, .31)
box(.59, .55, .18, .27, "Waveform\naverage", "#F3E5F5", bold=True)
arrow(.50, .79, .59, .72); arrow(.50, .55, .59, .68); arrow(.50, .31, .59, .62)
box(.82, .55, .16, .27, "Decode\npayload", "#E5ECF6", bold=True)
arrow(.77, .685, .82, .685)
ax.text(.69, .38, r"$\bar y=x+\frac{1}{K}\sum_i w_i$", ha="center", va="center", fontsize=8.5)
ax.text(.69, .16, r"$\hat m\notin\{m_1,\ldots,m_K\}$", ha="center", va="center", fontsize=8.5, color="#8B1E18")
ax.text(.69, .04, "no coalition payload recovered", ha="center", va="center", fontsize=7.4, color="#8B1E18")
fig.subplots_adjust(left=.005, right=.995, top=.97, bottom=.03)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_scenario.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
plt.close(fig)

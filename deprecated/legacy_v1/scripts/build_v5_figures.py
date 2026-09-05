#!/usr/bin/env python3
"""Recompute V5 summaries from existing records and draw publication figures."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PAPER = ROOT / "paper_v5_revision_20260831"
FIG = PAPER / "figures"
ANA = PAPER / "analysis"
FIG.mkdir(parents=True, exist_ok=True)
ANA.mkdir(parents=True, exist_ok=True)

MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}
BITS = {"audioseal": 16, "wavmark": 16, "timbrewm": 10, "voicemark": 16, "wmcodec": 16}
# Okabe-Ito-derived, colorblind-friendly palette. Markers remain distinct.
COLOR = {
    "audioseal": "#0072B2", "wavmark": "#009E73", "timbrewm": "#CC79A7",
    "voicemark": "#D55E00", "wmcodec": "#E69F00",
}
MARKER = {"audioseal": "o", "wavmark": "s", "timbrewm": "D", "voicemark": "^", "wmcodec": "v"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.2, "axes.titlesize": 8.7, "axes.labelsize": 8.2,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.6,
    "axes.linewidth": .65, "lines.linewidth": 1.2, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=400, bbox_inches="tight", pad_inches=.025,
                    facecolor="white")
    plt.close(fig)


def read_json_retry(path: Path, attempts: int = 8) -> dict:
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            if attempt + 1 == attempts:
                raise
            time.sleep(.2)
    raise RuntimeError(path)


def cluster_mean_ci(df: pd.DataFrame, value: str, speaker: str = "speaker",
                    seed: int = 20260831, reps: int = 10000) -> tuple[float, float, float]:
    groups = [g[value].to_numpy(float) for _, g in df.groupby(speaker)]
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    n = len(groups)
    for i in range(reps):
        ix = rng.integers(0, n, n)
        vals[i] = np.mean(np.concatenate([groups[j] for j in ix]))
    return float(df[value].mean()), *np.quantile(vals, [.025, .975]).tolist()


def cluster_ratio_ci(df: pd.DataFrame, numerator: str, denominator: str,
                     seed: int, reps: int = 10000) -> tuple[float, float, float]:
    g = df.groupby("speaker")[[numerator, denominator]].sum()
    a, b = g[numerator].to_numpy(float), g[denominator].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    n = len(g)
    for i in range(reps):
        ix = rng.integers(0, n, n)
        vals[i] = a[ix].sum() / b[ix].sum()
    return float(df[numerator].sum() / df[denominator].sum()), *np.quantile(vals, [.025, .975]).tolist()


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E1E1E1", lw=.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def draw_scenario() -> None:
    """Compact vector redraw of the original first-version scenario."""
    fig, ax = plt.subplots(figsize=(7.16, 1.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, text, edge="#277DA1", face="white", fontsize=8.2):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.012",
                           linewidth=1.05, edgecolor=edge, facecolor=face)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                                fontsize=fontsize, color="#222")
        return p

    def wave(x0, x1, y, color="#69757A", amp=.035, phase=0, lw=1.05):
        x = np.linspace(x0, x1, 240)
        z = (np.sin(2*np.pi*(6.2*(x-x0)/(x1-x0)+phase))
             + .45*np.sin(2*np.pi*(13.1*(x-x0)/(x1-x0)+.2+phase)))
        ax.plot(x, y + amp*z/1.45, color=color, lw=lw, solid_capstyle="round")

    def arrow(x0, y0, x1, y1, color="#6D7A80", lw=.9):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))

    ax.text(.02, .94, "Same audio content, different user payloads", fontsize=8.7,
            fontweight="bold", color="#222", va="top")
    wave(.025, .145, .51, amp=.045)
    ax.text(.085, .36, "original audio", ha="center", fontsize=7.8, color="#4D5558")
    box(.16, .39, .105, .23, "Watermark\nembedder")
    arrow(.145, .51, .16, .51)

    ys = [.77, .51, .25]
    users = ["A", "B", "C"]
    payloads = ["payload 1", "payload 0", "payload 1"]
    ucols = ["#0072B2", "#009E73", "#7E57C2"]
    for y, user, payload, col, phase in zip(ys, users, payloads, ucols, [0, .25, .5]):
        # Route into and out of the user node; no arrow crosses the circle.
        arrow(.265, .51, .306, y)
        circ = Circle((.33, y), .025, edgecolor="white", facecolor=col, lw=1.0, zorder=3)
        ax.add_patch(circ); ax.text(.33, y, user, ha="center", va="center", color="white",
                                   fontsize=8.2, fontweight="bold", zorder=4)
        arrow(.355, y, .38, y)
        wave(.38, .61, y, phase=phase)
        # Payload-sensitive segment uses both color and a thicker line.
        wave(.47, .525, y, color=col, amp=.04, phase=phase+.1, lw=1.9)
        ax.text(.495, y-.095 if y > .3 else y+.075, payload, ha="center", fontsize=7.8, color=col)

    ax.text(.64, .94, "(a) Direct decoding", fontsize=8.4, fontweight="bold", va="top")
    arrow(.61, .77, .69, .77)
    box(.69, .68, .105, .18, "Decoder")
    arrow(.795, .77, .825, .77)
    out = Circle((.85, .77), .025, edgecolor="#009E73", facecolor="#E9F7F1", lw=1.0)
    ax.add_patch(out); ax.text(.85, .77, "A", ha="center", va="center", fontsize=8.0,
                              color="#007A5E", fontweight="bold")
    ax.text(.885, .77, "traced to A", va="center", fontsize=7.8, color="#007A5E")

    ax.plot([.63, .995], [.58, .58], color="#D4DADD", lw=.7)
    ax.text(.64, .52, "(b) Multi-copy averaging", fontsize=8.4, fontweight="bold", va="top")
    # The three payload branches merge without passing through the user nodes.
    for y in ys:
        ax.plot([.61, .655], [y, .34], color="#8B969B", lw=.75)
    avg = Circle((.68, .34), .038, edgecolor="#E69F00", facecolor="#FFF4D8", lw=1.1)
    ax.add_patch(avg); ax.text(.68, .34, "mean", ha="center", va="center", fontsize=7.5,
                              color="#8A5B00", fontweight="bold")
    arrow(.718, .34, .735, .34)
    wave(.735, .82, .34, amp=.035)
    arrow(.82, .34, .835, .34)
    box(.835, .25, .075, .18, "Decoder")
    arrow(.91, .34, .925, .34)
    box(.925, .245, .07, .19, "Unmatched\npayload", edge="#D55E00", face="#FFF0E8", fontsize=7.7)
    fig.subplots_adjust(left=.008, right=.995, bottom=.04, top=.98)
    save(fig, "fig1_scenario_v5")


def draw_scenario_original() -> None:
    """Retain the first-draft Fig. 1 artwork and repair its final labels."""
    import fitz

    source = fitz.open(ROOT / "paper_rewriting_output" / "fig1_original_first_version.pdf")
    out = fitz.open()
    page = out.new_page(width=source[0].rect.width, height=source[0].rect.height)
    page.show_pdf_page(page.rect, source, 0)
    # Replace only the three labels requested by the revision; the original
    # information structure, waveforms, branches, and average icon are retained.
    replacements = [
        (fitz.Rect(282, 104, 382, 118), "(b) Multi-copy averaging", 8.1),
    ]
    for rect, _, _ in replacements:
        page.add_redact_annot(rect, fill=(1, 1, 1))
    page.add_redact_annot(fitz.Rect(477, 139, 514, 196), fill=(1, 1, 1))
    page.apply_redactions()
    for rect, label, size in replacements:
        page.insert_text((rect.x0+2, rect.y0+9.2), label, fontsize=size,
                         fontname="hebo", color=(.12, .16, .18))
    page.draw_line(fitz.Point(477, 158.5), fitz.Point(483, 158.5),
                   color=(.34, .39, .41), width=1.0)
    # Restore the decoder's right edge, which touches the replaced output label.
    page.draw_line(fitz.Point(477.4, 145.7), fitz.Point(477.4, 174.4),
                   color=(.08, .36, .50), width=1.5)
    box = fitz.Rect(483, 145, 513, 175)
    page.draw_rect(box, color=(.75, .22, .10), fill=(1.0, .94, .91), width=1.3)
    page.insert_text((484.2, 157.0), "Unmatched", fontsize=5.0,
                     fontname="hebo", color=(.60, .16, .07))
    page.insert_text((488.2, 165.1), "payload", fontsize=5.0,
                     fontname="hebo", color=(.60, .16, .07))
    out.save(FIG / "fig1_scenario_v5.pdf", garbage=4, deflate=True)
    p = out[0]
    p.get_pixmap(matrix=fitz.Matrix(3.2, 3.2), alpha=False).save(FIG / "fig1_scenario_v5.png")
    (FIG / "fig1_scenario_v5.svg").write_text(p.get_svg_image(), encoding="utf-8")
    out.close(); source.close()


def reconstruct_coalitions(model: str, k: int) -> list[dict]:
    from registry import coalition_seed, int_to_bits, sample_coalition, speaker_trial_index
    rows = []
    d = BITS[model]
    for trial_id, (speaker, local_t) in enumerate(speaker_trial_index(n_total=300)):
        rng = np.random.default_rng(coalition_seed(speaker, k, local_t))
        payloads = sample_coalition(rng, model, k)
        bits = np.asarray([int_to_bits(v, d) for v in payloads], dtype=int)
        p = bits.mean(axis=0)
        # Independent bit recombination: one participant bit is selected uniformly
        # at each coordinate. Participant strings are distinct, so probabilities add.
        match_prob = 0.0
        for row in bits:
            prob = np.prod(np.where(row == 1, p, 1-p))
            match_prob += float(prob)
        rows.append({"model": model, "K": k, "speaker": speaker, "trial_id": trial_id,
                     "bit_recombination_tf": 1.0-match_prob})
    return rows


def draw_risk() -> None:
    stats = []
    raw = []
    for mi, model in enumerate(MODELS):
        for k in (2, 3, 5, 8):
            df = pd.read_csv(ROOT / "results" / "evaluation" / f"attack_{model}_K{k}.csv")
            df = df[df.method == "mean"].copy()
            df = df.rename(columns={"spk": "speaker"})
            df["nearest_distance"] = 1.0 - df.ACC_near_norm.astype(float)
            raw.append(df.assign(model=model, K=k))
            for metric in ("ASR", "nearest_distance", "PESQ", "STOI"):
                mean, lo, hi = cluster_mean_ci(df, metric, seed=20260831+mi*40+k)
                stats.append({"model": model, "K": k, "metric": metric,
                              "mean": mean, "low": lo, "high": hi,
                              "n_trials": len(df), "n_speakers": df.speaker.nunique()})
    sdf = pd.DataFrame(stats)
    rdf = pd.concat(raw, ignore_index=True)
    sdf.to_csv(ANA / "risk_and_quality_summary.csv", index=False)
    rdf[["model", "K", "speaker", "local_t", "gi", "ASR", "nearest_distance", "PESQ", "STOI"]].to_csv(
        ANA / "risk_trial_values.csv", index=False)

    baseline_rows = []
    for model in ("audioseal", "timbrewm"):  # representative 16- and 10-bit payload spaces
        for k in (2, 3, 5, 8):
            z = pd.DataFrame(reconstruct_coalitions(model, k))
            mean, lo, hi = cluster_mean_ci(z, "bit_recombination_tf", seed=20261200+BITS[model]*10+k)
            baseline_rows.append({"payload_bits": BITS[model], "K": k, "mean": mean,
                                  "low": lo, "high": hi})
    bdf = pd.DataFrame(baseline_rows)
    bdf.to_csv(ANA / "bit_recombination_reference.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(3.50, 4.55), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": .14})
    xbase = np.arange(4)
    offsets = np.linspace(-.20, .20, len(MODELS))
    for ax, metric, ylabel in [(axes[0], "ASR", "Tracing failure"),
                               (axes[1], "nearest_distance", "Nearest-participant\ndistance")]:
        for off, model in zip(offsets, MODELS):
            z = sdf[(sdf.model == model) & (sdf.metric == metric)].sort_values("K")
            y = z["mean"].to_numpy(); lo = z.low.to_numpy(); hi = z.high.to_numpy()
            ax.errorbar(xbase+off, y, yerr=[y-lo, hi-y], fmt=MARKER[model], ms=4.6,
                        color=COLOR[model], mfc="white", mew=1.0, capsize=2.0, lw=.9,
                        label=LABEL[model], zorder=3)
        ax.set_ylim(-.02, 1.06); ax.set_ylabel(ylabel); style_axis(ax)
    for d, marker, off in [(16, "x", -.29), (10, "+", .29)]:
        z = bdf[bdf.payload_bits == d].sort_values("K")
        y = z["mean"].to_numpy(); lo = z.low.to_numpy(); hi = z.high.to_numpy()
        axes[0].errorbar(xbase+off, y, yerr=[y-lo, hi-y], fmt=marker, ms=5.2,
                         color="#666666", capsize=2, lw=.8, mew=1.1,
                         label=f"Bit recombination ({d}-bit)", zorder=2)
    axes[1].set_xticks(xbase, ["2", "3", "5", "8"])
    axes[1].set_xlabel("Coalition size $K$")
    axes[0].tick_params(axis="x", bottom=False)
    handles = [Line2D([0], [0], marker=MARKER[m], color=COLOR[m], mfc="white", lw=0,
                      label=LABEL[m], markersize=4.5) for m in MODELS]
    handles += [Line2D([0], [0], marker="x", color="#666", lw=0, label="Recomb. 16-bit", markersize=5),
                Line2D([0], [0], marker="+", color="#666", lw=0, label="Recomb. 10-bit", markersize=5)]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(.51, .005), handletextpad=.35, columnspacing=.8)
    fig.subplots_adjust(left=.19, right=.98, top=.985, bottom=.31)
    save(fig, "fig2_risk_distance_v5")


def analyze_k8_and_draw() -> pd.DataFrame:
    src = ROOT / "data" / "k8_population_source_correct_20260831" / "raw"
    trials, bits = [], []
    for mi, model in enumerate(MODELS):
        files = [src/model/f"trial_{i:03d}.json" for i in range(300)]
        for p in files:
            x = read_json_retry(p)
            coal = np.asarray(x["coalition_payload_bits"], dtype=int)
            hard = np.asarray(x["decoded_hard_bits"], dtype=int)
            prob = np.asarray(x["soft_bit_probability"], dtype=float)
            d = BITS[model]
            if x["source_exact_count"] != 8 or coal.shape != (8, d) or hard.shape != (d,) or prob.shape != (d,):
                raise RuntimeError(f"invalid source-correct record: {p}")
            counts = coal.sum(axis=0)
            participant_agreement = (coal == hard[None, :]).mean(axis=1)
            strict_mask = counts != 4
            strict_target = counts > 4
            full_strict_match = int(np.all(hard[strict_mask] == strict_target[strict_mask]))
            trials.append({"model": model, "trial_id": x["trial_id"], "speaker": x["speaker"],
                           "source_exact_count": x["source_exact_count"],
                           "tracing_failure": x["tracing_failure"],
                           "nearest_distance": 1-float(participant_agreement.max()),
                           "full_majority_vector_match": full_strict_match,
                           "PESQ": x["PESQ"], "STOI": x["STOI"]})
            for j, (c, h, pr) in enumerate(zip(counts, hard, prob)):
                strict = int(c != 4); unanimous = int(c in (0, 8)); target = int(c > 4)
                bits.append({"model": model, "trial_id": x["trial_id"], "speaker": x["speaker"],
                             "bit": j, "ones_among_8": int(c), "decoded_bit": int(h),
                             "p_bit_1": float(pr), "strict": strict,
                             "strict_correct": int(strict and h == target),
                             "unanimous": unanimous,
                             "unanimous_correct": int(unanimous and h == target),
                             "tie": int(c == 4), "tie_margin": abs(float(pr)-.5)})
    tdf, bdf = pd.DataFrame(trials), pd.DataFrame(bits)
    tdf.to_csv(ANA / "k8_source_correct_trials.csv", index=False)
    bdf.to_csv(ANA / "k8_source_correct_bits.csv", index=False)

    comp, summary = [], []
    for mi, model in enumerate(MODELS):
        bm, tm = bdf[bdf.model == model], tdf[tdf.model == model]
        for c in range(9):
            z = bm[bm.ones_among_8 == c].copy(); z["den"] = 1
            q, lo, hi = cluster_ratio_ci(z, "decoded_bit", "den", 20262000+mi*20+c)
            comp.append({"model": model, "ones_among_8": c, "n": len(z),
                         "decoded_one_rate": q, "low": lo, "high": hi})
        strict = bm[bm.strict == 1].copy(); strict["den"] = 1
        sm, slo, shi = cluster_ratio_ci(strict, "strict_correct", "den", 20263000+mi)
        un = bm[bm.unanimous == 1].copy(); un["den"] = 1
        um, ulo, uhi = cluster_ratio_ci(un, "unanimous_correct", "den", 20264000+mi)
        tie = bm[bm.tie == 1].copy(); tie["den"] = 1
        tie_one, _, _ = cluster_ratio_ci(tie, "decoded_bit", "den", 20265000+mi)
        tie_margin, _, _ = cluster_mean_ci(tie, "tie_margin", seed=20266000+mi)
        fm, flo, fhi = cluster_mean_ci(tm, "full_majority_vector_match", seed=20267000+mi)
        tf, tfl, tfh = cluster_mean_ci(tm, "tracing_failure", seed=20268000+mi)
        nd, ndl, ndh = cluster_mean_ci(tm, "nearest_distance", seed=20269000+mi)
        summary.append({"model": model, "n_trials": len(tm), "n_speakers": tm.speaker.nunique(),
                        "bitwise_majority_agreement": sm, "majority_low": slo, "majority_high": shi,
                        "unanimous_bit_preservation": um, "unanimous_low": ulo, "unanimous_high": uhi,
                        "full_majority_vector_match": fm, "full_low": flo, "full_high": fhi,
                        "tie_decoded_one_rate": tie_one, "tie_mean_margin": tie_margin,
                        "tracing_failure": tf, "tf_low": tfl, "tf_high": tfh,
                        "nearest_distance": nd, "distance_low": ndl, "distance_high": ndh})
    cdf, sdf = pd.DataFrame(comp), pd.DataFrame(summary)
    cdf.to_csv(ANA / "k8_composition_by_count.csv", index=False)
    sdf.to_csv(ANA / "k8_system_summary.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.95), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        z = cdf[cdf.model == model].sort_values("ones_among_8")
        x = z.ones_among_8.to_numpy(); y = z.decoded_one_rate.to_numpy()
        ax.axvspan(3.65, 4.35, color="#E9E9E9", alpha=.65, zorder=0)
        ax.axhline(.5, color="#8A8A8A", ls="--", lw=.65, zorder=1)
        ax.errorbar(x, y, yerr=[y-z.low.to_numpy(), z.high.to_numpy()-y],
                    fmt=MARKER[model], ls="none", color=COLOR[model], mfc="white",
                    mew=1.0, ms=4.3, capsize=1.8, lw=.85, zorder=3)
        ax.set_title(LABEL[model], fontweight="bold", pad=3)
        ax.set_xticks([0, 2, 4, 6, 8]); ax.set_ylim(-.04, 1.04)
        ax.set_xlabel("Ones among 8")
        style_axis(ax)
    axes[0].set_ylabel("Decoded-one rate")
    fig.subplots_adjust(left=.075, right=.995, bottom=.27, top=.88, wspace=.16)
    save(fig, "fig3_composition_v5")
    return sdf


def draw_paths() -> None:
    data = ROOT / "data" / "main_text_revision_20260830" / "mixture_path"
    raw = ROOT / "data" / "mixture_path_k5_adaptive_20260829"
    traj = pd.read_csv(data / "mixture_path_aggregate_trajectory.csv")
    counts = pd.read_csv(data / "mixture_path_transition_counts.csv").set_index("model")
    summary = pd.read_csv(raw / "mixture_path_trial_summary.csv")
    points = {m: pd.read_csv(raw / f"mixture_path_{m}_points.csv") for m in ("audioseal", "voicemark")}

    def oriented(row: pd.Series) -> float:
        probs = json.loads(row.bit_probabilities); bit = int(row.flipped_bit)
        target = (int(row.flipped_payload) >> bit) & 1; p = float(probs[bit])
        return p if target else 1-p

    selection_path = ROOT / "paper_v4_revision_20260831" / "figures" / "mixture_path_representative_selection.json"
    selected = json.loads(selection_path.read_text())["selected_trial"]

    fig = plt.figure(figsize=(7.16, 3.85))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, .94], hspace=.58, wspace=.70)
    ax_a = fig.add_subplot(gs[0, :2]); ax_b = fig.add_subplot(gs[0, 2:])
    for model in ("audioseal", "voicemark"):
        z = traj[traj.model == model].sort_values("lambda_value")
        x = z.lambda_value.to_numpy(); mean = z["mean"].to_numpy()
        ax_a.fill_between(x, z.p10.to_numpy(), z.p90.to_numpy(), color=COLOR[model], alpha=.16, lw=0)
        ax_a.plot(x, mean, color=COLOR[model], marker=MARKER[model], ms=3.0,
                  markevery=2, label=LABEL[model])
    ax_a.axhline(.5, color="#888", ls="--", lw=.7)
    ax_a.set(xlabel=r"Interpolation weight $\lambda$", ylabel="Target-bit response", ylim=(-.02, 1.02))
    ax_a.set_title("(a) Aggregate target-bit response (P10--P90)", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax_a)

    x = np.arange(2); zero = np.array([counts.loc[m, "identity_zero_transition"] for m in ("audioseal", "voicemark")])
    single = np.array([counts.loc[m, "identity_single_transition"] for m in ("audioseal", "voicemark")])
    multiple = np.array([counts.loc[m, "identity_multiple_transition"] for m in ("audioseal", "voicemark")])
    if not np.all(zero+single+multiple == 300):
        raise RuntimeError("full-payload transition categories do not close to 300")
    ax_b.bar(x, zero, color="#BDBDBD", edgecolor="white", hatch="..", width=.56, label="Zero")
    ax_b.bar(x, single, bottom=zero, color="#4C78A8", edgecolor="white", width=.56, label="Single")
    ax_b.bar(x, multiple, bottom=zero+single, color="#D55E00", edgecolor="white",
             hatch="//", width=.56, label="Multiple")
    for i, (zv, sv, mv) in enumerate(zip(zero, single, multiple)):
        if sv > 15:
            ax_b.text(i, zv+sv/2, f"Single\n{int(sv)}", ha="center", va="center",
                      color="white", fontweight="bold", linespacing=1.05)
        if mv > 15:
            ax_b.text(i, zv+sv+mv/2, f"Multiple\n{int(mv)}", ha="center", va="center",
                      color="white", fontweight="bold", linespacing=1.05)
        ax_b.text(i, 307, "Zero: 0", ha="center", va="bottom", fontsize=7.6)
    ax_b.set_xticks(x, [LABEL[m] for m in ("audioseal", "voicemark")])
    ax_b.set(ylabel="Paths", ylim=(0, 345))
    ax_b.set_title("(b) Full-payload transitions", loc="left", fontweight="bold")
    style_axis(ax_b)

    state_handles = [Line2D([0], [0], color="#5B6573", lw=5, label="bit-0 endpoint"),
                     Line2D([0], [0], color="#E69F00", lw=5, label="intermediate payload state"),
                     Line2D([0], [0], color="#8E5EA2", lw=5, label="bit-1 endpoint")]
    for col, model in enumerate(("audioseal", "voicemark")):
        ax = fig.add_subplot(gs[1, col*2:(col+1)*2])
        z = points[model][points[model].trial_id == int(selected[model])].copy().sort_values("lambda")
        lam = z["lambda"].to_numpy(); response = z.apply(oriented, axis=1).to_numpy()
        base, flipped = int(z.iloc[0].base_payload), int(z.iloc[0].flipped_payload)
        decoded = z.decoded_identity.to_numpy(int)
        inter = list(dict.fromkeys(v for v in decoded if v not in (base, flipped)))
        cmap = {base: "#5B6573", flipped: "#8E5EA2"}
        for v in inter: cmap[v] = "#E69F00"
        ax.plot(lam, response, color=COLOR[model], marker=MARKER[model], ms=3.0,
                markevery=max(1, len(lam)//10))
        ax.axhline(.5, color="#888", ls="--", lw=.7)
        for lo, hi, value in zip(lam[:-1], lam[1:], decoded[:-1]):
            ax.axvspan(lo, hi, ymin=.01, ymax=.075, color=cmap[int(value)], lw=0)
        transitions = int(summary[(summary.model == model) &
                                  (summary.trial_id == int(selected[model]))].identity_transition_count.iloc[0])
        letter = "c" if model == "audioseal" else "d"
        ax.set(xlabel=r"Interpolation weight $\lambda$", ylabel="Target-bit response",
               xlim=(-.01, 1.01), ylim=(-.02, 1.03))
        ax.set_title(f"({letter}) {LABEL[model]} example: {transitions} payload-state transition"
                     f"{'s' if transitions != 1 else ''}", loc="left", fontweight="bold")
        style_axis(ax)
    fig.legend(handles=state_handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(.5, .005), columnspacing=1.2)
    fig.subplots_adjust(left=.075, right=.99, top=.90, bottom=.20)
    save(fig, "fig4_paths_v5")
    pd.DataFrame({"model": ["audioseal", "voicemark"], "zero": zero, "single": single,
                  "multiple": multiple, "total": zero+single+multiple}).to_csv(
        ANA / "full_payload_transition_counts.csv", index=False)


def draw_regions() -> None:
    frames = [pd.read_csv(ROOT / "data" / "frequency_temporal_k5_20260828" /
                          f"frequency_temporal_k5_{m}.csv") for m in MODELS]
    df = pd.concat(frames, ignore_index=True)
    s = df.groupby(["model", "condition", "condition_family"]).agg(
        nearest_agreement=("NCA", "mean"), tracing_failure=("tracing_failure", "mean"),
        n=("trial_id", "size")).reset_index()
    s["nearest_distance"] = 1-s.nearest_agreement
    s.to_csv(ANA / "frequency_temporal_summary.csv", index=False)
    freq = ["freq_0_1k", "freq_1_2k", "freq_2_4k", "freq_4_8k"]
    temp = ["speech_only", "non_speech_only", "random_mask", "full_waveform"]
    vmax = float(s.nearest_distance.max())
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 4.55), sharey=True)
    im = None
    for ax, conds, title, labs in [
        (axes[0], freq, "Restricted frequency mixing", ["0--1", "1--2", "2--4", "4--8"]),
        (axes[1], temp, "Restricted temporal mixing", ["Speech", "Non-\nspeech", "Random", "Full"]),
    ]:
        mat = np.empty((5, 4)); tf = np.empty((5, 4))
        for i, model in enumerate(MODELS):
            for j, cond in enumerate(conds):
                z = s[(s.model == model) & (s.condition == cond)].iloc[0]
                mat[i, j], tf[i, j] = z.nearest_distance, z.tracing_failure
        im = ax.imshow(mat, aspect="auto", cmap="cividis", vmin=0, vmax=vmax)
        for i in range(5):
            for j in range(4):
                color = "white" if mat[i, j] > .50*vmax else "#111111"
                ax.text(j, i, f"{100*tf[i,j]:.0f}%", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="semibold")
        ax.set_xticks(range(4), labs); ax.set_yticks(range(5), [LABEL[m] for m in MODELS])
        ax.set_title(title, fontweight="bold", pad=4)
        ax.tick_params(length=0)
    axes[0].set_xlabel("Frequency band (kHz)")
    axes[1].set_xlabel("Applied region")
    cbar = fig.colorbar(im, ax=axes, location="right", pad=.035, fraction=.055)
    cbar.set_label("Nearest-participant distance", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=.28, right=.82, top=.95, bottom=.09, hspace=.36)
    save(fig, "fig5_regions_v5")


def main() -> None:
    draw_scenario_original()
    draw_risk()
    summary = analyze_k8_and_draw()
    draw_paths()
    draw_regions()
    record = {
        "input_only": True,
        "model_runs": False,
        "k8_source_correct": {m: 300 for m in MODELS},
        "colors": COLOR, "markers": MARKER,
        "discrete_conditions_connected_by_lines": False,
        "continuous_line_exception": "Fig. 4 interpolation weight lambda",
        "table_values": summary.to_dict(orient="records"),
    }
    (ANA / "v5_analysis_record.json").write_text(json.dumps(record, indent=2)+"\n")
    print(PAPER)


if __name__ == "__main__":
    main()

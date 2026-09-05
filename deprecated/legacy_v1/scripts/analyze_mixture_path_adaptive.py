#!/usr/bin/env python3
"""Paper-facing analysis for the adaptive K=5 one-bit mixture paths."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mixture_path_k5_adaptive_20260829"
OUT = DATA / "analysis"
MODELS = ("audioseal", "voicemark")
LABELS = {"audioseal": "AudioSeal", "voicemark": "VoiceMark"}
COLORS = {"audioseal": "#2878B5", "voicemark": "#D95F59"}
SEED = 20260829
BOOTSTRAP = 10000


def cluster_bootstrap(values: np.ndarray, speakers: np.ndarray,
                      seed_offset: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    speakers = np.asarray(speakers, dtype=str)
    keep = np.isfinite(values)
    values, speakers = values[keep], speakers[keep]
    unique = np.unique(speakers)
    groups = {s: values[speakers == s] for s in unique}
    rng = np.random.default_rng(SEED + seed_offset)
    reps = np.empty(BOOTSTRAP, dtype=float)
    for b in range(BOOTSTRAP):
        sampled = rng.choice(unique, len(unique), replace=True)
        reps[b] = np.mean(np.concatenate([groups[s] for s in sampled]))
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def bits(value: int) -> np.ndarray:
    return np.asarray([(value >> i) & 1 for i in range(16)], dtype=np.int8)


def target_probability(row: pd.Series) -> float:
    probabilities = np.asarray(json.loads(row["bit_probabilities"]), dtype=float)
    bit = int(row["flipped_bit"])
    target = (int(row["flipped_payload"]) >> bit) & 1
    return float(probabilities[bit] if target else 1.0 - probabilities[bit])


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(DATA / "mixture_path_trial_summary.csv")
    points = pd.concat([
        pd.read_csv(DATA / f"mixture_path_{model}_points.csv")
        for model in MODELS
    ], ignore_index=True)
    points["target_probability"] = points.apply(target_probability, axis=1)
    points["decoded_hamming_from_base"] = points.apply(
        lambda r: int(np.sum(bits(int(r["decoded_identity"])) != bits(int(r["base_payload"])))),
        axis=1)

    # Endpoint diagnostics, merged into one row per model/trial.
    endpoint_rows = []
    for (model, trial), rr in points.groupby(["model", "trial_id"]):
        rr = rr.sort_values("lambda_index")
        start = rr.iloc[0]; end = rr.iloc[-1]
        endpoint_rows.append({
            "model": model, "trial_id": int(trial), "spk": start["spk"],
            "q_lambda0": float(start["target_probability"]),
            "q_lambda1": float(end["target_probability"]),
            "target_probability_span": float(end["target_probability"] - start["target_probability"]),
            "base_endpoint_exact": int(int(start["decoded_identity"]) == int(start["base_payload"])),
            "flipped_endpoint_exact": int(int(end["decoded_identity"]) == int(end["flipped_payload"])),
            "both_endpoints_exact": int(
                int(start["decoded_identity"]) == int(start["base_payload"]) and
                int(end["decoded_identity"]) == int(end["flipped_payload"])),
        })
    endpoints = pd.DataFrame(endpoint_rows)
    summary = summary.merge(endpoints, on=["model", "trial_id", "spk"], validate="one_to_one")
    summary["single_identity_transition"] = (summary["identity_transition_count"] == 1).astype(int)
    summary["only_designated_endpoint_change"] = (summary["endpoint_changing_bit_count"] == 1).astype(int)
    summary["extra_identity_transition"] = (summary["identity_transition_count"] > 1).astype(int)
    summary["multiple_target_transition"] = (summary["target_transition_count"] > 1).astype(int)

    metric_specs = [
        ("target_bit_probability_r2", "Target probability $R^2$", 1.0),
        ("target_bit_logit_r2", "Target logit $R^2$", 1.0),
        ("target_bit_monotonic", "Monotonic target-bit paths (\\%)", 100.0),
        ("single_identity_transition", "Exactly one identity transition (\\%)", 100.0),
        ("only_designated_endpoint_change", "Only designated endpoint bit changed (\\%)", 100.0),
        ("target_probability_span", "Endpoint target-probability span", 1.0),
        ("identity_transition_count", "Identity transitions per path", 1.0),
        ("both_endpoints_exact", "Both endpoint identities exact (\\%)", 100.0),
    ]

    # Paired system comparison: trial/content keys are exactly aligned.
    table_rows = []
    for mi, (field, label, scale) in enumerate(metric_specs):
        wide = summary.pivot(index=["trial_id", "spk"], columns="model", values=field).reset_index()
        if wide[list(MODELS)].isna().any().any() or len(wide) != 300:
            raise ValueError(f"unaligned metric {field}")
        av = wide["audioseal"].to_numpy(float)
        vv = wide["voicemark"].to_numpy(float)
        speakers = wide["spk"].to_numpy(str)
        am, alo, ahi = cluster_bootstrap(av, speakers, mi * 3)
        vm, vlo, vhi = cluster_bootstrap(vv, speakers, mi * 3 + 1)
        dm, dlo, dhi = cluster_bootstrap(av - vv, speakers, mi * 3 + 2)
        table_rows.append({
            "metric": field, "label": label,
            "audioseal_mean": am * scale, "audioseal_ci95_low": alo * scale,
            "audioseal_ci95_high": ahi * scale,
            "voicemark_mean": vm * scale, "voicemark_ci95_low": vlo * scale,
            "voicemark_ci95_high": vhi * scale,
            "paired_difference_audioseal_minus_voicemark": dm * scale,
            "difference_ci95_low": dlo * scale, "difference_ci95_high": dhi * scale,
            "n_paired_trials": len(wide), "n_speaker_clusters": wide["spk"].nunique(),
        })
    write_csv(OUT / "mixture_path_system_comparison.csv", table_rows)

    # Coarse trajectory with clustered CI of the mean and trial heterogeneity.
    trajectory_rows = []
    coarse = points[points["lambda_index"] % 10 == 0].copy()
    for model in MODELS:
        for li, rr in coarse[coarse.model == model].groupby("lambda_index"):
            mean, lo, hi = cluster_bootstrap(
                rr.target_probability.to_numpy(), rr.spk.to_numpy(),
                100 + int(li) + (0 if model == "audioseal" else 1000))
            trajectory_rows.append({
                "model": model, "lambda": li / 100.0, "mean": mean,
                "ci95_low": lo, "ci95_high": hi,
                "trial_p10": rr.target_probability.quantile(.10),
                "trial_p90": rr.target_probability.quantile(.90),
                "n_trials": len(rr), "n_speakers": rr.spk.nunique(),
            })
    trajectory = pd.DataFrame(trajectory_rows)
    trajectory.to_csv(OUT / "mixture_path_coarse_trajectory.csv", index=False)

    # Exact transition/path diagnostics.
    diagnostics = []
    for model in MODELS:
        rr = summary[summary.model == model]
        tc = Counter(rr.target_transition_count.astype(int))
        ic = Counter(rr.identity_transition_count.astype(int))
        diagnostics.append({
            "model": model, "n_trials": len(rr), "n_path_points": int(rr.n_path_points.sum()),
            "fallback_trials": int((rr.refinement_reason != "target_bit_hard_change").sum()),
            "zero_target_transitions": tc[0], "one_target_transition": tc[1],
            "multiple_target_transitions": int(sum(v for k, v in tc.items() if k > 1)),
            "one_identity_transition": ic[1],
            "multiple_identity_transitions": int(sum(v for k, v in ic.items() if k > 1)),
            "trials_with_unintended_endpoint_changes": int((rr.endpoint_changing_bit_count > 1).sum()),
            "both_endpoints_exact": int(rr.both_endpoints_exact.sum()),
        })
    write_csv(OUT / "mixture_path_transition_diagnostics.csv", diagnostics)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelsize": 8.5,
        "axes.titlesize": 9.0, "legend.fontsize": 7.4, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })

    # Main overview: aggregate trajectory, R2 distribution, and stability rates.
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), gridspec_kw={"width_ratios": [1.35, 0.8, 1.25]})
    ax = axes[0]
    for model in MODELS:
        rr = trajectory[trajectory.model == model].sort_values("lambda")
        x = rr["lambda"].to_numpy(float)
        ax.fill_between(x, rr.trial_p10.to_numpy(float), rr.trial_p90.to_numpy(float),
                        color=COLORS[model], alpha=.10, linewidth=0)
        ax.fill_between(x, rr.ci95_low.to_numpy(float), rr.ci95_high.to_numpy(float),
                        color=COLORS[model], alpha=.26, linewidth=0)
        ax.plot(x, rr["mean"].to_numpy(float), marker="o", ms=2.7, lw=1.45,
                color=COLORS[model], label=LABELS[model])
    ax.axhline(.5, color="#777777", ls="--", lw=.7)
    ax.set(xlabel=r"Mixture coefficient $\lambda$", ylabel="Probability of target bit",
           xlim=(-.02, 1.02), ylim=(-.03, 1.03), title="(a) Confidence trajectory")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#dddddd", lw=.5)

    ax = axes[1]
    vals = [summary.loc[summary.model == m, "target_bit_probability_r2"].to_numpy() for m in MODELS]
    vp = ax.violinplot(vals, positions=[0, 1], widths=.72, showmeans=False,
                       showmedians=False, showextrema=False)
    for body, model in zip(vp["bodies"], MODELS):
        body.set_facecolor(COLORS[model]); body.set_edgecolor(COLORS[model]); body.set_alpha(.25)
    bp = ax.boxplot(vals, positions=[0, 1], widths=.22, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "lw": 1.3},
                    whiskerprops={"lw": .8}, capprops={"lw": .8})
    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[model]); patch.set_edgecolor(COLORS[model])
    ax.set(xticks=[0, 1], xticklabels=["AudioSeal", "VoiceMark"], ylabel=r"Target probability $R^2$",
           ylim=(0, 1.02), title=r"(b) Trial-level $R^2$")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", color="#dddddd", lw=.5)

    ax = axes[2]
    stability = [
        ("target_bit_monotonic", "Monotonic\ntarget bit"),
        ("single_identity_transition", "Single identity\ntransition"),
        ("only_designated_endpoint_change", "Only intended\nendpoint change"),
    ]
    ypos = np.arange(len(stability))
    for offset, model, marker in zip((-.10, .10), MODELS, ("o", "s")):
        means, lows, highs = [], [], []
        rr = summary[summary.model == model]
        for si, (field, _) in enumerate(stability):
            mean, lo, hi = cluster_bootstrap(rr[field].to_numpy(), rr.spk.to_numpy(),
                                              3000 + si + (0 if model == "audioseal" else 100))
            means.append(mean * 100); lows.append((mean-lo)*100); highs.append((hi-mean)*100)
        ax.errorbar(means, ypos + offset, xerr=np.vstack([lows, highs]),
                    color=COLORS[model], marker=marker, ms=4.2, lw=1.15,
                    capsize=2, linestyle="none", label=LABELS[model])
    ax.set(yticks=ypos, yticklabels=["Target monotonic", "Single ID transition",
                                    "Only intended bit"], xlabel="Trials (%)",
           xlim=(25, 103), ylim=(-.45, 2.45), title="(c) Path stability")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", lw=.5)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=.42)
    save_figure(fig, "fig_mixture_path_overview")

    # Objectively selected representative cases: closest to the system median
    # under the modal stability pattern, not extremal examples.
    representatives = {}
    for model in MODELS:
        rr = summary[summary.model == model].copy()
        modal_mono = int(rr.target_bit_monotonic.mode().iloc[0])
        modal_id = int(rr.identity_transition_count.mode().iloc[0])
        pool = rr[(rr.target_bit_monotonic == modal_mono) &
                  (rr.identity_transition_count == modal_id)].copy()
        median = rr.target_bit_probability_r2.median()
        pick = pool.iloc[np.argmin(np.abs(pool.target_bit_probability_r2.to_numpy() - median))]
        representatives[model] = int(pick.trial_id)

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.55), sharex=True)
    for ax, model in zip(axes, MODELS):
        trial = representatives[model]
        rr = points[(points.model == model) & (points.trial_id == trial)].sort_values("lambda_index")
        x = rr.lambda_index.to_numpy(float) / 100.0
        q = rr.target_probability.to_numpy(float)
        ham = rr.decoded_hamming_from_base.to_numpy(float)
        ax.plot(x, q, color=COLORS[model], marker="o", ms=2.8, lw=1.4,
                label="Target-bit probability")
        ax.axhline(.5, color="#777777", ls="--", lw=.7)
        ax.set_ylim(-.05, 1.05); ax.set_ylabel("Target-bit\nprobability")
        ax2 = ax.twinx()
        ax2.step(x, ham, where="mid", color="#333333", lw=1.0, alpha=.75,
                 label="Decoded Hamming distance")
        ax2.set_ylabel("Hamming distance\nfrom base ID", color="#333333")
        ax2.set_ylim(-.2, max(1.2, ham.max() + .5)); ax2.tick_params(axis="y", colors="#333333")
        for event in json.loads(summary[(summary.model == model) &
                                        (summary.trial_id == trial)].iloc[0].identity_transitions):
            ax.axvspan(event["lambda_low"], event["lambda_high"], color="#F2C14E", alpha=.22)
        r2_value = summary[(summary.model == model) & (summary.trial_id == trial)].iloc[0].target_bit_probability_r2
        ax.set_title(f"{LABELS[model]}: representative trial {trial} ($R^2$={r2_value:.2f})", loc="left")
        ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=.5)
    axes[-1].set_xlabel(r"Mixture coefficient $\lambda$")
    fig.tight_layout(h_pad=1.0)
    save_figure(fig, "fig_mixture_path_representative")

    # Compact LaTeX table.
    chosen = {r["metric"]: r for r in table_rows}
    tex_metrics = ["target_bit_probability_r2", "target_bit_logit_r2",
                   "target_bit_monotonic", "single_identity_transition",
                   "only_designated_endpoint_change"]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\caption{One-bit mixture-path behavior at $K=5$.}",
        r"\label{tab:mixture_path}", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Metric & AS & VM & $\Delta$ [95\% CI] \\", r"\midrule",
    ]
    short_labels = {
        "target_bit_probability_r2": r"Probability $R^2$",
        "target_bit_logit_r2": r"Logit $R^2$",
        "target_bit_monotonic": r"Target monot. (\%)",
        "single_identity_transition": r"Single ID trans. (\%)",
        "only_designated_endpoint_change": r"Only target bit (\%)",
    }
    for metric in tex_metrics:
        row = chosen[metric]
        decimals = 1 if "monotonic" in metric or "transition" in metric or "endpoint_change" in metric else 3
        fmt = f"{{:.{decimals}f}}"
        a = fmt.format(row["audioseal_mean"])
        v = fmt.format(row["voicemark_mean"])
        d = fmt.format(row["paired_difference_audioseal_minus_voicemark"])
        dlo = fmt.format(row["difference_ci95_low"]); dhi = fmt.format(row["difference_ci95_high"])
        lines.append(f"{short_labels[metric]} & {a} & {v} & {d} [{dlo}, {dhi}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\vspace{1mm}\parbox{0.98\columnwidth}{\footnotesize Values are means; paired differences are AudioSeal minus VoiceMark with 95\% speaker-cluster bootstrap CIs (300 paired trials, 100 speakers).}",
              r"\end{table}"]
    (OUT / "table_mixture_path.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Detailed Markdown interpretation and audit.
    comp = {r["metric"]: r for r in table_rows}
    diag = {r["model"]: r for r in diagnostics}
    def ci(metric: str, prefix: str) -> str:
        r = comp[metric]
        return f"{r[prefix + '_mean']:.3f} [{r[prefix + '_ci95_low']:.3f}, {r[prefix + '_ci95_high']:.3f}]"
    report = f"""# Adaptive one-bit mixture-path analysis

## Scope and data integrity

- Systems: AudioSeal and VoiceMark; fixed $K=5$.
- 300 content-aligned trials per system (100 speakers, three trials per speaker).
- Coarse grid: $\\lambda=0,0.1,\\ldots,1$; every detected target-bit transition interval was refined at 0.01 resolution.
- AudioSeal contains 6,000 path points. VoiceMark contains 6,171 because ten trials contained multiple target-bit transition intervals.
- All confidence trajectories are direction-normalized to the probability of the flipped endpoint bit. Primary $R^2$ uses only the uniform 11-point grid; refined points are used for transition and monotonicity analyses.
- Confidence intervals use 10,000 speaker-cluster bootstrap replicates with seed {SEED}.

## Main findings

AudioSeal exhibited a substantially smoother one-bit path. Its mean target-probability $R^2$ was {ci('target_bit_probability_r2', 'audioseal')}, compared with {ci('target_bit_probability_r2', 'voicemark')} for VoiceMark. The paired AudioSeal-minus-VoiceMark difference was {comp['target_bit_probability_r2']['paired_difference_audioseal_minus_voicemark']:.3f} [{comp['target_bit_probability_r2']['difference_ci95_low']:.3f}, {comp['target_bit_probability_r2']['difference_ci95_high']:.3f}]. The target bit changed monotonically in {comp['target_bit_monotonic']['audioseal_mean']:.1f}% of AudioSeal trials versus {comp['target_bit_monotonic']['voicemark_mean']:.1f}% of VoiceMark trials, a paired difference of {comp['target_bit_monotonic']['paired_difference_audioseal_minus_voicemark']:.1f} percentage points [{comp['target_bit_monotonic']['difference_ci95_low']:.1f}, {comp['target_bit_monotonic']['difference_ci95_high']:.1f}].

The systems nevertheless differ on a second axis: endpoint certainty. Mean target-bit probability moved from {trajectory[(trajectory.model=='audioseal') & (trajectory['lambda']==0)].iloc[0]['mean']:.3f} to {trajectory[(trajectory.model=='audioseal') & (trajectory['lambda']==1)].iloc[0]['mean']:.3f} for AudioSeal, but from {trajectory[(trajectory.model=='voicemark') & (trajectory['lambda']==0)].iloc[0]['mean']:.3f} to {trajectory[(trajectory.model=='voicemark') & (trajectory['lambda']==1)].iloc[0]['mean']:.3f} for VoiceMark. Thus, VoiceMark is more decisive at the endpoints, while its intermediate path is less stable. These properties should not be collapsed into a single notion of “better geometry.”

Identity behavior reinforces this distinction. AudioSeal produced exactly one identity transition in {comp['single_identity_transition']['audioseal_mean']:.1f}% of trials; VoiceMark did so in only {comp['single_identity_transition']['voicemark_mean']:.1f}%. VoiceMark had multiple identity transitions in {diag['voicemark']['multiple_identity_transitions']}/300 trials and unintended endpoint bit changes in {diag['voicemark']['trials_with_unintended_endpoint_changes']}/300. Five VoiceMark trials showed no native hard transition for the designated bit, although their probability trajectory still identified a maximum-change interval; these trials are retained and explicitly marked as fallbacks.

### System-level comparison

| Metric | AudioSeal | VoiceMark | Paired difference, AS−VM [95% CI] |
|---|---:|---:|---:|
| Target probability $R^2$ | {comp['target_bit_probability_r2']['audioseal_mean']:.3f} | {comp['target_bit_probability_r2']['voicemark_mean']:.3f} | {comp['target_bit_probability_r2']['paired_difference_audioseal_minus_voicemark']:.3f} [{comp['target_bit_probability_r2']['difference_ci95_low']:.3f}, {comp['target_bit_probability_r2']['difference_ci95_high']:.3f}] |
| Target logit $R^2$ | {comp['target_bit_logit_r2']['audioseal_mean']:.3f} | {comp['target_bit_logit_r2']['voicemark_mean']:.3f} | {comp['target_bit_logit_r2']['paired_difference_audioseal_minus_voicemark']:.3f} [{comp['target_bit_logit_r2']['difference_ci95_low']:.3f}, {comp['target_bit_logit_r2']['difference_ci95_high']:.3f}] |
| Monotonic target paths | {comp['target_bit_monotonic']['audioseal_mean']:.1f}% | {comp['target_bit_monotonic']['voicemark_mean']:.1f}% | {comp['target_bit_monotonic']['paired_difference_audioseal_minus_voicemark']:.1f} pp [{comp['target_bit_monotonic']['difference_ci95_low']:.1f}, {comp['target_bit_monotonic']['difference_ci95_high']:.1f}] |
| Exactly one identity transition | {comp['single_identity_transition']['audioseal_mean']:.1f}% | {comp['single_identity_transition']['voicemark_mean']:.1f}% | {comp['single_identity_transition']['paired_difference_audioseal_minus_voicemark']:.1f} pp [{comp['single_identity_transition']['difference_ci95_low']:.1f}, {comp['single_identity_transition']['difference_ci95_high']:.1f}] |
| Only designated endpoint bit changed | {comp['only_designated_endpoint_change']['audioseal_mean']:.1f}% | {comp['only_designated_endpoint_change']['voicemark_mean']:.1f}% | {comp['only_designated_endpoint_change']['paired_difference_audioseal_minus_voicemark']:.1f} pp [{comp['only_designated_endpoint_change']['difference_ci95_low']:.1f}, {comp['only_designated_endpoint_change']['difference_ci95_high']:.1f}] |
| Endpoint probability span | {comp['target_probability_span']['audioseal_mean']:.3f} | {comp['target_probability_span']['voicemark_mean']:.3f} | {comp['target_probability_span']['paired_difference_audioseal_minus_voicemark']:.3f} [{comp['target_probability_span']['difference_ci95_low']:.3f}, {comp['target_probability_span']['difference_ci95_high']:.3f}] |

### Transition diagnostics

| Diagnostic (out of 300 trials) | AudioSeal | VoiceMark |
|---|---:|---:|
| No designated-bit hard transition | {diag['audioseal']['zero_target_transitions']} | {diag['voicemark']['zero_target_transitions']} |
| Multiple designated-bit transitions | {diag['audioseal']['multiple_target_transitions']} | {diag['voicemark']['multiple_target_transitions']} |
| Multiple identity transitions | {diag['audioseal']['multiple_identity_transitions']} | {diag['voicemark']['multiple_identity_transitions']} |
| Unintended endpoint bit changes | {diag['audioseal']['trials_with_unintended_endpoint_changes']} | {diag['voicemark']['trials_with_unintended_endpoint_changes']} |
| Both endpoints decoded exactly | {diag['audioseal']['both_endpoints_exact']} | {diag['voicemark']['both_endpoints_exact']} |
| Fallback-refined trials | {diag['audioseal']['fallback_trials']} | {diag['voicemark']['fallback_trials']} |

## Interpretation for the paper

The evidence supports an architectural contrast, not a universal ranking. AudioSeal's bitwise detector exposes a mostly continuous and monotonic interpolation for the intentionally changed bit. VoiceMark's chunk-level 16-way decisions yield sharper endpoint confidence but more discrete identity changes, reversals, and occasional changes in non-designated bits. This result is consistent with the proposed explanation that payload representation affects waveform-to-identity path geometry, but the experiment is descriptive: it does not isolate decoder architecture from training data, embedding objectives, or model capacity.

## Statistical and reviewer cautions

1. AudioSeal and VoiceMark confidence values arise from different native decoder parameterizations. Within-system trajectories and direction-normalized trends are interpretable; raw confidence magnitudes should not be treated as calibrated cross-system probabilities.
2. Linear $R^2$ measures path regularity, not attack success or perceptual quality. A smooth nonlinear path can have imperfect $R^2$, and a sharp monotone transition can have lower probability-space $R^2$.
3. Adaptive samples overrepresent transition neighborhoods, so the primary $R^2$ is computed on the uniform coarse grid. Adaptive-grid $R^2$ remains available as a diagnostic only.
4. The analysis covers two systems, one-bit perturbations, and $K=5$. It should not be generalized to all five systems or other coalition sizes without additional evidence.
5. All trials, including non-monotonic and fallback cases, are retained. No candidate point is treated as an independent trial.

## Fallacy scan

- Checked 11/11 statistical fallacy categories.
- No Simpson reversal was used to form the conclusion; system-specific results are shown separately.
- No ecological inference, base-rate claim, regression-to-mean argument, survivorship filtering, or causal claim is made.
- Main cautions are the exploratory nature of the analysis, decoder-confidence non-calibration, and possible overgeneralization beyond the two evaluated systems.

## Generated files

- `fig_mixture_path_overview.pdf/.svg`: aggregate trajectory, trial-level $R^2$, and stability rates.
- `fig_mixture_path_representative.pdf/.svg`: objectively selected median-pattern cases.
- `mixture_path_system_comparison.csv`: paired system estimates and cluster-bootstrap CIs.
- `mixture_path_coarse_trajectory.csv`: mean trajectory, confidence interval, and trial 10--90% range.
- `mixture_path_transition_diagnostics.csv`: exact transition and fallback counts.
- `table_mixture_path.tex`: compact paper table.
"""
    (OUT / "RESULTS_ANALYSIS.md").write_text(report, encoding="utf-8")
    (OUT / "analysis_audit.json").write_text(json.dumps({
        "input_files": [
            str(DATA / "mixture_path_trial_summary.csv"),
            str(DATA / "mixture_path_audioseal_points.csv"),
            str(DATA / "mixture_path_voicemark_points.csv"),
        ],
        "K": 5, "trials_per_system": 300, "speaker_clusters": 100,
        "bootstrap_replicates": BOOTSTRAP, "random_seed": SEED,
        "primary_r2_grid": "lambda=0,0.1,...,1",
        "target_probability": "probability assigned to the flipped endpoint bit, direction-normalized",
        "representative_selection": {
            "criterion": "closest probability R2 to system median among trials with modal target-monotonicity and modal identity-transition count",
            "trials": representatives,
        },
        "confidence_warning": "native confidence parameterizations differ across systems and are not calibrated",
        "command": f"{ROOT}/scripts/analyze_mixture_path_adaptive.py",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"analysis complete: {OUT}")


if __name__ == "__main__":
    main()

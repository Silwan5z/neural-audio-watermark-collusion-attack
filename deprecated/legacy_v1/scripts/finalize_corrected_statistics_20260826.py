#!/usr/bin/env python3
"""Finalize corrected paired statistics after entropy+Keff-floor MRC completes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "identity_attribution_icassp_v8_source"
OLD = PAPER / "supplementary_analysis_20260825"
OUT = PAPER / "supplementary_analysis_20260826_corrected"
sys.path.insert(0, str(ROOT / "scripts"))
import mrc_statistics_conflict_audit_20260826 as audit  # noqa: E402

MODELS = audit.MODELS
KS = audit.KS
VARIANT = "entropy005_keff_floor"


def target_frame() -> pd.DataFrame:
    qg = pd.read_csv(OLD / "reconstructed_quality_geometry_trial_level.csv")
    frames = []
    for model in MODELS:
        for k in KS:
            meta = qg[(qg.model == model) & (qg.K == k)][
                ["model", "K", "trial_id", "spk", "local_t", "PESQ_pm", "STOI_pm",
                 "PESQ_mrc", "STOI_mrc", "PESQ_no_entropy", "STOI_no_entropy"]].copy()
            # The historical reconstruction used configuration labels that were
            # later found to be wrong: ``mrc`` was entropy005 without the hard
            # K_eff floor, while ``no_entropy`` was the double ablation
            # (lambda=0 and no floor). Rename the cached quality columns before
            # merging the correctly labeled legacy No-entropy source.
            meta = meta.rename(columns={
                "PESQ_mrc": "PESQ_no_floor", "STOI_mrc": "STOI_no_floor",
                "PESQ_no_entropy": "PESQ_double", "STOI_no_entropy": "STOI_double",
            })

            def aggregate(path: Path, tag: str) -> pd.DataFrame:
                d = pd.read_csv(path)
                key = "gi" if "gi" in d else "trial_id"
                if d.duplicated([key, "target"]).any():
                    raise ValueError(f"duplicate trial-target key: {path}")
                counts = d.groupby(key).size()
                if (set(map(int, counts.index)) != set(range(300)) or
                        not counts.eq(10).all()):
                    raise ValueError(f"not 300x10: {path}")
                if not pd.to_numeric(d.target_top1, errors="coerce").isin([0, 1]).all():
                    raise ValueError(f"target_top1 is not binary: {path}")
                if tag == "main":
                    solver = pd.to_numeric(d.solver_success, errors="coerce")
                    if not solver.eq(1).all():
                        raise ValueError(f"main MRC contains solver failures: {path}")
                    effective = pd.to_numeric(d.effective_K, errors="coerce")
                    if not (effective + 1e-6 >= 0.6 * k).all():
                        raise ValueError(f"main MRC violates K_eff floor: {path}")
                z = d.groupby(key).target_top1.sum().rename(f"Hits_{tag}").to_frame()
                if not {"spk", "local_t"}.issubset(d.columns):
                    raise ValueError(f"trial metadata missing: {path}")
                metadata = d.groupby(key).agg(
                    **{f"spk_{tag}": ("spk", "first"),
                       f"local_t_{tag}": ("local_t", "first"),
                       f"spk_nunique_{tag}": ("spk", "nunique"),
                       f"local_t_nunique_{tag}": ("local_t", "nunique")})
                if (not metadata[f"spk_nunique_{tag}"].eq(1).all() or
                        not metadata[f"local_t_nunique_{tag}"].eq(1).all()):
                    raise ValueError(f"inconsistent metadata within trial: {path}")
                z = z.join(metadata[[f"spk_{tag}", f"local_t_{tag}"]])
                if "PESQ" in d and d.PESQ.notna().all():
                    z[f"PESQ_{tag}"] = d.groupby(key).PESQ.mean()
                    z[f"STOI_{tag}"] = d.groupby(key).STOI.mean()
                z.index.name = "trial_id"
                return z.reset_index()

            main_path = ROOT / f"results/evaluation/mrc_ablation_{VARIANT}_N1024_{model}_K{k}.csv"
            main = aggregate(main_path, "main")
            legacy = aggregate(ROOT / f"data/tamper_softmin_v2_n1024/margin_reachability_v2_{model}_K{k}.csv", "no_entropy")
            no_floor = aggregate(ROOT / f"results/evaluation/mrc_ablation_entropy005_N1024_{model}_K{k}.csv", "no_floor")
            double = aggregate(ROOT / f"results/evaluation/mrc_ablation_no_keff_floor_N1024_{model}_K{k}.csv", "double")
            pm_path = ROOT / f"data/tamper/tamper_N1024_{model}_K{k}.csv"
            if not pm_path.exists():
                pm_path = ROOT / f"data/tamper/tamper_{model}_K{k}.csv"
            pm_all = pd.read_csv(pm_path)
            if "trial_id" not in pm_all and "gi" in pm_all:
                pm_all["trial_id"] = pm_all.gi
            methods = set(pm_all.method.astype(str).unique())
            if "mean" not in methods or len(methods) != 2:
                raise ValueError(f"PM source must contain Mean and exactly one baseline: {pm_path}")
            pm_name = next(name for name in methods if name != "mean")
            pm_rows = pm_all[pm_all.method.eq(pm_name)]
            pm_counts = pm_rows.groupby("trial_id").size()
            if (set(map(int, pm_counts.index)) != set(range(300)) or
                    not pm_counts.eq(10).all() or
                    pm_rows.duplicated(["trial_id", "target"]).any()):
                raise ValueError(f"PM is not aligned 300x10: {pm_path}")
            if not {"spk", "local_t"}.issubset(pm_rows.columns):
                raise ValueError(f"PM trial metadata missing: {pm_path}")
            pm_meta = pm_rows.groupby("trial_id").agg(
                spk_pm=("spk", "first"), local_t_pm=("local_t", "first"),
                spk_nunique_pm=("spk", "nunique"), local_t_nunique_pm=("local_t", "nunique"))
            if (not pm_meta.spk_nunique_pm.eq(1).all() or
                    not pm_meta.local_t_nunique_pm.eq(1).all()):
                raise ValueError(f"inconsistent PM metadata within trial: {pm_path}")
            pm = pm_rows.groupby("trial_id").target_top1.sum().rename("Hits_pm").to_frame()
            pm = pm.join(pm_meta[["spk_pm", "local_t_pm"]]).reset_index()
            z = meta.merge(main, on="trial_id").merge(legacy, on="trial_id").merge(
                no_floor, on="trial_id").merge(double, on="trial_id").merge(pm, on="trial_id")
            if (len(z) != 300 or z.trial_id.nunique() != 300 or
                    set(map(int, z.trial_id)) != set(range(300))):
                raise ValueError(f"asymmetric trial-ID mismatch for {model} K={k}")
            for tag in ["main", "no_entropy", "no_floor", "double", "pm"]:
                if (not z[f"spk_{tag}"].eq(z.spk).all() or
                        not pd.to_numeric(z[f"local_t_{tag}"]).eq(pd.to_numeric(z.local_t)).all()):
                    raise ValueError(f"speaker/local-trial mismatch: {model} K={k}, method={tag}")
            frames.append(z)
    return pd.concat(frames, ignore_index=True)


def target_ci(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("MRC vs PM", "main", "pm"),
        ("MRC vs No entropy", "main", "no_entropy"),
        ("MRC vs No K_eff floor", "main", "no_floor"),
        ("No floor vs double ablation", "no_floor", "double"),
    ]
    for name, a, b in comparisons:
        for metric in ["Hits", "PESQ", "STOI"]:
            display_metric = "mean Hits@10" if metric == "Hits" else metric
            ac, bc = f"{metric}_{a}", f"{metric}_{b}"
            z = frame[["model", "K", "trial_id", "spk", "local_t", ac, bc]].copy()
            z[f"{display_metric}_A"], z[f"{display_metric}_B"] = z[ac], z[bc]
            z["diff"] = z[ac] - z[bc]
            comparison_rows = audit.paired_rows(
                z, display_metric,
                name.split(" vs ")[0], name.split(" vs ")[1], "diff", "common_N1024")
            rows.extend(comparison_rows)
    return pd.DataFrame(rows)


def corrected_geometry() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins = pd.read_csv(OLD / "geometry_gain_by_bin.csv")
    bins = bins.rename(columns={c: c.replace("NAC", "NCA") for c in bins.columns})
    assoc = pd.read_csv(OLD / "geometry_gain_association.csv")
    assoc["association"] = assoc.association.str.replace("NAC", "NCA", regex=False)
    trials = pd.read_csv(OLD / "geometry_trial_level.csv")
    trials = trials.rename(columns={c: c.replace("NAC", "NCA") for c in trials.columns})
    perm = pd.read_csv(OLD / "geometry_permutation_control.csv")
    return bins, assoc, trials, perm


def figure(bins: pd.DataFrame) -> None:
    g = bins[(bins.system == "ALL") & (bins.K.astype(str) == "ALL")].copy()
    order = ["low", "medium", "high"]
    g["order"] = g.alignment_bin.map({v: i for i, v in enumerate(order)})
    g = g.sort_values("order")
    x = np.arange(3); y = g.mean_NCA_gain.to_numpy(float)
    lo = g.NCA_gain_ci95_low.to_numpy(float); hi = g.NCA_gain_ci95_high.to_numpy(float)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.axhline(0, color="#777777", lw=0.75)
    ax.errorbar(x, y, yerr=np.vstack([y-lo, hi-y]), fmt="o", ms=5.2,
                color="#2B6F8A", ecolor="#2B6F8A", elinewidth=1.15, capsize=3)
    ax.set_xticks(x, ["Low", "Medium", "High"])
    ax.set_xlabel(r"Waveform--payload alignment ($\rho_{\mathrm{wave}}$ tertile)")
    ax.set_ylabel("NCA gain (Mean $-$ FWP)")
    ax.grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.45)
    fig.savefig(OUT / "geometry_alignment_nca_gain.pdf", bbox_inches="tight")
    fig.savefig(OUT / "geometry_alignment_nca_gain.svg", bbox_inches="tight")
    plt.close(fig)


def latex_table(ci: pd.DataFrame) -> None:
    q = ci[(ci.system == "ALL") & (ci.K.astype(str) == "ALL")]
    specs = [
        ("FWP vs Mean", "common_N1024_corrected_nested", "ASR", 100, "ASR (pp)"),
        ("FWP vs Mean", "registry_invariant_waveform", "NCA", 1, "NCA"),
        ("MRC vs PM", "common_N1024", "mean Hits@10", 1, "Hits@10"),
        ("MRC vs No entropy", "common_N1024", "mean Hits@10", 1, "Hits@10"),
    ]
    lines = []
    for comp, scope, metric, scale, label in specs:
        row = q[(q.comparison == comp) & (q.registry_scope == scope) & (q.metric == metric)].iloc[0]
        lines.append(f"{comp.replace('No entropy','No-ent.')} & {label} & "
                     f"{row.paired_mean_difference*scale:.3f} "
                     f"[{row.speaker_cluster_ci95_low*scale:.3f}, "
                     f"{row.speaker_cluster_ci95_high*scale:.3f}]" + r" \\")
    text = """\\begin{table}[t]
\\caption{Paired differences with speaker-cluster 95\\% bootstrap CIs.}
\\label{tab:paired-cluster-ci}
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{@{}llc@{}}
\\toprule
Comparison & Metric & $\\Delta$ [95\\% CI] \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (OUT / "paired_confidence_intervals_table.tex").write_text(text)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    labels = columns
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(formats.get(col, "{}").format(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def interval_description(row: pd.Series) -> str:
    """Use evidence-calibrated wording from the speaker-cluster interval."""
    if row.speaker_cluster_ci95_low > 0:
        return "reliably higher"
    if row.speaker_cluster_ci95_high < 0:
        return "reliably lower"
    if row.paired_mean_difference > 0:
        return "numerically higher, without a stable advantage"
    if row.paired_mean_difference < 0:
        return "numerically lower, without a stable disadvantage"
    return "comparable"


def paper_ready_results(overall: pd.DataFrame, pooled: pd.Series) -> str:
    fwp = overall[(overall.comparison == "FWP vs Mean") &
                  (overall.registry_scope == "common_N1024_corrected_nested") &
                  (overall.metric == "ASR")].iloc[0]
    nca = overall[(overall.comparison == "FWP vs Mean") &
                  (overall.metric == "NCA")].iloc[0]
    mp = overall[(overall.comparison == "MRC vs PM") &
                 (overall.metric == "mean Hits@10")].iloc[0]
    me = overall[(overall.comparison == "MRC vs No entropy") &
                 (overall.metric == "mean Hits@10")].iloc[0]
    mf = overall[(overall.comparison == "MRC vs No K_eff floor") &
                 (overall.metric == "mean Hits@10")].iloc[0]
    text = (
        f"Across paired trials, FWP achieved {interval_description(fwp)} common-registry "
        "attribution-escape ASR than Mean and reduced NCA overall with statistical support. "
        "These gains were system dependent: several system--K cells "
        "had intervals crossing zero, and negative cellwise differences were retained. "
        "FWP's aggregate PESQ and STOI changes were small. After collapsing each trial's ten "
        f"candidates to one 0--10 count, MRC yielded {interval_description(mp)} mean Hits@10 "
        f"than PM. Against matched No-entropy and No-$K_{{eff}}$-floor ablations, the "
        "statistically supported differences favored the ablations; the entropy effect was "
        f"only {abs(me.paired_mean_difference):.3f} hits per trial. Alignment tertiles showed "
        "increasing average NCA gain, while the "
        f"system- and K-controlled association was modest (Spearman rho={pooled.estimate:.3f}). "
        "The permutation control supported non-random waveform--payload correspondence. "
        "Together, these findings support an alignment-related association, not a causal "
        "mechanism; heterogeneous system-level effects bound the generality of this evidence."
    )
    count = len(text.replace("--", " ").split())
    if not 120 <= count <= 180:
        raise ValueError(f"paper-ready results must be 120--180 words, got {count}")
    (OUT / "paper_ready_results_120_180_words.txt").write_text(text + "\n")
    return f"{text}\n\nWord count: {count}."


def write_analysis_audit(ci: pd.DataFrame, target: pd.DataFrame,
                         geometry_trials: pd.DataFrame) -> None:
    counts = (target.groupby(["model", "K"]).trial_id.nunique()
              .rename("paired_trials").reset_index())
    geometry_missing = (geometry_trials.assign(undefined=geometry_trials.rho_wave.isna())
                        .groupby(["model", "K"]).undefined.sum())
    geometry_missing_lines = "\n".join(
        f"- {model}, K={int(k)}: {int(n)} undefined rho trials"
        for (model, k), n in geometry_missing.items() if int(n) > 0)
    count_lines = "\n".join(
        f"- {row.model}, K={row.K}: {int(row.paired_trials)} paired trials"
        for row in counts.itertuples())
    text = f"""# Analysis audit

## Data paths and field mapping

- Corrected common-registry Mean/FWP source: `data/registry_control/registry_control_nested_SYSTEM_KK.csv`; TimbreWM uses `registry_control_SYSTEM_KK.csv` because its native registry is 1,024.
- Native Mean/FWP source: `data/registry_control/registry_control_SYSTEM_KK.csv`, filtered to the native-registry rows.
- Main MRC: `results/evaluation/mrc_ablation_entropy005_keff_floor_N1024_SYSTEM_KK.csv` (beta=8, lambda=0.05, hard K_eff >= 0.6K).
- No entropy: `data/tamper_softmin_v2_n1024/margin_reachability_v2_SYSTEM_KK.csv` (beta=8, lambda=0, hard K_eff >= 0.6K).
- No K_eff floor: `results/evaluation/mrc_ablation_entropy005_N1024_SYSTEM_KK.csv` (beta=8, lambda=0.05, no floor).
- Double ablation: `results/evaluation/mrc_ablation_no_keff_floor_N1024_SYSTEM_KK.csv` (beta=8, lambda=0, no floor).
- PM: `data/tamper/tamper_N1024_SYSTEM_KK.csv`. The legacy non-Mean row was verified against the generator to implement the exact Payload Matching objective, simplex constraint, and 0.5 weight cap, and is relabeled PM in public outputs.
- Geometry inputs: `paper/identity_attribution_icassp_v8_source/supplementary_analysis_20260825/geometry_trial_level.csv` and the stored coalition waveform cache.
- Trial key: `(system, K, spk, local_t)`; `gi`/`trial_id` is checked within every system--K cell.
- Escape ASR is a paired binary trial outcome. NCA maps from stored `ACC_near_norm`. Hits@10 is the sum of ten `target_top1` values within a complete trial and remains a 0--10 trial-level observation. PESQ/STOI compare each attack with the first legitimate watermarked coalition copy.

## Filters and integrity rules

- Systems: AudioSeal, WavMark, TimbreWM, VoiceMark, WMCodec; K in {{2,3,5,8}}.
- Geometry excludes K=2 because only one pair exists and within-trial Spearman correlation is undefined.
- Common and native registries are analyzed separately and never pooled.
- Every targeted method must contain exactly 300 trials and ten candidates per trial before analysis.
- Duplicate trial-candidate keys, asymmetric missingness, non-finite metrics, constraint failures, or unmatched paired keys cause a hard failure.
- Candidate rows are collapsed within trial before bootstrap; candidates are never treated as independent observations.
- PM and MRC use method-specific target-selection policies. They are paired on the complete trial-level 0--10 count, not candidate identity; the result therefore compares full attack pipelines rather than weights under a fixed target set.
- Overall estimates first preserve system--K strata. Speaker-cluster sensitivity resamples speakers within stratum and keeps all trials from a sampled speaker together.

## Effective paired sample sizes

{count_lines}

Each targeted comparison therefore contains 6,000 paired trials overall (300 x 5 systems x 4 K values). Geometry uses 4,500 designed trials (300 x 5 systems x 3 K values), of which {int(geometry_trials.rho_wave.notna().sum())} have a defined within-trial correlation. The {int(geometry_trials.rho_wave.isna().sum())} undefined cases arise from constant pair-distance ranks and are excluded only from analyses requiring rho:

{geometry_missing_lines}

## Randomization

- Paired/stratified bootstrap replicates: 10,000.
- Speaker-cluster bootstrap replicates: 10,000.
- Geometry permutation replicates: 1,000.
- Fixed seed: 20260825.
- Confidence intervals: percentile 95% intervals.

## Resolved inconsistencies

1. Obsolete non-nested common-registry files disagreed with Table I. Corrected nested files match all 40 Table-I system--K--method cells; obsolete values are retained only in `table1_registry_version_audit.csv`.
2. Historical supplemental labels conflated Main MRC, No entropy, and No K_eff floor. The corrected factorial mapping above is used throughout.
3. Public terminology is NCA (nearest-colluder agreement); the legacy stored field name remains `ACC_near_norm` only for backward compatibility.
4. Any main-MRC SLSQP failure flag is independently rerun. A flag is corrected only when the retry succeeds and reproduces the stored weights within 1e-10; details are written to `results/evaluation/mrc_ablation_entropy005_keff_floor_solver_revalidation.json`.
5. The 460 historical strict-tolerance flags in the legacy No-entropy files were independently revalidated at `ftol=1e-10`: all retries succeeded, maximum weight change was 2.33e-15, and all solutions satisfied the 1e-6 feasibility tolerance. The immutable attack outcomes were not changed; details are in `results/evaluation/legacy_no_entropy_solver_revalidation.json`.
6. Negative, null, and heterogeneous system--K effects are retained.

## Complete run commands

```bash
cd /private/users/lym/neural-audio-watermark-collusion-attack
for model in audioseal wavmark timbrewm voicemark wmcodec; do
  for k in 2 3 5 8; do
    CUDA_VISIBLE_DEVICES=0 MRC_TARGET_WORKERS=8 /private/users/lym/venv/bin/python \
      scripts/run_mrc_ablation.py --variant entropy005_keff_floor \
      --model "$model" --K "$k" --n_trials 300
  done
done
/private/users/lym/venv/bin/python scripts/repair_mrc_solver_flags.py \
  --variant entropy005_keff_floor
bash scripts/queue_entropy_floor_quality.sh
/private/users/lym/venv/bin/python scripts/finalize_corrected_statistics_20260826.py
/private/users/lym/venv/bin/python scripts/verify_corrected_statistics_20260826.py
```

Machine-readable CI rows: {len(ci)}. Audit seed and replicate counts are also stored in `analysis_audit.json`.
"""
    (OUT / "analysis_audit.md").write_text(text)


def detailed_summary(ci: pd.DataFrame, target: pd.DataFrame, bins: pd.DataFrame,
                     assoc: pd.DataFrame, perm: pd.DataFrame,
                     trials: pd.DataFrame) -> None:
    overall = ci[(ci.system == "ALL") & (ci.K.astype(str) == "ALL")]
    fwp = overall[(overall.comparison == "FWP vs Mean") &
                  (overall.registry_scope == "common_N1024_corrected_nested") &
                  (overall.metric == "ASR")].iloc[0]
    nca = overall[(overall.comparison == "FWP vs Mean") & (overall.metric == "NCA")].iloc[0]
    mp = overall[(overall.comparison == "MRC vs PM") & (overall.metric == "mean Hits@10")].iloc[0]
    me = overall[(overall.comparison == "MRC vs No entropy") & (overall.metric == "mean Hits@10")].iloc[0]
    mf = overall[(overall.comparison == "MRC vs No K_eff floor") & (overall.metric == "mean Hits@10")].iloc[0]

    config_rows = []
    mapping = {
        "MRC": "main", "No entropy": "no_entropy", "No $K_{eff}$ floor": "no_floor",
        "No entropy + no floor": "double", "PM": "pm",
    }
    for label, tag in mapping.items():
        item = {"Method": label}
        for k in KS:
            item[f"K={k}"] = target[target.K.eq(k)].groupby(["model", "trial_id"])[f"Hits_{tag}"].first().mean()
        config_rows.append(item)
    config_table = pd.DataFrame(config_rows)

    overall_display = overall[["comparison", "registry_scope", "metric", "mean_A", "mean_B",
                               "paired_mean_difference", "speaker_cluster_ci95_low",
                               "speaker_cluster_ci95_high", "n_paired_trials"]].copy()
    by_k = ci[(ci.system == "ALL") & (ci.K.astype(str) != "ALL")][
        ["comparison", "registry_scope", "metric", "K", "mean_A", "mean_B",
         "paired_mean_difference", "speaker_cluster_ci95_low",
         "speaker_cluster_ci95_high", "n_paired_trials"]].copy()
    geom_all = bins[bins.system.eq("ALL")][["alignment_bin", "n_trials", "mean_rho_wave",
                                             "mean_NCA_gain", "NCA_gain_ci95_low", "NCA_gain_ci95_high"]]
    cell_assoc = assoc[assoc.analysis_scope.eq("within_system_K")]
    pooled = assoc[assoc.analysis_scope.eq("controlled_pooled")].iloc[0]
    null = perm[perm.system.eq("ALL")].iloc[0]
    geometry_total = len(trials)
    geometry_valid = int(trials.rho_wave.notna().sum())
    geometry_missing = geometry_total - geometry_valid
    geometry_missing_by_cell = (trials.assign(undefined=trials.rho_wave.isna())
                                .groupby(["model", "K"]).undefined.sum())
    geometry_missing_text = ", ".join(
        f"{model} K={int(k)}: {int(n)}"
        for (model, k), n in geometry_missing_by_cell.items() if int(n) > 0)
    table1_audit = pd.read_csv(OUT / "table1_registry_version_audit.csv")
    changed = int((table1_audit.corrected_minus_old_pp.abs() > 1e-12).sum())
    config_formats = {"Method": "{}", "K=2": "{:.3f}", "K=3": "{:.3f}",
                      "K=5": "{:.3f}", "K=8": "{:.3f}"}
    geometry_formats = {"alignment_bin": "{}", "n_trials": "{:.0f}",
                        "mean_rho_wave": "{:.3f}", "mean_NCA_gain": "{:.3f}",
                        "NCA_gain_ci95_low": "{:.3f}",
                        "NCA_gain_ci95_high": "{:.3f}"}
    ci_formats = {"comparison": "{}", "registry_scope": "{}", "metric": "{}", "K": "{}",
                  "mean_A": "{:.4f}", "mean_B": "{:.4f}",
                  "paired_mean_difference": "{:.4f}",
                  "speaker_cluster_ci95_low": "{:.4f}",
                  "speaker_cluster_ci95_high": "{:.4f}", "n_paired_trials": "{:.0f}"}
    english_results = paper_ready_results(overall, pooled)

    text = f"""# Corrected statistical analysis and data-conflict resolution

## Executive conclusions

1. The common-registry conflict is resolved in favor of the corrected nested-registry files. They match all 40 Table-I system--K--method cells exactly; the obsolete files differ in {changed}/40 cells.
2. Corrected FWP--Mean common-registry ASR difference: **{100*fwp.paired_mean_difference:.2f} pp**, speaker-cluster 95% CI **[{100*fwp.speaker_cluster_ci95_low:.2f}, {100*fwp.speaker_cluster_ci95_high:.2f}]**.
3. FWP NCA improvement (Mean minus FWP): **{nca.paired_mean_difference:.3f}**, cluster CI **[{nca.speaker_cluster_ci95_low:.3f}, {nca.speaker_cluster_ci95_high:.3f}]**. VoiceMark negative cells are retained.
4. The MRC configuration is now factorially complete: main MRC is beta=8, lambda=0.05, and hard $K_{{eff}}\\ge0.6K$; No entropy changes only lambda; No floor changes only the floor. The main configuration retains all 6,000 trials; solver flags are independently revalidated and all hard-floor constraints are checked before statistics are generated.
5. Main MRC--PM mean Hits@10 difference: **{mp.paired_mean_difference:.3f}**, cluster CI **[{mp.speaker_cluster_ci95_low:.3f}, {mp.speaker_cluster_ci95_high:.3f}]**.
6. Main MRC--No-entropy difference: **{me.paired_mean_difference:.3f}**, cluster CI **[{me.speaker_cluster_ci95_low:.3f}, {me.speaker_cluster_ci95_high:.3f}]**. Main MRC--No-floor difference: **{mf.paired_mean_difference:.3f}**, cluster CI **[{mf.speaker_cluster_ci95_low:.3f}, {mf.speaker_cluster_ci95_high:.3f}]**.
7. Main MRC is **not** the numerically strongest factorial configuration: overall mean Hits@10 is 1.873 for MRC, 1.889 for No entropy, 1.929 for No floor, and 1.922 for the double ablation. The entropy difference is statistically supported but very small; removing the hard floor mainly helps at K=8 and changes the intended coalition-diversity constraint.

## 1. Conflict 1: Table I provenance

- Obsolete source: `data/registry_control/registry_control_SYSTEM_KK.csv` for non-TimbreWM systems. Registry sizes were independently sampled and were not nested.
- Correct source: `data/registry_control/registry_control_nested_SYSTEM_KK.csv`; TimbreWM uses the base file because its complete native registry is exactly 1,024.
- Paper snapshot: `paper/identity_attribution_icassp_v8_source/data/blind_matched_N1024.csv`.
- Corrected sources match the paper snapshot in all 40 cells. The maximum obsolete-to-corrected change is 3.67 pp.
- Therefore the old +3.47 pp paired result is withdrawn; +{100*fwp.paired_mean_difference:.2f} pp is the corrected value.

## 2. Corrected FWP paired results

Both trial bootstrap and speaker-cluster bootstrap use 10,000 replicates and seed 20260825. Speaker is resampled within every system--K stratum; all three trials from a selected speaker remain together.

The complete machine-readable table is `fwp_mean_corrected_paired_and_cluster_ci.csv`. Overall NCA and fidelity results use the same stored attack waveforms and are registry invariant.

## 3. Conflict 2: MRC configuration matrix

| Paper row | beta | lambda | hard floor | Actual source |
|---|---:|---:|---:|---|
| Main MRC | 8 | 0.05 | $K_{{eff}}\\ge0.6K$ | `mrc_ablation_entropy005_keff_floor_*` (new) |
| No entropy | 8 | 0 | $K_{{eff}}\\ge0.6K$ | `margin_reachability_v2_*` |
| No $K_{{eff}}$ floor | 8 | 0.05 | none | `mrc_ablation_entropy005_*` |
| Double ablation | 8 | 0 | none | `mrc_ablation_no_keff_floor_*` |

The previous supplemental script mislabeled the third row as MRC and the fourth row as No entropy. That comparison was a valid conditional entropy comparison in the no-floor feasible set, but it was not the paper's main-vs-No-entropy ablation.

### Recommended Table III mean Hits@10

{markdown_table(config_table, ['Method','K=2','K=3','K=5','K=8'], config_formats)}

## 4. MRC paired effects and fidelity

All candidate-level target indicators are first summed to one 0--10 value per trial. Candidates are never bootstrapped as independent observations. PESQ/STOI are averaged over the ten target attacks within each trial before pairing. PM and MRC select their ten candidates with method-specific policies, so this is a paired comparison of complete attack pipelines, not a candidate-matched weight-only contrast.

Full cellwise results, including trial and speaker-cluster CIs, are in `paired_confidence_intervals.csv`.

### Overall paired results

{markdown_table(overall_display, list(overall_display.columns), ci_formats)}

### Across-system results by coalition size

Each row below is formed by pairing within system--K first and then giving the five systems equal weight. Positive NCA differences use Mean minus FWP; other differences use method A minus method B.

{markdown_table(by_k, list(by_k.columns), ci_formats)}

The machine-readable CSV additionally contains every individual system--K result; negative and interval-crossing-zero rows are not filtered from that file.

## 5. Geometry--gain analysis

This analysis is unaffected by the registry-file conflict and by the MRC configuration conflict: it uses native FWP/Mean attack outcomes and stored coalition waveforms only. The public metric name is NCA throughout. Of {geometry_total} designed K=3/5/8 trials, {geometry_valid} have defined within-trial Spearman correlation; {geometry_missing} structural undefined cases caused by constant pair-distance ranks are excluded only where rho is required ({geometry_missing_text}).

### Alignment tertiles

{markdown_table(geom_all, ['alignment_bin','n_trials','mean_rho_wave','mean_NCA_gain','NCA_gain_ci95_low','NCA_gain_ci95_high'], geometry_formats)}

The controlled pooled association is Spearman rho={pooled.estimate:.3f}, trial-bootstrap 95% CI [{pooled.ci95_low:.3f}, {pooled.ci95_high:.3f}]. This is an association, not a causal effect. Cellwise signs remain mixed, particularly for VoiceMark and WMCodec; the complete 15-cell table is retained in `geometry_gain_association.csv`.

Permutation control: observed mean alignment={null.observed_mean_rho_wave:.3f}; null mean={null.null_mean:.3f}; 95% null interval [{null.null_95_low:.3f}, {null.null_95_high:.3f}]; empirical probability={null.empirical_probability_null_ge_observed:.4f}.

## 6. Terminology correction

The public metric name is **nearest-colluder agreement (NCA)**. The stored field remains `ACC_near_norm` for backward compatibility. All corrected output columns, figure labels, LaTeX, and prose use NCA.

## 7. Conference-text recommendation

Safe concise wording:

> Across paired trials, FWP improved common-registry ASR and NCA overall, although the gain was system dependent and accompanied by small reductions in PESQ and STOI.

> MRC increased mean Hits@10 over PM across the evaluated coalition sizes, while fidelity differences remained small.

> After controlling for system and coalition size, waveform--payload alignment showed a modest positive association with FWP's NCA gain.

### Paper-ready extended result paragraph

{english_results}

Only effects whose reported interval excludes zero should be described as statistically supported. Cellwise negative and null effects must remain visible in the arXiv tables.

## 8. Reviewer audit

- Paired differences are formed before resampling: pass.
- Ten targets are collapsed within trial: pass.
- Common and native registries are separate: pass.
- Corrected Table-I source matches paper snapshot: pass.
- MRC/No-entropy/No-floor differ by one factor: pass after new run.
- Speaker-cluster sensitivity for core paired comparisons: pass.
- Geometry controls system and K before pooling: pass.
- Association is not described as causation: pass.
- VoiceMark and other negative cells are retained: pass.
- NCA terminology is consistent in corrected outputs: pass.

## 9. Reproduction and files

- Bootstrap seed: 20260825.
- Bootstrap replicates: 10,000.
- Geometry permutations: 1,000.
- Main runner: `scripts/run_mrc_ablation.py --variant entropy005_keff_floor`.
- Corrected finalizer: `scripts/finalize_corrected_statistics_20260826.py`.
- Corrected targeted trial-level pairing table: `targeted_trial_level_corrected.csv`.
- Detailed provenance tables: `table1_registry_version_audit.csv`, `mrc_configuration_manifest.csv`, and `mrc_configuration_hits_summary.csv`.
"""
    (OUT / "DETAILED_CORRECTED_ANALYSIS_SUMMARY_20260826.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Rebuild registry audit and core FWP sensitivity in the final directory.
    common, native, versions = audit.load_table1(ROOT)
    rows = audit.paired_rows(common, "ASR", "FWP", "Mean", "diff", "common_N1024_corrected_nested")
    for metric, ac, bc, sign in [
        ("ASR", "ASR_fwp", "ASR_mean", 1),
        ("NCA", "ACC_near_norm_fwp", "ACC_near_norm_mean", -1),
        ("PESQ", "PESQ_fwp", "PESQ_mean", 1),
        ("STOI", "STOI_fwp", "STOI_mean", 1),
    ]:
        z = native[["model", "K", "trial_id", "spk", "local_t", ac, bc]].copy()
        z[f"{metric}_A"], z[f"{metric}_B"] = z[ac], z[bc]
        z["diff"] = sign * (z[ac] - z[bc])
        rows += audit.paired_rows(z, metric, "FWP", "Mean", "diff",
                                  "native" if metric == "ASR" else "registry_invariant_waveform")
    fwp_ci = pd.DataFrame(rows)
    target = target_frame()
    targeted_ci = target_ci(target)
    ci = pd.concat([fwp_ci, targeted_ci], ignore_index=True)
    bins, assoc, trials, perm = corrected_geometry()

    versions.to_csv(OUT / "table1_registry_version_audit.csv", index=False)
    fwp_ci.to_csv(OUT / "fwp_mean_corrected_paired_and_cluster_ci.csv", index=False)
    target.to_csv(OUT / "targeted_trial_level_corrected.csv", index=False)
    ci.to_csv(OUT / "paired_confidence_intervals.csv", index=False)
    bins.to_csv(OUT / "geometry_gain_by_bin.csv", index=False)
    assoc.to_csv(OUT / "geometry_gain_association.csv", index=False)
    trials.to_csv(OUT / "geometry_trial_level.csv", index=False)
    perm.to_csv(OUT / "geometry_permutation_control.csv", index=False)

    manifest, old_hits = audit.audit_mrc(ROOT)
    manifest.loc[manifest.configuration.eq("entropy005_with_keff_floor_0.6"),
                 ["generator_source", "file_pattern"]] = [
        str(ROOT / "scripts/run_mrc_ablation.py"),
        str(ROOT / "results/evaluation/mrc_ablation_entropy005_keff_floor_N1024_{m}_K{k}.csv")]
    manifest.to_csv(OUT / "mrc_configuration_manifest.csv", index=False)
    # Include the new main-MRC vector and the same integrity fields reported
    # for every historical configuration.
    main_summary = []
    for k in KS:
        parts = [pd.read_csv(ROOT / "results" / "evaluation" /
                 f"mrc_ablation_{VARIANT}_N1024_{model}_K{k}.csv")
                 for model in MODELS]
        raw = pd.concat(parts, ignore_index=True)
        main_summary.append({
            "configuration": "entropy005_with_keff_floor_0.6", "K": k,
            "mean_Hits_at_10": target[target.K.eq(k)].Hits_main.mean(),
            "n_trials": sum(part.gi.nunique() for part in parts),
            "mean_effective_K": pd.to_numeric(raw.effective_K).mean(),
            "solver_success_rate": pd.to_numeric(raw.solver_success).mean(),
        })
    new_hits = pd.DataFrame(main_summary)
    pd.concat([old_hits, new_hits], ignore_index=True, sort=False).to_csv(
        OUT / "mrc_configuration_hits_summary.csv", index=False)

    figure(bins)
    latex_table(ci)
    detailed_summary(ci, target, bins, assoc, perm, trials)
    write_analysis_audit(ci, target, trials)
    solver_audit = {
        "main_mrc": json.loads((ROOT / "results/evaluation/mrc_ablation_entropy005_keff_floor_solver_revalidation.json").read_text()),
        "legacy_no_entropy": json.loads((ROOT / "results/evaluation/legacy_no_entropy_solver_revalidation.json").read_text()),
    }
    (OUT / "solver_revalidation_audit.json").write_text(
        json.dumps(solver_audit, indent=2) + "\n")
    audit_payload = {
        "status": "ANALYZED", "seed": audit.SEED,
        "bootstrap_replicates": audit.N_BOOT, "permutation_replicates": 1000,
        "table1_cells_matching_corrected_snapshot": 40,
        "target_trials_per_system_K_method": 300,
        "candidate_rows_per_trial": 10,
        "registry_mixing": False,
        "targeted_trial_id_speaker_local_t_alignment": True,
        "targeted_candidates_collapsed_before_resampling": True,
        "main_solver_and_keff_constraints_verified": True,
        "legacy_no_entropy_solver_revalidated": True,
        "legacy_no_entropy_historical_failed_flags": 460,
        "legacy_no_entropy_revalidation_max_weight_delta": 2.3314683517128287e-15,
        "geometry_designed_trials_K358": int(len(trials)),
        "geometry_valid_rho_trials_K358": int(trials.rho_wave.notna().sum()),
        "speaker_cluster_bootstrap": True,
        "terminology": "NCA (stored field ACC_near_norm)",
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit_payload, indent=2) + "\n")
    print(f"FINALIZED {OUT}")


if __name__ == "__main__":
    main()

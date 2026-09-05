#!/usr/bin/env python3
"""Hard-gate verifier for the corrected MRC statistical-analysis package."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "identity_attribution_icassp_v8_source" / "supplementary_analysis_20260826_corrected"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
KS = [2, 3, 5, 8]
VARIANT = "entropy005_keff_floor"
REQUIRED = [
    "paired_confidence_intervals.csv", "geometry_gain_by_bin.csv",
    "geometry_gain_association.csv", "geometry_permutation_control.csv",
    "geometry_trial_level.csv", "geometry_alignment_nca_gain.pdf",
    "geometry_alignment_nca_gain.svg", "paired_confidence_intervals_table.tex",
    "paper_ready_results_120_180_words.txt", "analysis_audit.md",
    "analysis_audit.json", "DETAILED_CORRECTED_ANALYSIS_SUMMARY_20260826.md",
    "targeted_trial_level_corrected.csv",
    "table1_registry_version_audit.csv", "mrc_configuration_manifest.csv",
    "mrc_configuration_hits_summary.csv", "solver_revalidation_audit.json",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_main_data() -> dict[str, int]:
    legacy_audit_path = ROOT / "results" / "evaluation" / "legacy_no_entropy_solver_revalidation.json"
    require(legacy_audit_path.exists(), "legacy No-entropy solver audit is missing")
    legacy_audit = json.loads(legacy_audit_path.read_text())
    require(legacy_audit.get("historical_failed_flags") == 460,
            "unexpected legacy No-entropy failure-flag count")
    require(legacy_audit.get("all_relaxed_retries_success_exact_and_feasible") is True,
            "legacy No-entropy solver audit has unresolved rows")
    require(legacy_audit.get("max_weight_delta", 1.0) <= 1e-8,
            "legacy No-entropy revalidation changed attack weights")
    totals = {"files": 0, "trials": 0, "candidate_rows": 0}
    for model in MODELS:
        for k in KS:
            path = ROOT / "results" / "evaluation" / f"mrc_ablation_{VARIANT}_N1024_{model}_K{k}.csv"
            require(path.exists(), f"missing main result: {path}")
            d = pd.read_csv(path)
            key = "gi" if "gi" in d else "trial_id"
            require(len(d) == 3000, f"not 3000 rows: {path}")
            require(not d.duplicated([key, "target"]).any(), f"duplicate key: {path}")
            counts = d.groupby(key).size()
            require(set(map(int, counts.index)) == set(range(300)), f"trial IDs not 0..299: {path}")
            require(counts.eq(10).all(), f"not ten candidates per trial: {path}")
            require(pd.to_numeric(d.target_top1, errors="coerce").isin([0, 1]).all(),
                    f"non-binary target_top1: {path}")
            require(pd.to_numeric(d.solver_success, errors="coerce").eq(1).all(),
                    f"solver failure remains: {path}")
            effective = pd.to_numeric(d.effective_K, errors="coerce")
            require((effective + 1e-6 >= 0.6 * k).all(), f"K_eff violation: {path}")
            require(np.isclose(pd.to_numeric(d.beta).unique(), 8.0).all(), f"beta mismatch: {path}")
            require(np.isclose(pd.to_numeric(d.entropy_gamma).unique(), 0.05).all(),
                    f"entropy mismatch: {path}")
            require(np.isclose(pd.to_numeric(d.keff_frac).unique(), 0.6).all(),
                    f"K_eff fraction mismatch: {path}")
            for field in ["PESQ", "STOI"]:
                require(field in d, f"missing {field}: {path}")
                values = pd.to_numeric(d[field], errors="coerce")
                require(values.notna().all(), f"non-finite {field}: {path}")
            require(pd.to_numeric(d.PESQ).between(1.0, 5.0).all(), f"PESQ out of range: {path}")
            require(pd.to_numeric(d.STOI).between(-1.0, 1.0).all(), f"STOI out of range: {path}")
            require(d.quality_reference.eq("first_legitimate_watermarked_coalition_copy").all(),
                    f"quality reference mismatch: {path}")
            totals["files"] += 1; totals["trials"] += 300; totals["candidate_rows"] += 3000
    return totals


def verify_outputs() -> dict[str, int]:
    for name in REQUIRED:
        path = OUT / name
        require(path.exists() and path.stat().st_size > 0, f"missing/empty output: {path}")
    require((OUT / "geometry_alignment_nca_gain.pdf").stat().st_size > 1000, "PDF too small")
    require((OUT / "geometry_alignment_nca_gain.svg").stat().st_size > 1000, "SVG too small")

    ci = pd.read_csv(OUT / "paired_confidence_intervals.csv")
    required_cols = {"comparison", "registry_scope", "metric", "system", "K", "mean_A", "mean_B",
                     "paired_mean_difference", "trial_bootstrap_ci95_low", "trial_bootstrap_ci95_high",
                     "speaker_cluster_ci95_low", "speaker_cluster_ci95_high", "n_paired_trials",
                     "bootstrap_replicates", "seed"}
    require(required_cols.issubset(ci.columns), "paired CI schema incomplete")
    require(ci.bootstrap_replicates.eq(10_000).all() and ci.seed.eq(20260825).all(),
            "paired bootstrap settings mismatch")
    require((ci.trial_bootstrap_ci95_low <= ci.trial_bootstrap_ci95_high).all(), "invalid trial CI")
    require((ci.speaker_cluster_ci95_low <= ci.speaker_cluster_ci95_high).all(), "invalid cluster CI")
    allowed_metrics = {"ASR", "NCA", "mean Hits@10", "PESQ", "STOI"}
    require(set(ci.metric).issubset(allowed_metrics), f"unexpected metrics: {set(ci.metric)-allowed_metrics}")
    overall = ci[(ci.system == "ALL") & (ci.K.astype(str) == "ALL")]
    require(overall.n_paired_trials.eq(6000).all(), "overall comparisons are not 6000 paired trials")
    scopes = set(ci.registry_scope)
    require("common_N1024_corrected_nested" in scopes and "native" in scopes,
            "common/native registry scopes missing")
    targeted = pd.read_csv(OUT / "targeted_trial_level_corrected.csv")
    require(len(targeted) == 6000, "corrected targeted trial table is not 6000 rows")
    require(not targeted.duplicated(["model", "K", "trial_id"]).any(),
            "duplicate corrected targeted trial")
    require(targeted.groupby(["model", "K"]).size().eq(300).all(),
            "corrected targeted cells are not 300 trials")

    bins = pd.read_csv(OUT / "geometry_gain_by_bin.csv")
    require({"low", "medium", "high"}.issubset(set(bins.alignment_bin)), "alignment bins missing")
    require(not any("NAC" in c for c in bins.columns), "legacy NAC column remains")
    cell_bins = bins[bins.system != "ALL"]
    require(len(bins) == 48 and len(cell_bins) == 45, "geometry tertile row count mismatch")
    require(set(cell_bins.K.astype(str)) == {"3", "5", "8"}, "K=2 entered geometry bins")
    require(cell_bins.groupby(["system", "K"]).alignment_bin.nunique().eq(3).all(),
            "a system-K cell lacks three alignment bins")
    require(len(bins[bins.system == "ALL"]) == 3, "aggregate alignment bins missing")
    geometry_trials = pd.read_csv(OUT / "geometry_trial_level.csv")
    require(len(geometry_trials) == 4500, "geometry trial table is not 4,500 rows")
    require(geometry_trials.rho_wave.notna().sum() == 4465,
            "unexpected number of defined geometry correlations")
    require(set(geometry_trials.K.astype(str)) == {"3", "5", "8"},
            "geometry trial table has an invalid K")
    assoc = pd.read_csv(OUT / "geometry_gain_association.csv")
    require((assoc.seed == 20260825).all(), "association seed mismatch")
    require((assoc[assoc.analysis_scope.eq("within_system_K")].shape[0] == 15),
            "expected 15 system-K associations")
    require((assoc[assoc.analysis_scope.eq("controlled_pooled")].shape[0] == 1),
            "controlled pooled association missing")
    require(set(assoc[assoc.analysis_scope.eq("within_system_K")].K.astype(str)) == {"3", "5", "8"},
            "K=2 entered geometry association")
    perm = pd.read_csv(OUT / "geometry_permutation_control.csv")
    require(perm.permutation_replicates.eq(1000).all() and perm.seed.eq(20260825).all(),
            "permutation settings mismatch")
    require(perm[(perm.system == "ALL") & (perm.K.astype(str) == "ALL")].shape[0] == 1,
            "aggregate permutation result missing")
    require(len(perm) == 16 and set(perm[perm.system != "ALL"].K.astype(str)) == {"3", "5", "8"},
            "permutation cell coverage mismatch")

    paragraph = (OUT / "paper_ready_results_120_180_words.txt").read_text().strip()
    words = len(paragraph.replace("--", " ").split())
    require(120 <= words <= 180, f"paper-ready paragraph has {words} words")
    latex = (OUT / "paired_confidence_intervals_table.tex").read_text()
    require(latex.count(r"\\") >= 5, "LaTeX table row endings incomplete")
    public_text = "\n".join((OUT / name).read_text(errors="ignore") for name in REQUIRED
                            if name.endswith((".md", ".tex", ".txt", ".svg")))
    require(not re.search(r"\bTCT\b|any\s*@\s*10|SI[-_ ]?SDR", public_text, re.I),
            "prohibited metric/method appears in public outputs")
    require(not re.search(r"\bNAC\b", public_text), "legacy NAC terminology remains")
    solver_audit = json.loads((OUT / "solver_revalidation_audit.json").read_text())
    require(len(solver_audit["main_mrc"].get("revalidated", [])) == 1,
            "main MRC solver audit does not contain the expected one exact retry")
    require(solver_audit["legacy_no_entropy"].get(
        "all_relaxed_retries_success_exact_and_feasible") is True,
            "packaged legacy solver audit has unresolved rows")
    audit = json.loads((OUT / "analysis_audit.json").read_text())
    require(audit["bootstrap_replicates"] == 10_000 and audit["permutation_replicates"] == 1000,
            "audit replicate counts mismatch")
    config = pd.read_csv(OUT / "mrc_configuration_hits_summary.csv")
    main_config = config[config.configuration.eq("entropy005_with_keff_floor_0.6")]
    require(len(main_config) == 4 and set(main_config.K.astype(int)) == set(KS),
            "main MRC configuration summary lacks four K rows")
    require(main_config.n_trials.eq(1500).all(), "main MRC K summaries are not 1500 trials")
    require(np.isclose(main_config.solver_success_rate, 1.0).all(),
            "main MRC summary solver rate is not 100%")
    return {"paired_ci_rows": len(ci), "paragraph_words": words,
            "geometry_bin_rows": len(bins), "association_rows": len(assoc),
            "permutation_rows": len(perm)}


def main() -> None:
    report = {"main_data": verify_main_data(), "outputs": verify_outputs(), "status": "PASS"}
    path = OUT / "FINAL_VERIFICATION_REPORT.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"VERIFICATION_PASS {path}")


if __name__ == "__main__":
    main()

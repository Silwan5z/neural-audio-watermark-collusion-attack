#!/usr/bin/env python3
"""Audit registry-version and MRC-configuration conflicts without GPU work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260825
N_BOOT = 10_000
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
KS = [2, 3, 5, 8]


def rng_for(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256(str(SEED).encode())
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode())
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))


def ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(np.asarray(x, dtype=float), [0.025, 0.975]))


def trial_bootstrap(groups: list[pd.DataFrame], diff_col: str, label: str) -> tuple[float, float]:
    rng = rng_for("trial", label)
    out = np.zeros(N_BOOT)
    for g in groups:
        values = g[diff_col].to_numpy(float)
        out += values[rng.integers(0, len(values), (N_BOOT, len(values)))].mean(axis=1) / len(groups)
    return ci(out)


def speaker_bootstrap(groups: list[pd.DataFrame], diff_col: str, label: str) -> tuple[float, float]:
    """Resample speaker clusters within each system-K stratum."""
    rng = rng_for("speaker", label)
    out = np.zeros(N_BOOT)
    for g in groups:
        by = {spk: q[diff_col].to_numpy(float) for spk, q in g.groupby("spk", sort=True)}
        speakers = np.asarray(sorted(by), dtype=object)
        if len(speakers) != 100 or sorted(map(len, by.values())) != [3] * 100:
            raise ValueError(f"expected 100 speakers x 3 trials, got {len(speakers)} clusters")
        # Every speaker contributes exactly three trials, so resampling
        # clusters and averaging all sampled trials is exactly equivalent to
        # resampling the 100 speaker-level trial means.
        cluster_means = np.asarray([by[s].mean() for s in speakers], dtype=float)
        contribution = np.empty(N_BOOT)
        batch = 250
        for start in range(0, N_BOOT, batch):
            size = min(batch, N_BOOT - start)
            idx = rng.integers(0, len(cluster_means), (size, len(cluster_means)))
            contribution[start:start + size] = cluster_means[idx].mean(axis=1)
        out += contribution / len(groups)
    return ci(out)


def paired_rows(frame: pd.DataFrame, metric: str, a: str, b: str,
                diff_col: str, scope: str) -> list[dict]:
    cells = {(m, int(k)): g.sort_values("trial_id")
             for (m, k), g in frame.groupby(["model", "K"], sort=True)}
    rows = []

    def emit(system: str, k: object, groups: list[pd.DataFrame]) -> None:
        tlo, thi = trial_bootstrap(groups, diff_col, f"{scope}|{metric}|{system}|{k}")
        slo, shi = speaker_bootstrap(groups, diff_col, f"{scope}|{metric}|{system}|{k}")
        rows.append({
            "comparison": f"{a} vs {b}", "registry_scope": scope, "metric": metric,
            "system": system, "K": k,
            "mean_A": float(np.mean([g[f'{metric}_A'].mean() for g in groups])),
            "mean_B": float(np.mean([g[f'{metric}_B'].mean() for g in groups])),
            "paired_mean_difference": float(np.mean([g[diff_col].mean() for g in groups])),
            "trial_bootstrap_ci95_low": tlo, "trial_bootstrap_ci95_high": thi,
            "speaker_cluster_ci95_low": slo, "speaker_cluster_ci95_high": shi,
            "n_paired_trials": int(sum(len(g) for g in groups)),
            "n_speaker_clusters": int(sum(g.spk.nunique() for g in groups)),
            "n_strata": len(groups), "bootstrap_replicates": N_BOOT, "seed": SEED,
        })

    for (m, k), g in cells.items():
        emit(m, k, [g])
    for k in KS:
        emit("ALL", k, [cells[(m, k)] for m in MODELS])
    emit("ALL", "ALL", [cells[(m, k)] for m in MODELS for k in KS])
    return rows


def load_table1(project: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common, native, version_rows = [], [], []
    paper = pd.read_csv(project / "paper/identity_attribution_icassp_v8_source/data/blind_matched_N1024.csv")
    for model in MODELS:
        for k in KS:
            old_path = project / f"data/registry_control/registry_control_{model}_K{k}.csv"
            corrected_path = project / f"data/registry_control/registry_control_nested_{model}_K{k}.csv"
            selected_path = old_path if model == "timbrewm" else corrected_path
            old = pd.read_csv(old_path).query("N_registry == 1024")
            selected = pd.read_csv(selected_path).query("N_registry == 1024")
            if selected.duplicated(["trial_id", "method"]).any():
                raise ValueError(f"duplicates: {selected_path}")
            if selected.groupby("method").size().to_dict() != {"fwp": 300, "mean": 300}:
                raise ValueError(f"incomplete: {selected_path}")
            p = selected.pivot(index=["model", "K", "trial_id", "spk", "local_t"],
                               columns="method", values="ASR").reset_index()
            p["ASR_A"], p["ASR_B"] = p.fwp, p["mean"]
            p["diff"] = p.ASR_A - p.ASR_B
            common.append(p[["model", "K", "trial_id", "spk", "local_t", "ASR_A", "ASR_B", "diff"]])
            for method in ["mean", "fwp"]:
                got = selected[selected.method.eq(method)].ASR.mean()
                expected = paper[(paper.model.eq(model)) & (paper.K.eq(k)) & (paper.method.eq(method))].ASR_rate.iloc[0]
                if not np.isclose(got, expected):
                    raise ValueError(f"paper mismatch {model} K={k} {method}: {got} != {expected}")
                version_rows.append({
                    "system": model, "K": k, "method": method,
                    "old_ASR": old[old.method.eq(method)].ASR.mean(),
                    "corrected_ASR": got, "paper_Table_I_ASR": expected,
                    "corrected_minus_old_pp": 100 * (got - old[old.method.eq(method)].ASR.mean()),
                    "selected_source": str(selected_path),
                })

            attack = pd.read_csv(project / f"data/attack/attack_{model}_K{k}.csv")
            if "trial_id" not in attack and "gi" in attack:
                attack["trial_id"] = attack["gi"]
            if attack.duplicated(["trial_id", "method"]).any():
                raise ValueError(f"duplicates in attack {model} K={k}")
            q = attack.pivot(index=["model", "K", "trial_id", "spk", "local_t"],
                             columns="method", values=["ASR", "ACC_near_norm", "PESQ", "STOI"])
            q.columns = [f"{x}_{y}" for x, y in q.columns]
            native.append(q.reset_index())
    return pd.concat(common, ignore_index=True), pd.concat(native, ignore_index=True), pd.DataFrame(version_rows)


def audit_mrc(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "legacy_softmin_base": {
            "pattern": "data/tamper_softmin_v2_n1024/margin_reachability_v2_{m}_K{k}.csv",
            "beta": 8.0, "entropy_lambda": 0.0, "keff_floor_fraction": 0.6,
            "source": "scripts/margin_reachability_v2.py",
        },
        "no_keff_floor": {
            "pattern": "results/evaluation/mrc_ablation_no_keff_floor_N1024_{m}_K{k}.csv",
            "beta": 8.0, "entropy_lambda": 0.0, "keff_floor_fraction": np.nan,
            "source": "scripts/run_mrc_ablation.py",
        },
        "entropy005_no_floor": {
            "pattern": "results/evaluation/mrc_ablation_entropy005_N1024_{m}_K{k}.csv",
            "beta": 8.0, "entropy_lambda": 0.05, "keff_floor_fraction": np.nan,
            "source": "scripts/run_mrc_ablation.py",
        },
    }
    manifest, summaries = [], []
    for name, spec in specs.items():
        for k in KS:
            all_hits, all_eff, all_solver = [], [], []
            for model in MODELS:
                path = project / spec["pattern"].format(m=model, k=k)
                d = pd.read_csv(path)
                trial_key = "gi" if "gi" in d else "trial_id"
                if d.duplicated([trial_key, "target"]).any():
                    raise ValueError(f"duplicates: {path}")
                counts = d.groupby(trial_key).size()
                if len(counts) != 300 or not counts.eq(10).all():
                    raise ValueError(f"not 300x10: {path}")
                all_hits.extend(d.groupby(trial_key).target_top1.sum().tolist())
                all_eff.extend(pd.to_numeric(d.effective_K).tolist())
                all_solver.extend(pd.to_numeric(d.solver_success).tolist())
            summaries.append({"configuration": name, "K": k, "mean_Hits_at_10": np.mean(all_hits),
                              "n_trials": len(all_hits), "mean_effective_K": np.mean(all_eff),
                              "solver_success_rate": np.mean(all_solver)})
        manifest.append({
            "configuration": name, "beta": spec["beta"],
            "entropy_lambda": spec["entropy_lambda"],
            "keff_floor_fraction": spec["keff_floor_fraction"],
            "hard_keff_floor_enabled": bool(np.isfinite(spec["keff_floor_fraction"])),
            "selection": "softmin", "attack_weights": "target-optimized",
            "generator_source": str(project / spec["source"]),
            "file_pattern": str(project / spec["pattern"]),
        })
    missing = {
        "configuration": "entropy005_with_keff_floor_0.6",
        "beta": 8.0, "entropy_lambda": 0.05, "keff_floor_fraction": 0.6,
        "hard_keff_floor_enabled": True, "selection": "softmin",
        "attack_weights": "target-optimized", "generator_source": "NOT IMPLEMENTED/RUN",
        "file_pattern": "MISSING",
    }
    manifest.append(missing)
    return pd.DataFrame(manifest), pd.DataFrame(summaries)


def main() -> None:
    project = Path("/private/users/lym/neural-audio-watermark-collusion-attack")
    out = Path("/private/users/luojiehui/mrc_statistics_conflict_audit_20260826")
    out.mkdir(parents=True, exist_ok=True)
    common, native, versions = load_table1(project)
    rows = paired_rows(common, "ASR", "FWP", "Mean", "diff", "common_N1024_corrected_nested")
    for metric, a_col, b_col, sign in [
        ("ASR", "ASR_fwp", "ASR_mean", 1),
        ("NCA", "ACC_near_norm_fwp", "ACC_near_norm_mean", -1),
        ("PESQ", "PESQ_fwp", "PESQ_mean", 1),
        ("STOI", "STOI_fwp", "STOI_mean", 1),
    ]:
        z = native[["model", "K", "trial_id", "spk", "local_t", a_col, b_col]].copy()
        z[f"{metric}_A"], z[f"{metric}_B"] = z[a_col], z[b_col]
        z["diff"] = sign * (z[a_col] - z[b_col])
        rows += paired_rows(z, metric, "FWP", "Mean", "diff",
                            "native" if metric == "ASR" else "registry_invariant_waveform")
    ci_frame = pd.DataFrame(rows)
    manifest, hits = audit_mrc(project)
    versions.to_csv(out / "table1_registry_version_audit.csv", index=False)
    ci_frame.to_csv(out / "fwp_mean_corrected_paired_and_cluster_ci.csv", index=False)
    manifest.to_csv(out / "mrc_configuration_manifest.csv", index=False)
    hits.to_csv(out / "mrc_configuration_hits_summary.csv", index=False)
    status = {
        "table1_corrected_matches_paper_cells": 40,
        "table1_old_changed_cells": int((versions.corrected_minus_old_pp.abs() > 1e-12).sum()),
        "table1_max_abs_change_pp": float(versions.corrected_minus_old_pp.abs().max()),
        "missing_factorial_cell": "beta=8, lambda=0.05, K_eff>=0.6K",
        "seed": SEED, "bootstrap_replicates": N_BOOT,
    }
    (out / "audit_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    print("OUTPUT", out)


if __name__ == "__main__":
    main()

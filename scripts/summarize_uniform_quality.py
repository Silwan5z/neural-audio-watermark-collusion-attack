#!/usr/bin/env python3
"""Merge quality checkpoints and produce paper-table aggregates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "results" / "quality"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")


def load_model(model: str) -> pd.DataFrame:
    paths = []
    base = QUALITY / f"{model}.csv"
    if base.exists():
        paths.append(base)
    paths.extend(sorted(QUALITY.glob(f"{model}.shard*of*.csv")))
    if not paths:
        raise FileNotFoundError(model)
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    # Shards occur after a partially completed unsharded checkpoint and should
    # replace duplicate keys from that checkpoint.
    combined = combined.drop_duplicates(["model", "k", "trial_id"], keep="last")
    combined = combined.sort_values(["k", "trial_id"]).reset_index(drop=True)
    expected = {(k, trial_id) for k in (2, 3, 5, 8)
                for trial_id in range(300)}
    observed = set(zip(combined["k"].astype(int),
                       combined["trial_id"].astype(int)))
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        raise RuntimeError(f"{model}: incomplete quality records; missing={missing}")
    visqol_paths = []
    visqol_base = QUALITY / f"visqol_{model}.csv"
    if visqol_base.exists():
        visqol_paths.append(visqol_base)
    visqol_paths.extend(sorted(QUALITY.glob(f"visqol_{model}.shard*of*.csv")))
    if not visqol_paths:
        raise FileNotFoundError(visqol_base)
    visqol = pd.concat(
        [pd.read_csv(path) for path in visqol_paths], ignore_index=True)
    visqol = visqol.drop_duplicates(
        ["model", "k", "trial_id"], keep="last")
    if len(visqol) != 1200:
        raise RuntimeError(f"{model}: expected 1200 ViSQOL records")
    visqol_observed = set(zip(visqol["k"].astype(int),
                              visqol["trial_id"].astype(int)))
    if visqol_observed != expected:
        missing = sorted(expected - visqol_observed)[:10]
        raise RuntimeError(f"{model}: incomplete ViSQOL records; missing={missing}")
    if not visqol["visqol"].between(1.0, 5.0).all():
        raise RuntimeError(f"{model}: ViSQOL scores must be in [1, 5]")
    return combined.merge(
        visqol[["model", "k", "trial_id", "visqol"]],
        on=["model", "k", "trial_id"], how="left", validate="one_to_one")


def main() -> None:
    frames = [load_model(model) for model in MODELS]
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials["si_sdr_for_table"] = all_trials["published_si_sdr"].fillna(
        all_trials["recomputed_si_sdr"])
    all_trials.to_csv(QUALITY / "all_trials.csv", index=False)

    by_k = (all_trials.groupby(["model", "k"], sort=False)
            .agg(n=("trial_id", "size"),
                 pesq=("published_pesq", "mean"),
                 stoi=("published_stoi", "mean"),
                 visqol=("visqol", "mean"),
                 si_sdr=("si_sdr_for_table", "mean"),
                 snr=("snr", "mean"),
                 mean_si_sdr_reproduction_error=("si_sdr_abs_error", "mean"),
                 max_si_sdr_reproduction_error=("si_sdr_abs_error", "max"))
            .reset_index())
    by_k.to_csv(QUALITY / "summary_by_system_k.csv", index=False)

    table = (all_trials.groupby("model", sort=False)
             .agg(n=("trial_id", "size"),
                  pesq_avg=("published_pesq", "mean"),
                  stoi_avg=("published_stoi", "mean"),
                  visqol_avg=("visqol", "mean"),
                  si_sdr_avg=("si_sdr_for_table", "mean"),
                  snr_avg=("snr", "mean"))
             .reset_index())
    table.to_csv(QUALITY / "table2_quality_means.csv", index=False)

    audit = {}
    for model, group in all_trials[all_trials["k"] < 8].groupby("model"):
        errors = group["si_sdr_abs_error"].dropna()
        audit[model] = {
            "n_checked": int(len(errors)),
            "mean_abs_si_sdr_error_db": float(errors.mean()),
            "median_abs_si_sdr_error_db": float(errors.median()),
            "max_abs_si_sdr_error_db": float(errors.max()),
            "fraction_within_0_05_db": float((errors <= 0.05).mean()),
        }
    (QUALITY / "reproduction_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

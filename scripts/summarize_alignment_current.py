#!/usr/bin/env python3
"""Merge and summarize the current-protocol K=5 alignment experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
SHIFTS = (0, -10, 10, -20, 20, -50, 50)
METRICS = ("pesq", "stoi", "si_sdr", "snr")


def load_model(directory: Path, model: str) -> pd.DataFrame:
    direct = directory / f"{model}.csv"
    if direct.exists() and model != "wavmark":
        paths = [direct]
    elif model == "wavmark":
        paths = sorted(directory.glob("wavmark.shard*of3.csv"))
        paths += sorted(directory.glob("wavmark.shard*of7.csv"))
        if direct.exists():
            paths.append(direct)
    else:
        raise FileNotFoundError(direct)
    if not paths:
        raise FileNotFoundError(f"no inputs found for {model}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.drop_duplicates(["model", "trial_id", "shift_ms"], keep="last")
    frame = frame.sort_values(["trial_id", "shift_ms"]).reset_index(drop=True)
    expected = {(trial, shift) for trial in range(300) for shift in SHIFTS}
    actual = set(zip(frame.trial_id.astype(int), frame.shift_ms.astype(int)))
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise RuntimeError(
            f"{model}: expected 2100 cells, got {len(frame)}; "
            f"missing={missing}, extra={extra}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "results" / "alignment_current")
    args = parser.parse_args()
    directory = args.input_dir

    frames = []
    for model in MODELS:
        frame = load_model(directory, model)
        if model == "wavmark":
            frame.to_csv(directory / "wavmark.csv", index=False)
        frames.append(frame)
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials.to_csv(directory / "all_trials.csv", index=False)

    summary = (all_trials.groupby(["model", "shift_ms"], as_index=False)
               .agg(n=("escaped", "size"), tf_pct=("escaped", lambda x: 100*x.mean()),
                    attribution_margin=("attribution_margin", "mean"),
                    **{metric: (metric, "mean") for metric in METRICS}))
    summary.to_csv(directory / "summary_by_system_shift.csv", index=False)

    grouped_rows = []
    conditions = (("aligned", (0,)), ("pm10", (-10, 10)),
                  ("pm20", (-20, 20)), ("pm50", (-50, 50)))
    for model in MODELS:
        model_frame = all_trials[all_trials.model == model]
        for label, shifts in conditions:
            subset = model_frame[model_frame.shift_ms.isin(shifts)]
            grouped_rows.append({
                "model": model,
                "condition": label,
                "n": len(subset),
                "tf_pct": 100 * subset.escaped.mean(),
                "attribution_margin": subset.attribution_margin.mean(),
                **{metric: subset[metric].mean() for metric in METRICS},
            })
    grouped = pd.DataFrame(grouped_rows)
    grouped.to_csv(directory / "summary_direction_averaged.csv", index=False)

    cross_system = (grouped.groupby("condition", as_index=False)
                    .agg(system_count=("model", "size"),
                         tf_pct=("tf_pct", "mean"),
                         attribution_margin=("attribution_margin", "mean"),
                         **{metric: (metric, "mean") for metric in METRICS}))
    baseline = cross_system[cross_system.condition == "aligned"].iloc[0]
    cross_system["tf_delta_pp_vs_aligned"] = cross_system.tf_pct - baseline.tf_pct
    for metric in METRICS:
        cross_system[f"{metric}_delta_vs_aligned"] = (
            cross_system[metric] - baseline[metric])
    cross_system.to_csv(directory / "summary_cross_system.csv", index=False)

    print(grouped.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCross-system means:")
    print(cross_system.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()

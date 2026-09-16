#!/usr/bin/env python3
"""Validate and summarize the K=5 independent-codec stress test."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "codec_stress_test"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
CODECS = ("none", "mp3_128k", "opus_64k")


def load_model(model: str) -> pd.DataFrame:
    paths = []
    base = RESULTS / f"{model}.csv"
    if base.exists():
        paths.append(base)
    paths.extend(sorted(RESULTS.glob(f"{model}.shard*of*.csv")))
    if not paths:
        raise FileNotFoundError(model)
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.drop_duplicates(["model", "trial_id", "codec"], keep="last")
    expected = {(trial, codec) for trial in range(300) for codec in CODECS}
    observed = set(zip(frame["trial_id"].astype(int), frame["codec"]))
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        raise RuntimeError(f"{model}: incomplete codec records; missing={missing}")
    if len(frame) != 900 or set(frame["model"]) != {model}:
        raise RuntimeError(f"{model}: expected 900 rows")
    if frame["source_path"].nunique() != 300:
        raise RuntimeError(f"{model}: expected 300 distinct recordings")
    return frame


def main() -> None:
    frames = [load_model(model) for model in MODELS]
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials = all_trials.sort_values(
        ["model", "trial_id", "codec"]).reset_index(drop=True)
    all_trials.to_csv(RESULTS / "all_trials.csv", index=False)

    by_system = (all_trials.groupby(["model", "codec"], sort=False)
                 .agg(n=("trial_id", "size"),
                      tf_pct=("escaped", lambda values: 100 * values.mean()),
                      attribution_margin=("attribution_margin", "mean"),
                      pesq=("pesq", "mean"),
                      stoi=("stoi", "mean"),
                      si_sdr=("si_sdr", "mean"),
                      snr=("snr", "mean"))
                 .reset_index())
    by_system.to_csv(RESULTS / "summary_by_system_codec.csv", index=False)

    cross_system = (by_system.groupby("codec", sort=False)
                    .agg(system_count=("model", "size"),
                         tf_pct=("tf_pct", "mean"),
                         attribution_margin=("attribution_margin", "mean"),
                         pesq=("pesq", "mean"),
                         stoi=("stoi", "mean"),
                         si_sdr=("si_sdr", "mean"),
                         snr=("snr", "mean"))
                    .reset_index())
    baseline = cross_system[cross_system["codec"] == "none"].iloc[0]
    for metric in ("tf_pct", "pesq", "stoi", "si_sdr", "snr"):
        cross_system[f"{metric}_delta_vs_none"] = (
            cross_system[metric] - baseline[metric])
    cross_system.to_csv(RESULTS / "summary_cross_system.csv", index=False)
    print(by_system.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(cross_system.to_string(index=False,
                                 float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

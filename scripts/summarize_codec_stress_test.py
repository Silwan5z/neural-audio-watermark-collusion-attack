#!/usr/bin/env python3
"""Validate and summarize the K=5 independent-codec stress test."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "codec_stress_test"
MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
CODECS = ("none", "mp3_128k", "opus_64k")


def load_model(model: str, directory: Path) -> pd.DataFrame:
    paths = []
    base = directory / f"{model}.csv"
    if base.exists():
        paths.append(base)
    paths.extend(sorted(directory.glob(f"{model}.shard*of*.csv")))
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
    required = {"valid_post_codec_copy_count", "all_post_codec_copies_valid"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{model}: missing post-codec single-copy controls")
    if frame["source_path"].nunique() != 300:
        raise RuntimeError(f"{model}: expected 300 distinct recordings")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_model(model, args.input_dir) for model in MODELS]
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials = all_trials.sort_values(
        ["model", "trial_id", "codec"]).reset_index(drop=True)
    all_trials.to_csv(args.output_dir / "all_trials.csv", index=False)

    by_system = (all_trials.groupby(["model", "codec"], sort=False)
                 .agg(n=("trial_id", "size"),
                      valid_post_codec_copy_pct=(
                          "valid_post_codec_copy_count",
                          lambda values: 100 * values.sum() / (5 * len(values))),
                      all_post_codec_copies_valid_pct=(
                          "all_post_codec_copies_valid",
                          lambda values: 100 * values.mean()),
                      tf_pct=("escaped", lambda values: 100 * values.mean()),
                      attribution_margin=("attribution_margin", "mean"),
                      pesq=("pesq", "mean"),
                      stoi=("stoi", "mean"),
                      si_sdr=("si_sdr", "mean"),
                      snr=("snr", "mean"))
                 .reset_index())
    conditional = (all_trials[all_trials["all_post_codec_copies_valid"] == 1]
                   .groupby(["model", "codec"], sort=False)
                   .agg(valid_subset_n=("trial_id", "size"),
                        valid_subset_tf_pct=(
                            "escaped", lambda values: 100 * values.mean()))
                   .reset_index())
    by_system = by_system.merge(
        conditional, on=["model", "codec"], how="left")
    by_system["valid_subset_n"] = by_system["valid_subset_n"].fillna(0).astype(int)
    by_system.to_csv(
        args.output_dir / "summary_by_system_codec.csv", index=False)

    cross_system = (by_system.groupby("codec", sort=False)
                    .agg(system_count=("model", "size"),
                         valid_post_codec_copy_pct=(
                             "valid_post_codec_copy_pct", "mean"),
                         all_post_codec_copies_valid_pct=(
                             "all_post_codec_copies_valid_pct", "mean"),
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
    cross_system.to_csv(
        args.output_dir / "summary_cross_system.csv", index=False)
    print(by_system.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(cross_system.to_string(index=False,
                                 float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

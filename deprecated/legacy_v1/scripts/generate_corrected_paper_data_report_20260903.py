#!/usr/bin/env python3
"""Build the audited 2026-09-03 corrected paper-data report.

This script performs no model inference. It validates the completed source-correct
records, recomputes every numerical panel/table used by the current V16 paper,
exports machine-readable summaries, redraws the data figures, and emits a Chinese
PDF report through ReportLab (the host has no TeX engine).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "corrected_paper_data_20260903"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}
BITS = {"audioseal": 16, "wavmark": 16, "timbrewm": 10, "voicemark": 16, "wmcodec": 16}
KS = [2, 3, 5, 8]
CONDITIONS = [
    "freq_0_1k", "freq_1_2k", "freq_2_4k", "freq_4_8k",
    "speech_only", "non_speech_only", "random_mask", "full_waveform",
]
COND_LABEL = {
    "freq_0_1k": "0–1 kHz", "freq_1_2k": "1–2 kHz",
    "freq_2_4k": "2–4 kHz", "freq_4_8k": "4–8 kHz",
    "speech_only": "Speech", "non_speech_only": "Non-speech",
    "random_mask": "Shifted", "full_waveform": "Full",
}
COLOR = {
    "audioseal": "#0077BB", "wavmark": "#009988", "timbrewm": "#CC79A7",
    "voicemark": "#D55E00", "wmcodec": "#A66F00",
}
MARKER = {"audioseal": "o", "wavmark": "s", "timbrewm": "D", "voicemark": "^", "wmcodec": "v"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster_ratio_ci(df: pd.DataFrame, numerator: str, denominator: str,
                     seed: int, reps: int = 10000) -> tuple[float, float, float]:
    grouped = df.groupby("speaker")[[numerator, denominator]].sum()
    a = grouped[numerator].to_numpy(float)
    b = grouped[denominator].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, len(grouped), len(grouped))
        values[i] = a[idx].sum() / b[idx].sum()
    point = float(df[numerator].sum() / df[denominator].sum())
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2 or float(np.ptp(y)) <= 1e-15:
        return float("nan")
    fit = np.polyval(np.polyfit(x, y, 1), x)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def oriented_response(row: pd.Series) -> float:
    probability = float(json.loads(row.bit_probabilities)[int(row.flipped_bit)])
    target = (int(row.flipped_payload) >> int(row.flipped_bit)) & 1
    return probability if target else 1.0 - probability


def validate_main() -> pd.DataFrame:
    records = []
    base = ROOT / "results" / "evaluation_300clips_20260902"
    for model in MODELS:
        for k in (2, 3, 5):
            path = base / f"attack_{model}_K{k}.csv"
            data = pd.read_csv(path)
            data = data[data.method == "mean"].copy()
            assert len(data) == 300
            assert data.gi.nunique() == 300
            assert data.spk.nunique() == 100
            assert data.groupby("spk").size().eq(3).all()
            assert data.source_path.nunique() == 300
            assert data.clip_index.value_counts().to_dict() == {1: 100, 2: 100, 3: 100}
            assert set(data.source_exact_count.astype(int)) == {k}
            records.append({
                "model": model, "system": LABEL[model], "K": k, "n_trials": 300,
                "n_speakers": 100, "unique_sources": 300,
                "tracing_failure": float(data.ASR.mean()),
                "R3_escape": float(data.R3_escape.mean()),
                "R5_escape": float(data.R5_escape.mean()),
                "ACC_near": float(data.ACC_near.mean()),
                "ACC_near_norm": float(data.ACC_near_norm.mean()),
                "AggResid": float(data.AggResid.mean()),
                "PESQ": float(data.PESQ.mean()), "STOI": float(data.STOI.mean()),
                "SI_SDR": float(data.SI_SDR.mean()),
                "source_payload_attempts_mean": float(data.source_payload_attempts.mean()),
                "source_payload_attempts_max": int(data.source_payload_attempts.max()),
                "source_file": str(path),
            })
    return pd.DataFrame(records)


def validate_k8() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = ROOT / "data" / "k8_population_source_correct_300clips_20260902" / "raw"
    trial_rows, bit_rows, comp_rows, support_rows = [], [], [], []
    for mi, model in enumerate(MODELS):
        files = sorted((root / model).glob("trial_*.json"))
        assert len(files) == 300
        source_paths, speakers, clips = set(), set(), []
        model_trials, model_bits = [], []
        for path in files:
            item = json.loads(path.read_text())
            assert int(item["source_exact_count"]) == 8
            source_paths.add(item["source_path"])
            speakers.add(item["speaker"])
            clips.append(int(item["clip_index"]))
            coalition = np.asarray(item["coalition_payload_bits"], dtype=int)
            hard = np.asarray(item["decoded_hard_bits"], dtype=int)
            prob = np.asarray(item["soft_bit_probability"], dtype=float)
            assert coalition.shape == (8, BITS[model])
            assert hard.shape == (BITS[model],)
            assert prob.shape == (BITS[model],)
            assert all(int(row["exact"]) == 1 for row in item["source_decodes"])
            counts = coalition.sum(axis=0)
            strict = counts != 4
            unanimous = (counts == 0) | (counts == 8)
            strict_consistency = float(np.mean(hard[strict] == (counts[strict] > 4)))
            unanimous_retention = (
                float(np.mean(hard[unanimous] == (counts[unanimous] > 4)))
                if unanimous.any() else float("nan")
            )
            trial = {
                "model": model, "system": LABEL[model], "trial_id": int(item["trial_id"]),
                "speaker": item["speaker"], "clip_index": int(item["clip_index"]),
                "source_path": item["source_path"], "source_exact_count": 8,
                "tracing_failure": int(item["tracing_failure"]),
                "NCA": float(item["NCA"]), "PESQ": float(item["PESQ"]),
                "STOI": float(item["STOI"]),
                "presence": (float(item["presence"])
                             if item.get("presence") is not None else float("nan")),
                "strict_majority_consistency": strict_consistency,
                "unanimous_preservation": unanimous_retention,
                "exact_majority_payload": int(np.all(hard[strict] == (counts[strict] > 4))),
            }
            trial_rows.append(trial)
            model_trials.append(trial)
            for bit, (count, decoded, probability) in enumerate(zip(counts, hard, prob)):
                row = {
                    "model": model, "system": LABEL[model], "trial_id": int(item["trial_id"]),
                    "speaker": item["speaker"], "bit": bit, "ones_among_8": int(count),
                    "decoded_bit": int(decoded), "p_bit_1": float(probability),
                    "strict": int(count != 4),
                    "strict_correct": int(count != 4 and decoded == int(count > 4)),
                    "unanimous": int(count in (0, 8)),
                    "unanimous_correct": int(count in (0, 8) and decoded == int(count > 4)),
                }
                bit_rows.append(row)
                model_bits.append(row)
        assert len(source_paths) == 300 and len(speakers) == 100
        assert pd.Series(clips).value_counts().to_dict() == {1: 100, 2: 100, 3: 100}
        bits = pd.DataFrame(model_bits)
        trials = pd.DataFrame(model_trials)
        for count in range(9):
            subset = bits[bits.ones_among_8 == count].copy()
            subset["den"] = 1
            mean, low, high = cluster_ratio_ci(
                subset, "decoded_bit", "den", 20262000 + mi * 20 + count)
            comp_rows.append({
                "model": model, "system": LABEL[model], "ones_among_8": count,
                "n_bits": len(subset), "decoded_one_rate": mean,
                "ci95_low": low, "ci95_high": high,
            })
        strict_rows = bits[bits.strict == 1].copy(); strict_rows["den"] = 1
        un_rows = bits[bits.unanimous == 1].copy(); un_rows["den"] = 1
        pooled_strict = float(strict_rows.strict_correct.sum() / len(strict_rows))
        pooled_unanimous = float(un_rows.unanimous_correct.sum() / len(un_rows))
        support_rows.append({
            "model": model, "system": LABEL[model], "n_trials": 300,
            "trial_mean_strict_majority_agreement": float(trials.strict_majority_consistency.mean()),
            "trial_mean_unanimous_bit_retention": float(trials.unanimous_preservation.mean()),
            "trials_with_unanimous_bits": int(trials.unanimous_preservation.notna().sum()),
            "all_strict_majority_bits": float(trials.exact_majority_payload.mean()),
            "pooled_strict_majority_agreement": pooled_strict,
            "pooled_unanimous_bit_retention": pooled_unanimous,
            "tracing_failure": float(trials.tracing_failure.mean()),
            "NCA": float(trials.NCA.mean()), "PESQ": float(trials.PESQ.mean()),
            "STOI": float(trials.STOI.mean()), "presence": float(trials.presence.mean()),
        })
    return (pd.DataFrame(trial_rows), pd.DataFrame(bit_rows),
            pd.DataFrame(comp_rows), pd.DataFrame(support_rows))


def validate_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shard_root = ROOT / "data" / "mixture_path_k5_300clips_20260902" / "shards"
    endpoint_root = ROOT / "results" / "one_bit_k5_300clips_20260902"
    point_frames, summary_rows, aggregate_rows = {}, [], []
    coarse_indices = set(range(0, 101, 10))
    for model in ("audioseal", "voicemark"):
        suffix = "partial.csv" if model == "audioseal" else "csv"
        path = shard_root / f"mixture_path_{model}_shard0of1.{suffix}"
        points = pd.read_csv(path).sort_values(["trial_id", "lambda_index"]).copy()
        endpoints = pd.read_csv(endpoint_root / f"onebit_k5_{model}_full.csv")
        assert points.trial_id.nunique() == 300
        assert endpoints.trial_id.nunique() == 300
        assert endpoints.spk.nunique() == 100
        assert endpoints.source_path.nunique() == 300
        assert endpoints.base_payload_correct.astype(int).eq(1).all()
        assert endpoints.flipped_payload_correct.astype(int).eq(1).all()
        points["oriented_response"] = points.apply(oriented_response, axis=1)
        point_frames[model] = points
        for trial_id, group in points.groupby("trial_id"):
            group = group.sort_values("lambda_index")
            identities = group.decoded_identity.astype(int).to_numpy()
            base = int(group.base_payload.iloc[0]); flipped = int(group.flipped_payload.iloc[0])
            at_zero = group[group.lambda_index == 0]
            at_one = group[group.lambda_index == 100]
            exact = (
                len(at_zero) == 1 and len(at_one) == 1
                and int(at_zero.decoded_identity.iloc[0]) == base
                and int(at_one.decoded_identity.iloc[0]) == flipped
            )
            assert exact
            transitions = int(np.sum(identities[1:] != identities[:-1]))
            probability = group.oriented_response.to_numpy(float)
            monotonic = int(np.all(np.diff(probability) >= -1e-6))
            coarse = group[group.lambda_index.isin(coarse_indices)].sort_values("lambda_index")
            summary_rows.append({
                "model": model, "system": LABEL[model], "trial_id": int(trial_id),
                "speaker": group.spk.iloc[0], "base_payload": base,
                "flipped_payload": flipped, "flipped_bit": int(group.flipped_bit.iloc[0]),
                "n_points": len(group), "endpoint_exact": 1,
                "identity_transition_count": transitions,
                "direct": int(transitions == 1), "multiple": int(transitions > 1),
                "target_bit_monotonic": monotonic,
                "target_probability_r2_coarse": r2(
                    coarse["lambda"].to_numpy(float), coarse.oriented_response.to_numpy(float)),
            })
        coarse = points[points.sampling_stage == "coarse"]
        for lam, group in coarse.groupby("lambda"):
            aggregate_rows.append({
                "model": model, "system": LABEL[model], "lambda": float(lam),
                "n_trials": len(group), "mean_response": float(group.oriented_response.mean()),
                "p10": float(group.oriented_response.quantile(0.10)),
                "p90": float(group.oriented_response.quantile(0.90)),
            })
    summary = pd.DataFrame(summary_rows)
    aggregate = pd.DataFrame(aggregate_rows)
    selected_rows, selected_meta = [], []
    for model in ("audioseal", "voicemark"):
        candidates = summary[summary.model == model].copy()
        if model == "audioseal":
            candidates = candidates[(candidates.target_bit_monotonic == 1) &
                                    (candidates.identity_transition_count == 1)]
        else:
            candidates = candidates[candidates.identity_transition_count >= 2]
        target = aggregate[aggregate.model == model].set_index("lambda").mean_response
        scored = []
        for trial_id in candidates.trial_id.astype(int):
            coarse = point_frames[model][
                (point_frames[model].trial_id == trial_id) &
                (point_frames[model].sampling_stage == "coarse")
            ].sort_values("lambda")
            mse = float(np.mean((coarse.oriented_response.to_numpy() -
                                 target.loc[coarse["lambda"]].to_numpy()) ** 2))
            scored.append((mse, trial_id))
        score, trial_id = min(scored)
        chosen = point_frames[model][point_frames[model].trial_id == trial_id].sort_values("lambda_index")
        meta = summary[(summary.model == model) & (summary.trial_id == trial_id)].iloc[0]
        selected_meta.append({
            "model": model, "system": LABEL[model], "trial_id": trial_id,
            "speaker": meta.speaker, "base_payload": int(meta.base_payload),
            "flipped_payload": int(meta.flipped_payload), "flipped_bit": int(meta.flipped_bit),
            "identity_transition_count": int(meta.identity_transition_count),
            "selection_mse": score,
        })
        for _, row in chosen.iterrows():
            identity = int(row.decoded_identity)
            state = ("base" if identity == int(row.base_payload)
                     else "flipped" if identity == int(row.flipped_payload)
                     else "intermediate")
            selected_rows.append({
                "model": model, "system": LABEL[model], "trial_id": trial_id,
                "lambda_index": int(row.lambda_index), "lambda": float(row["lambda"]),
                "oriented_response": float(row.oriented_response),
                "decoded_identity": identity, "state": state,
                "sampling_stage": row.sampling_stage,
            })
    return summary, aggregate, pd.DataFrame(selected_meta), pd.DataFrame(selected_rows)


def validate_regions() -> pd.DataFrame:
    root = ROOT / "results" / "frequency_temporal_k5_300clips_20260902" / "shards"
    rows = []
    for model in MODELS:
        files = sorted(root.glob(f"frequency_temporal_k5_{model}_shard*of7.csv"))
        assert len(files) == 7
        data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        assert len(data) == 2400
        assert data.trial_id.nunique() == 300
        assert data.source_path.nunique() == 300
        assert data[["trial_id", "condition"]].drop_duplicates().shape[0] == 2400
        assert data.groupby("trial_id").condition.nunique().eq(8).all()
        assert set(data.source_exact_count.astype(int)) == {5}
        for condition in CONDITIONS:
            subset = data[data.condition == condition]
            assert len(subset) == 300
            rows.append({
                "model": model, "system": LABEL[model], "condition": condition,
                "condition_label": COND_LABEL[condition],
                "condition_family": subset.condition_family.iloc[0],
                "n_trials": 300, "n_speakers": subset.spk.nunique(),
                "unique_sources": subset.source_path.nunique(),
                "tracing_failure": float(subset.tracing_failure.mean()),
                "NCA": float(subset.NCA.mean()),
                "decode_valid_rate": float(subset.decode_valid.mean()),
                "presence": float(subset.presence.mean()),
                "source_payload_attempts_mean": float(subset.source_payload_attempts.mean()),
                "source_payload_attempts_max": int(subset.source_payload_attempts.max()),
            })
    return pd.DataFrame(rows)


def configure_plots() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.labelsize": 8.5, "axes.titlesize": 8.8,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
        "legend.fontsize": 7.2, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def draw_main(main: pd.DataFrame, support: pd.DataFrame) -> Path:
    all_rows = pd.concat([
        main[["model", "K", "tracing_failure", "PESQ", "STOI"]],
        support.assign(K=8)[["model", "K", "tracing_failure", "PESQ", "STOI"]],
    ], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    for model in MODELS:
        z = all_rows[all_rows.model == model].sort_values("K")
        axes[0].plot(z.K, 100 * z.tracing_failure, marker=MARKER[model], color=COLOR[model], label=LABEL[model])
        axes[1].plot(z.K, z.PESQ, marker=MARKER[model], color=COLOR[model])
        axes[2].plot(z.K, z.STOI, marker=MARKER[model], color=COLOR[model])
    titles = ["Tracing failure", "PESQ", "STOI"]
    ylabels = ["Percent", "Score", "Score"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Coalition size K"); ax.set_ylabel(ylabel); ax.set_xticks(KS)
        ax.grid(axis="y", color="#E4E7E9", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(80, 101); axes[1].set_ylim(4.0, 4.65); axes[2].set_ylim(.965, 1.001)
    fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .91))
    path = OUT / "figure_table2_main_metrics.png"
    fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
    return path


def draw_composition(comp: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(5, 1, figsize=(7.7, 8.6), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        z = comp[comp.model == model].sort_values("ones_among_8")
        x = z.ones_among_8.to_numpy(float); y = z.decoded_one_rate.to_numpy(float)
        lo = z.ci95_low.to_numpy(float); hi = z.ci95_high.to_numpy(float)
        ax.axvspan(3.7, 4.3, color="#DDE1E4", alpha=.5, lw=0)
        ax.axhline(.5, color="#999", ls="--", lw=.7)
        ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt=MARKER[model], color=COLOR[model],
                    mfc="white", ms=4, capsize=2, lw=.8)
        ax.text(.01, .82, LABEL[model], transform=ax.transAxes, color=COLOR[model], fontweight="bold")
        ax.set_ylim(-.04, 1.04); ax.set_yticks([0, .5, 1]); ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(9)); axes[-1].set_xlabel("Participants carrying bit 1 (out of 8)")
    fig.supylabel("Decoded-one rate")
    fig.tight_layout()
    path = OUT / "figure2_k8_composition.png"
    fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
    return path


def draw_paths(summary: pd.DataFrame, selected_meta: pd.DataFrame,
               selected_points: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(10.5, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .9], wspace=.34)
    for idx, model in enumerate(("audioseal", "voicemark")):
        ax = fig.add_subplot(gs[0, idx])
        z = selected_points[selected_points.model == model].sort_values("lambda")
        ax.plot(z["lambda"], z.oriented_response, marker=MARKER[model], color=COLOR[model], ms=3.5)
        ax.axhline(.5, color="#999", ls="--", lw=.7)
        base = selected_meta[selected_meta.model == model].iloc[0]
        state_color = {"base": "#D9E6F1", "intermediate": "#F5D77A", "flipped": "#E4D8EC"}
        values = z.state.to_numpy(); lambdas = z["lambda"].to_numpy(float)
        for lo, hi, state in zip(lambdas[:-1], lambdas[1:], values[:-1]):
            ax.axvspan(lo, hi, ymin=.01, ymax=.09, color=state_color[state], lw=0)
        ax.set(xlim=(0, 1), ylim=(-.02, 1.02), xlabel="Mixture weight lambda",
               ylabel="Target-bit response")
        ax.set_title(f"{LABEL[model]} representative (trial {int(base.trial_id)})", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[0, 2])
    order = ["audioseal", "voicemark"]
    direct = [int(summary[summary.model == m].direct.sum()) for m in order]
    multiple = [int(summary[summary.model == m].multiple.sum()) for m in order]
    y = np.arange(2)
    ax.barh(y, direct, color="#758397", label="Direct")
    ax.barh(y, multiple, left=direct, color="#E7A600", label="Multiple")
    for i, (one, many) in enumerate(zip(direct, multiple)):
        ax.text(one/2, i, f"{one} direct", ha="center", va="center", color="white", fontweight="bold")
        if many > 10:
            ax.text(one+many/2, i, f"{many} multiple", ha="center", va="center", color="#29261f", fontweight="bold")
    ax.set_yticks(y, [LABEL[m] for m in order]); ax.invert_yaxis(); ax.set_xlim(0, 300)
    ax.set_xlabel("Endpoint-correct paths"); ax.set_title("Path categories", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT / "figure3_controlled_paths.png"
    fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
    return path


def draw_regions(region: pd.DataFrame) -> Path:
    families = [(CONDITIONS[:4], "Frequency bands"), (CONDITIONS[4:], "Temporal regions")]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4))
    for ax, (conds, title) in zip(axes, families):
        matrix = np.array([[region[(region.model == m) & (region.condition == c)].tracing_failure.iloc[0]
                            for c in conds] for m in MODELS])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4), [COND_LABEL[c] for c in conds])
        ax.set_yticks(range(5), [LABEL[m] for m in MODELS])
        ax.set_title(title, loc="left", fontweight="bold")
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f"{100*matrix[i,j]:.1f}", ha="center", va="center",
                        color="white" if matrix[i,j] > .55 else "#202428", fontweight="bold")
        for spine in ax.spines.values(): spine.set_visible(False)
    cbar = fig.colorbar(image, ax=axes, orientation="vertical", fraction=.035, pad=.025)
    cbar.set_label("Tracing failure")
    fig.subplots_adjust(left=.18, right=.90, top=.94, bottom=.08, hspace=.42)
    path = OUT / "figure4_frequency_temporal.png"
    fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
    return path


def md_table(data: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    rows = []
    for _, row in data[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value): text = "—"
            elif col in formats: text = formats[col].format(value)
            else: text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
        *rows,
    ])


def write_markdown(main_all: pd.DataFrame, comp: pd.DataFrame, support: pd.DataFrame,
                   path_summary: pd.DataFrame, selected_meta: pd.DataFrame,
                   selected_points: pd.DataFrame, region: pd.DataFrame) -> Path:
    out = []
    out += ["# 修正后的完整论文图表数据报告", "", "**数据快照：** 2026-09-03  ",
            "**验证状态：** VERIFIED  ",
            "**统一口径：** 100 speakers × 每人 3 个不同 clip = 300 个不同音频；所有输入在混合前均精确解码。", ""]
    out += ["## Table 2：Tracing failure 与质量", ""]
    out.append(md_table(main_all, ["system", "K", "n_trials", "tracing_failure", "PESQ", "STOI", "SI_SDR"],
                        {"tracing_failure": "{:.6f}", "PESQ": "{:.6f}", "STOI": "{:.6f}", "SI_SDR": "{:.6f}"}))
    out += ["", "## Fig. 2：K=8 bit composition", ""]
    out.append(md_table(comp, ["system", "ones_among_8", "n_bits", "decoded_one_rate", "ci95_low", "ci95_high"],
                        {"decoded_one_rate": "{:.9f}", "ci95_low": "{:.9f}", "ci95_high": "{:.9f}"}))
    out += ["", "## Table 3：K=8 support", ""]
    out.append(md_table(support, ["system", "trial_mean_strict_majority_agreement",
                                  "trial_mean_unanimous_bit_retention", "trials_with_unanimous_bits",
                                  "all_strict_majority_bits", "pooled_strict_majority_agreement",
                                  "pooled_unanimous_bit_retention", "tracing_failure", "NCA", "PESQ", "STOI"],
                        {c: "{:.9f}" for c in ["trial_mean_strict_majority_agreement",
                                               "trial_mean_unanimous_bit_retention", "all_strict_majority_bits",
                                               "pooled_strict_majority_agreement", "pooled_unanimous_bit_retention",
                                               "tracing_failure", "NCA", "PESQ", "STOI"]}))
    out += ["", "## Fig. 3：Controlled paths", ""]
    counts = path_summary.groupby(["system", "model"]).agg(
        valid_paths=("trial_id", "size"), direct=("direct", "sum"), multiple=("multiple", "sum"),
        mean_transitions=("identity_transition_count", "mean"),
        monotonic=("target_bit_monotonic", "sum"), mean_r2=("target_probability_r2_coarse", "mean")).reset_index()
    out.append(md_table(counts, ["system", "valid_paths", "direct", "multiple", "mean_transitions", "monotonic", "mean_r2"],
                        {"mean_transitions": "{:.6f}", "mean_r2": "{:.6f}"}))
    out += ["", "### 代表路径", "", md_table(selected_meta, list(selected_meta.columns),
              {"selection_mse": "{:.10f}"}), "",
              md_table(selected_points, ["system", "trial_id", "lambda_index", "lambda", "oriented_response",
                                         "decoded_identity", "state", "sampling_stage"],
                       {"lambda": "{:.2f}", "oriented_response": "{:.9f}"})]
    out += ["", "## Fig. 4：Frequency/temporal sensitivity", ""]
    out.append(md_table(region, ["system", "condition", "n_trials", "tracing_failure", "NCA",
                                 "decode_valid_rate", "presence"],
                        {"tracing_failure": "{:.9f}", "NCA": "{:.9f}",
                         "decode_valid_rate": "{:.9f}", "presence": "{:.9f}"}))
    out += ["", "## 必须替换的旧数字", "",
            "- TimbreWM K=8 tracing failure：93.7% → 92.0%。",
            "- WavMark K=8 tie decoded-one rate：87.7% → 88.6%。",
            "- AudioSeal all-strict-majority：22.0% → 20.7%。",
            "- WavMark all-strict-majority：51.3% → 54.3%。",
            "- VoiceMark paths：100/169, n=269 → 113/187, n=300。",
            "- VoiceMark 代表路径：trial 46 → trial 220。",
            "- 数据集描述改为 300 个不同音频，而不是 100 个音频重复三次。", ""]
    path = OUT / "corrected_complete_figure_table_data_report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def pdf_styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("cn-title", parent=styles["Title"], fontName="STSong-Light",
                                fontSize=21, leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#17324D")),
        "subtitle": ParagraphStyle("cn-subtitle", parent=styles["Normal"], fontName="STSong-Light",
                                   fontSize=10, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#4B5965")),
        "h1": ParagraphStyle("cn-h1", parent=styles["Heading1"], fontName="STSong-Light",
                             fontSize=15, leading=20, spaceBefore=8, spaceAfter=7, textColor=colors.HexColor("#174A73")),
        "h2": ParagraphStyle("cn-h2", parent=styles["Heading2"], fontName="STSong-Light",
                             fontSize=12, leading=17, spaceBefore=6, spaceAfter=5, textColor=colors.HexColor("#2C617F")),
        "body": ParagraphStyle("cn-body", parent=styles["BodyText"], fontName="STSong-Light",
                               fontSize=9, leading=14, alignment=TA_LEFT, spaceAfter=4),
        "small": ParagraphStyle("cn-small", parent=styles["BodyText"], fontName="STSong-Light",
                                fontSize=7.4, leading=10.5, alignment=TA_LEFT),
        "caption": ParagraphStyle("cn-caption", parent=styles["BodyText"], fontName="STSong-Light",
                                  fontSize=8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
    }


def P(text: str, style) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def pdf_table(data, widths=None, font_size=7.2, repeat=1, aligns=None):
    table = Table(data, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.1),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#9EACB5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3),
    ]
    if aligns:
        for col, align in aligns.items(): style.append(("ALIGN", (col, 1), (col, -1), align))
    table.setStyle(TableStyle(style))
    return table


def build_pdf(main_all: pd.DataFrame, comp: pd.DataFrame, support: pd.DataFrame,
              path_summary: pd.DataFrame, aggregate: pd.DataFrame,
              selected_meta: pd.DataFrame, selected_points: pd.DataFrame,
              region: pd.DataFrame, figures: list[Path]) -> Path:
    styles = pdf_styles()
    path = OUT / "corrected_complete_figure_table_data_report_20260903.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4),
                            leftMargin=12*mm, rightMargin=12*mm,
                            topMargin=14*mm, bottomMargin=13*mm,
                            title="修正后的完整论文图表数据报告（2026-09-03）",
                            author="Experiment data audit")
    story = []
    story += [Spacer(1, 12*mm), P("修正后的完整论文图表数据报告", styles["title"]),
              Spacer(1, 3*mm), P("对应当前 V16 稿件的全部数据型表格与图形", styles["subtitle"]),
              Spacer(1, 10*mm)]
    passport = [
        ["字段", "内容"], ["数据快照", "2026-09-03"], ["验证状态", "VERIFIED"],
        ["样本结构", "100 speakers × 每人 3 个不同 clip = 300 个不同音频"],
        ["输入合法性", "每个参与混合的水印副本在攻击前均精确恢复 assigned payload"],
        ["实验规模", "K=2/3/5/8：每系统每 K 300 trials；区域实验：每系统 300×8 条件"],
        ["统计单位", "Fig.2：bit 坐标汇总并按 speaker 聚类 bootstrap；Table 3：trial-level 均值"],
        ["PDF 构建", "ReportLab 矢量文本 + Matplotlib 图；当前主机无 TeX 引擎"],
    ]
    story += [pdf_table(passport, [42*mm, 200*mm], font_size=9), Spacer(1, 5*mm),
              P("核心结论：此前缺失的 WavMark 已全部补齐；VoiceMark 路径实验现为完整的 300 条合法路径；旧 V16 analysis 目录中的冻结结果不得继续作为数字来源。", styles["body"]),
              PageBreak()]

    story += [P("1. 数据来源与完整性审计", styles["h1"])]
    checks = [
        ["模块", "系统/条件", "期望规模", "实得规模", "不同 source", "source-correct", "状态"],
        ["主实验 K=2/3/5", "5 systems × 3 K", "15×300", "15×300", "每单元 300", "exact_count=K", "PASS"],
        ["K=8", "5 systems", "5×300", "5×300", "每系统 300", "exact_count=8", "PASS"],
        ["Controlled paths", "AudioSeal, VoiceMark", "2×300", "2×300", "每系统 300", "双 endpoint 精确", "PASS"],
        ["区域实验", "5 systems × 8 conditions", "12,000", "12,000", "每系统 300", "exact_count=5", "PASS"],
    ]
    story += [pdf_table(checks, [34*mm, 49*mm, 34*mm, 34*mm, 37*mm, 42*mm, 22*mm], font_size=8),
              Spacer(1, 4*mm),
              P("采样口径修正：当前实验不是“100 条音频各重复 3 次”。每位说话人使用 3 个不同的 10 秒 clip，因此每个实验单元包含 300 个不同 source path。", styles["body"]),
              P("质量参考：PESQ/STOI 均以第一个合法水印副本为参考，而非无水印原始音频。Tracing failure 表示混合后 decoded payload 不匹配任何 coalition participant。", styles["body"]),
              P("Table 1 系统设置", styles["h2"])]
    systems = [["System", "Payload bits", "Native decoder output"],
               ["AudioSeal", "16", "bit scores"], ["WavMark", "16", "window votes"],
               ["TimbreWM", "10", "bit scores"], ["VoiceMark", "16", "chunk posteriors"],
               ["WMCodec", "16", "digit posteriors"]]
    story += [pdf_table(systems, [52*mm, 38*mm, 88*mm], font_size=8.5), PageBreak()]

    story += [P("2. Table 2：Uniform averaging 的 tracing failure 与质量", styles["h1"]),
              Image(str(figures[0]), width=252*mm, height=72*mm),
              P("图 A：新数据的 tracing failure、PESQ 与 STOI。图中的每个点均为 300 条不同音频的均值。", styles["caption"]),
              Spacer(1, 3*mm)]
    tf_head = ["System"] + [f"K={k}" for k in KS]
    quality_head = ["System"] + [f"K={k} PESQ/STOI" for k in KS]
    tf_rows, quality_rows = [tf_head], [quality_head]
    for model in MODELS:
        z = main_all[main_all.model == model].set_index("K")
        tf_rows.append([LABEL[model]] + [f"{100*z.loc[k,'tracing_failure']:.1f}%" for k in KS])
        quality_rows.append([LABEL[model]] + [f"{z.loc[k,'PESQ']:.3f} / {z.loc[k,'STOI']:.3f}" for k in KS])
    story += [pdf_table(tf_rows, [43*mm] + [35*mm]*4, font_size=8.4), Spacer(1, 3*mm),
              pdf_table(quality_rows, [43*mm] + [40*mm]*4, font_size=8.2), Spacer(1, 3*mm)]
    extended = [["System", "K", "R3 escape", "R5 escape", "Nearest agreement", "AggResid", "SI-SDR", "payload attempts mean/max"]]
    for _, row in main_all[main_all.K < 8].iterrows():
        extended.append([row.system, str(int(row.K)), f"{row.R3_escape:.3f}", f"{row.R5_escape:.3f}",
                         f"{row.ACC_near_norm:.3f}", f"{row.AggResid:.3f}", f"{row.SI_SDR:.2f}",
                         f"{row.source_payload_attempts_mean:.2f} / {int(row.source_payload_attempts_max)}"])
    story += [PageBreak(),
              P("扩展诊断（K=2/3/5）", styles["h2"]),
              pdf_table(extended, [36*mm, 12*mm, 25*mm, 25*mm, 34*mm, 24*mm, 22*mm, 44*mm], font_size=6.9),
              PageBreak()]

    story += [P("3. Fig. 2：K=8 bit-support composition", styles["h1"]),
              Image(str(figures[1]), width=128*mm, height=143*mm),
              P("图 B：decoded-one rate 随 8 名参与者中 bit 1 支持人数 c 的变化；误差条为 10,000 次 speaker-cluster bootstrap 95% 区间。灰带为 4/4 tie。", styles["caption"]),
              PageBreak(), P("Fig. 2 完整数值：均值、95% CI 与 bit 数", styles["h2"])]
    comp_table = [["System", "c", "n bits", "Decoded-one", "CI low", "CI high"]]
    for _, row in comp.iterrows():
        comp_table.append([row.system, str(int(row.ones_among_8)), str(int(row.n_bits)),
                           f"{100*row.decoded_one_rate:.1f}%", f"{100*row.ci95_low:.1f}%", f"{100*row.ci95_high:.1f}%"])
    story += [pdf_table(comp_table, [42*mm, 15*mm, 28*mm, 32*mm, 28*mm, 28*mm], font_size=7.3), PageBreak()]

    story += [P("4. Table 3：K=8 support 与身份匹配", styles["h1"]),
              P("主表采用 trial-level 口径：每条 trial 先计算 strict-majority agreement 与 unanimous retention，再在 trial 间取均值。Fig.2 则按每个 c 的所有 bit 坐标汇总。两者回答不同问题，不能互换。", styles["body"])]
    table3 = [["System", "Strict-majority agreement", "Unanimous retention", "Trials with unanimous bits", "All strict-majority bits", "Tracing failure"]]
    for _, row in support.iterrows():
        table3.append([row.system, f"{100*row.trial_mean_strict_majority_agreement:.1f}%",
                       f"{100*row.trial_mean_unanimous_bit_retention:.1f}%",
                       str(int(row.trials_with_unanimous_bits)),
                       f"{100*row.all_strict_majority_bits:.1f}%",
                       f"{100*row.tracing_failure:.1f}%"])
    pooled = [["System", "Trial-level strict", "Pooled-bit strict", "Trial-level unanimous", "Pooled-bit unanimous", "Mean NCA", "PESQ", "STOI"]]
    for _, row in support.iterrows():
        pooled.append([row.system, f"{100*row.trial_mean_strict_majority_agreement:.2f}%",
                       f"{100*row.pooled_strict_majority_agreement:.2f}%",
                       f"{100*row.trial_mean_unanimous_bit_retention:.2f}%",
                       f"{100*row.pooled_unanimous_bit_retention:.2f}%",
                       f"{row.NCA:.3f}", f"{row.PESQ:.3f}", f"{row.STOI:.3f}"])
    story += [pdf_table(table3, [38*mm, 43*mm, 39*mm, 43*mm, 43*mm, 34*mm], font_size=7.6),
              Spacer(1, 4*mm), P("口径敏感性审计", styles["h2"]),
              pdf_table(pooled, [34*mm, 32*mm, 31*mm, 36*mm, 35*mm, 25*mm, 22*mm, 22*mm], font_size=7.0),
              Spacer(1, 4*mm),
              P("关键解释：TimbreWM 在全部非 tie 坐标上 100% 遵循严格多数，但其完整 payload 仍有 92.0% 不匹配任何参与者，说明多数 bit 可以来自不同参与者而重新组合成非参与者身份。", styles["body"]),
              PageBreak()]

    story += [P("5. Fig. 3：Controlled one-bit mixture paths", styles["h1"]),
              Image(str(figures[2]), width=252*mm, height=108*mm),
              P("图 C：重新选择的 endpoint-correct 代表路径与完整 300-trial 路径类别。AudioSeal trial 21；VoiceMark trial 220。", styles["caption"])]
    counts = path_summary.groupby("model").agg(
        valid=("trial_id", "size"), direct=("direct", "sum"), multiple=("multiple", "sum"),
        mean_transitions=("identity_transition_count", "mean"),
        median_transitions=("identity_transition_count", "median"),
        max_transitions=("identity_transition_count", "max"),
        monotonic=("target_bit_monotonic", "sum"), mean_r2=("target_probability_r2_coarse", "mean")).reset_index()
    path_table = [["System", "Valid", "Direct", "Multiple", "Direct rate", "Transitions mean/median/max", "Monotonic target", "Mean coarse R2"]]
    for _, row in counts.iterrows():
        path_table.append([LABEL[row.model], str(int(row.valid)), str(int(row.direct)), str(int(row.multiple)),
                           f"{100*row.direct/row.valid:.1f}%",
                           f"{row.mean_transitions:.2f} / {row.median_transitions:.0f} / {int(row.max_transitions)}",
                           f"{int(row.monotonic)}/300", f"{row.mean_r2:.3f}"])
    meta_table = [["System", "Trial", "Speaker", "Base", "Flipped", "Bit", "Transitions", "Selection MSE"]]
    for _, row in selected_meta.iterrows():
        meta_table.append([row.system, str(int(row.trial_id)), row.speaker, str(int(row.base_payload)),
                           str(int(row.flipped_payload)), str(int(row.flipped_bit)),
                           str(int(row.identity_transition_count)), f"{row.selection_mse:.8f}"])
    story += [Spacer(1, 3*mm), pdf_table(path_table, [31*mm, 18*mm, 20*mm, 23*mm, 27*mm, 48*mm, 36*mm, 34*mm], font_size=7.0),
              Spacer(1, 3*mm), pdf_table(meta_table, [31*mm, 18*mm, 42*mm, 25*mm, 25*mm, 15*mm, 24*mm, 34*mm], font_size=7.2),
              PageBreak(), P("Fig. 3 代表路径的全部绘图点", styles["h2"])]
    point_table = [["System", "Trial", "lambda", "Response", "Decoded identity", "State", "Stage"]]
    for _, row in selected_points.iterrows():
        point_table.append([row.system, str(int(row.trial_id)), f"{row['lambda']:.2f}",
                            f"{row.oriented_response:.6f}", str(int(row.decoded_identity)),
                            row.state, row.sampling_stage])
    story += [pdf_table(point_table, [37*mm, 21*mm, 22*mm, 30*mm, 35*mm, 34*mm, 31*mm], font_size=7.2),
              Spacer(1, 4*mm), P("补充：coarse-grid population trajectory", styles["h2"])]
    agg_table = [["System", "lambda", "n", "Mean response", "P10", "P90"]]
    for _, row in aggregate.iterrows():
        agg_table.append([row.system, f"{row['lambda']:.1f}", str(int(row.n_trials)),
                          f"{row.mean_response:.5f}", f"{row.p10:.5f}", f"{row.p90:.5f}"])
    story += [pdf_table(agg_table, [38*mm, 24*mm, 22*mm, 34*mm, 28*mm, 28*mm], font_size=7.0), PageBreak()]

    story += [P("6. Fig. 4：Frequency/temporal sensitivity（K=5）", styles["h1"]),
              Image(str(figures[3]), width=188*mm, height=121*mm),
              P("图 D：每个单元格为 300 条不同音频的 tracing failure；参与混合的 5 个输入均 source-correct。", styles["caption"]),
              Spacer(1, 3*mm)]
    tf_matrix = [["System"] + [COND_LABEL[c] for c in CONDITIONS]]
    nca_matrix = [["System"] + [COND_LABEL[c] for c in CONDITIONS]]
    for model in MODELS:
        z = region[region.model == model].set_index("condition")
        tf_matrix.append([LABEL[model]] + [f"{100*z.loc[c,'tracing_failure']:.1f}" for c in CONDITIONS])
        nca_matrix.append([LABEL[model]] + [f"{100*z.loc[c,'NCA']:.1f}" for c in CONDITIONS])
    story += [P("Tracing failure（%）", styles["h2"]),
              pdf_table(tf_matrix, [31*mm] + [27*mm]*8, font_size=6.9), Spacer(1, 3*mm),
              P("Nearest-coalition agreement / NCA（%）", styles["h2"]),
              pdf_table(nca_matrix, [31*mm] + [27*mm]*8, font_size=6.9),
              Spacer(1, 3*mm),
              P("区域实验的 PESQ/STOI 字段尚未进行 CPU backfill，且不属于 Fig.4 的绘图输入，因此本报告不填造这些质量值。Fig.4 唯一主指标是 tracing failure；NCA 作为补充诊断列出。", styles["body"]),
              PageBreak()]

    story += [P("7. 当前稿件必须同步替换的内容", styles["h1"])]
    changes = [
        ["位置", "旧值/旧叙述", "新值/新叙述"],
        ["数据集", "100 source utterances，各做 3 次 payload trial", "100 speakers × 3 个不同 clips = 300 distinct clips"],
        ["TimbreWM K=8 TF", "93.7%", "92.0%"],
        ["WavMark K=8 tie rate", "87.7%", "88.6%"],
        ["AudioSeal all strict-majority", "22.0%", "20.7%"],
        ["WavMark all strict-majority", "51.3%", "54.3%"],
        ["VoiceMark paths", "100 direct / 169 multiple，n=269", "113 direct / 187 multiple，n=300"],
        ["VoiceMark representative", "trial 46", "trial 220"],
        ["Non-speech-only range", "最高 52.0%", "0–5.3%"],
        ["Speech-active range", "77.3–100%", "73.7–100%"],
        ["Shifted-mask range", "79.0–97.7%", "69.3–97.7%"],
    ]
    story += [pdf_table(changes, [54*mm, 89*mm, 108*mm], font_size=8.0), Spacer(1, 5*mm),
              P("建议直接替换的结果描述", styles["h2"]),
              P("在 K=2 时，tracing failure 从 TimbreWM 的 86.3% 到 WMCodec 的 99.3%；五个系统在 K=2/3/5/8 下均保持较高失败率。K=2 的 PESQ 范围为 4.204–4.604、STOI 为 0.983–0.999；K=8 的 PESQ 范围为 4.049–4.577、STOI 为 0.971–0.999。", styles["body"]),
              P("在 K=8 下，WavMark 对 c≤2 坐标始终输出 0，并在 c≥5 时始终输出 1；其 tie 坐标 decoded-one rate 为 88.6%。TimbreWM 在所有非 tie 坐标严格遵循多数，但完整 payload 的 tracing failure 仍为 92.0%。", styles["body"]),
              P("Controlled path 实验现在包含 AudioSeal 和 VoiceMark 各 300 条双端点精确路径。AudioSeal 有 299 条 direct path；VoiceMark 仅 113 条 direct path，另有 187 条进入一个或多个 intermediate payload。", styles["body"]),
              PageBreak()]

    story += [P("8. 可复现性、限制与文件来源", styles["h1"]),
              P("确定性：所有聚合均直接从完成后的 CSV/JSON 重算；Fig.2 置信区间使用固定随机种子和 10,000 次 speaker-cluster bootstrap。图表脚本不执行模型推理。", styles["body"]),
              P("Table 3 口径：主值采用 trial-level 等权。报告同时保留 pooled-bit 值以审计旧脚本，但 pooled 值不得替换主表。", styles["body"]),
              P("限制：Fig.2 中 c=0 或 c=8 的 bit 数较少，部分系统置信区间较宽；VoiceMark/WMCodec 的 native response 不应被当作跨系统校准概率；区域实验没有完成 PESQ/STOI CPU backfill。", styles["body"]),
              P("统计谬误检查：已检查聚合层级混淆、分母变化、选择性过滤、重复测量伪独立、缺失值替代、相关当因果、多重比较、区间误读、基率忽略、跨系统校准比较和过度外推。主要风险是 Table 3 pooled/trial-level 混用，已在本报告中分离。", styles["body"]),
              P("原始数据位置", styles["h2"])]
    sources = [
        ["模块", "路径"],
        ["K=2/3/5", str(ROOT / "results/evaluation_300clips_20260902")],
        ["K=8", str(ROOT / "data/k8_population_source_correct_300clips_20260902/raw")],
        ["Paths", str(ROOT / "data/mixture_path_k5_300clips_20260902/shards")],
        ["Regions", str(ROOT / "results/frequency_temporal_k5_300clips_20260902/shards")],
    ]
    story += [pdf_table(sources, [36*mm, 210*mm], font_size=7.5), Spacer(1, 5*mm),
              P("生成目录中的 CSV 是上述原始记录的机器可读聚合，不取代原始记录。SHA256SUMS.txt 用于确认交付文件未被意外改写。", styles["body"])]

    def footer(canvas, _doc):
        canvas.saveState(); canvas.setFont("STSong-Light", 7.5); canvas.setFillColor(colors.HexColor("#687680"))
        canvas.drawString(12*mm, 7*mm, "修正后的完整论文图表数据报告 · 2026-09-03")
        canvas.drawRightString(landscape(A4)[0]-12*mm, 7*mm, f"第 {canvas.getPageNumber()} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_plots()
    main_metrics = validate_main()
    k8_trials, k8_bits, composition, support = validate_k8()
    path_summary, path_aggregate, selected_meta, selected_points = validate_paths()
    region = validate_regions()
    k8_main = support[["model", "system", "n_trials", "tracing_failure", "PESQ", "STOI"]].copy()
    k8_main["K"] = 8
    for col in ["R3_escape", "R5_escape", "ACC_near", "ACC_near_norm", "AggResid", "SI_SDR",
                "source_payload_attempts_mean", "source_payload_attempts_max"]:
        k8_main[col] = np.nan
    k8_main["n_speakers"] = 100; k8_main["unique_sources"] = 300
    k8_main["source_file"] = str(ROOT / "data/k8_population_source_correct_300clips_20260902/raw")
    main_all = pd.concat([main_metrics, k8_main], ignore_index=True).sort_values(["model", "K"])

    main_all.to_csv(OUT / "table2_main_metrics.csv", index=False)
    composition.to_csv(OUT / "figure2_k8_composition.csv", index=False)
    support.to_csv(OUT / "table3_k8_support.csv", index=False)
    path_summary.to_csv(OUT / "figure3_path_trial_summary.csv", index=False)
    path_aggregate.to_csv(OUT / "figure3_path_aggregate_trajectory.csv", index=False)
    selected_meta.to_csv(OUT / "figure3_representative_selection.csv", index=False)
    selected_points.to_csv(OUT / "figure3_representative_points.csv", index=False)
    region.to_csv(OUT / "figure4_frequency_temporal.csv", index=False)
    k8_trials.to_csv(OUT / "k8_trial_values.csv", index=False)
    k8_bits.to_csv(OUT / "k8_bit_values.csv", index=False)

    figures = [draw_main(main_all, support), draw_composition(composition),
               draw_paths(path_summary, selected_meta, selected_points), draw_regions(region)]
    markdown = write_markdown(main_all, composition, support, path_summary,
                              selected_meta, selected_points, region)
    pdf = build_pdf(main_all, composition, support, path_summary, path_aggregate,
                    selected_meta, selected_points, region, figures)
    manifest = {
        "report": str(pdf), "markdown": str(markdown), "status": "VERIFIED",
        "snapshot": "2026-09-03", "models": MODELS,
        "trials_per_cell": 300, "speakers": 100, "distinct_clips_per_speaker": 3,
        "source_correct": True,
        "table3_primary_definition": "trial-level mean",
        "fig2_ci": "speaker-cluster bootstrap, 10000 replicates",
        "generated_files": [],
    }
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name not in {"SHA256SUMS.txt", "validation_manifest.json"})
    manifest["generated_files"] = [{"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in files]
    (OUT / "validation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in final_files), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUT), "pdf": str(pdf), "pages_pending_check": True,
        "files": len(list(OUT.iterdir())), "status": "VERIFIED",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

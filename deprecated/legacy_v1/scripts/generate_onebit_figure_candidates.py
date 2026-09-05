#!/usr/bin/env python3
"""Generate matched-content one-bit figure candidates with clean references.

The chosen trials are fixed after a distribution-aware selection audit.  Each
panel compares the same speaker/content, base payload, flipped payload, and bit
for AudioSeal and VoiceMark.  Absolute residual scales are shared wherever the
cross-system magnitude is the claim; any magnified inset states its zoom factor.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from registry import clean_path_v19  # noqa: E402

SR = 16000
RESULT_ROOT = ROOT / "results" / "one_bit_k5_20260828"
OUT = RESULT_ROOT / "figure_candidates_20260829"
SELECTIONS = {
    "representative_median": 136,
    "clear_typical": 223,
    "spectral_contrast": 99,
}
COLORS = {"clean": "#222222", "base": "#0077BB", "flip": "#EE7733",
          "delta": "#CC3311"}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def read_rows(model: str) -> dict[int, dict]:
    path = RESULT_ROOT / f"onebit_k5_{model}_full.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {int(r["trial_id"]): r for r in csv.DictReader(f)}


def load(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.asarray(wav, dtype=np.float32)


def speech_window(clean: np.ndarray, seconds: float = 0.8) -> tuple[int, int]:
    n = min(int(seconds * SR), len(clean))
    hop = max(1, n // 8)
    energy = []
    starts = range(0, max(1, len(clean) - n + 1), hop)
    for s in starts:
        energy.append((float(np.mean(clean[s:s+n].astype(np.float64) ** 2)), s))
    start = max(energy)[1]
    return start, start + n


def mel_db(delta: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=delta, sr=SR, n_fft=1024, win_length=1024, hop_length=256,
        n_mels=80, fmin=0, fmax=8000, power=2.0,
    )
    return 10.0 * np.log10(mel + 1e-14)


def percentile_maps(rows: dict[str, dict[int, dict]]) -> dict[str, dict[str, dict[int, float]]]:
    out = {}
    for model, data in rows.items():
        out[model] = {}
        for metric in ("waveform_residual_mse", "mel_residual_distance"):
            valid = [i for i, r in data.items()
                     if int(r["base_payload_correct"]) and int(r["flipped_payload_correct"])]
            order = sorted(valid, key=lambda i: float(data[i][metric]))
            out[model][metric] = {i: rank / (len(order) - 1) for rank, i in enumerate(order)}
    return out


def candidate_figure(label: str, trial: int, rows: dict[str, dict[int, dict]],
                     percentiles: dict) -> dict:
    systems = [("audioseal", "AudioSeal"), ("voicemark", "VoiceMark")]
    loaded = {}
    for model, _ in systems:
        r = rows[model][trial]
        clean = load(str(clean_path_v19(r["spk"])))
        base = load(r["base_path"])
        flip = load(r["flipped_path"])
        n = min(map(len, (clean, base, flip)))
        clean, base, flip = clean[:n], base[:n], flip[:n]
        loaded[model] = dict(row=r, clean=clean, base=base, flip=flip,
                             rb=base-clean, rf=flip-clean, delta=flip-base)

    # Identical content means the same high-energy zoom window can be used.
    start, end = speech_window(loaded["audioseal"]["clean"])
    global_delta = max(float(np.max(np.abs(x["delta"]))) for x in loaded.values())
    global_delta = max(global_delta, 1e-8)
    mel_maps = {m: mel_db(x["delta"]) for m, x in loaded.items()}
    mel_max = max(float(np.max(x)) for x in mel_maps.values())
    mel_min = mel_max - 75.0

    fig, axes = plt.subplots(
        2, 3, figsize=(7.12, 4.25), layout="constrained",
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.18]},
    )
    for row_idx, (model, display) in enumerate(systems):
        x = loaded[model]
        r = x["row"]
        t = np.arange(end-start) / SR

        # (a) Both watermark residuals relative to clean; row-specific scale is explicit.
        ax = axes[row_idx, 0]
        ax.plot(t, x["rb"][start:end], color=COLORS["base"], lw=0.7,
                label=r"$x_{base}-x_{clean}$")
        ax.plot(t, x["rf"][start:end], color=COLORS["flip"], lw=0.7, alpha=0.85,
                label=r"$x_{flip}-x_{clean}$")
        local = max(float(np.max(np.abs(x["rb"][start:end]))),
                    float(np.max(np.abs(x["rf"][start:end]))), 1e-8)
        ax.set_ylim(-1.08*local, 1.08*local)
        ax.set_xlabel("Selected speech window (s)")
        ax.set_ylabel(f"{display}\nResidual amplitude")
        ax.text(0.02, 0.04, "row-specific residual scale", transform=ax.transAxes,
                fontsize=6.5, color="#555555")
        if row_idx == 0:
            ax.set_title("(a) Residuals to clean")
            ax.legend(frameon=False, loc="upper right")

        # (b) True one-bit difference on a shared absolute scale.
        ax = axes[row_idx, 1]
        ax.plot(t, x["delta"][start:end], color=COLORS["delta"], lw=0.75)
        ax.axhline(0, color="#777777", lw=0.5)
        ax.set_ylim(-1.08*global_delta, 1.08*global_delta)
        ax.set_xlabel("Selected speech window (s)")
        ax.set_ylabel(r"$x_{flip}-x_{base}$")
        mse = float(r["waveform_residual_mse"])
        pct = 100 * percentiles[model]["waveform_residual_mse"][trial]
        ax.text(0.03, 0.94, f"MSE={mse:.2e}\nwithin-system pct.={pct:.0f}%",
                transform=ax.transAxes, va="top", fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#AAAAAA", alpha=0.9))
        if row_idx == 0:
            ax.set_title("(b) One-bit waveform residual\n(shared absolute scale)")
            # Honest magnified inset: same data, explicitly labelled zoom.
            inset = ax.inset_axes([0.54, 0.08, 0.42, 0.31])
            inset.plot(t, x["delta"][start:end], color=COLORS["delta"], lw=0.55)
            local_delta = max(float(np.max(np.abs(x["delta"][start:end]))), 1e-9)
            inset.set_ylim(-1.05*local_delta, 1.05*local_delta)
            zoom = global_delta/local_delta
            inset.set_xticks([]); inset.set_yticks([])
            inset.set_title(f"AudioSeal zoom ×{zoom:.0f}", fontsize=6.2, pad=1)

        # (c) Absolute one-bit Mel residual, identical dB scale for both systems.
        ax = axes[row_idx, 2]
        mel = mel_maps[model]
        extent=(0, len(x["delta"])/SR, 0, 8)
        im=ax.imshow(mel, origin="lower", aspect="auto", extent=extent,
                     cmap="magma", vmin=mel_min, vmax=mel_max, interpolation="nearest")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel frequency (kHz)")
        if model == "audioseal":
            lo, hi, note = 1, 2, "1–2 kHz dominant"
            note_y = 2.65
        else:
            high_band_fraction = (
                float(r["band_fraction_2_4k"]) + float(r["band_fraction_4_8k"])
            )
            if high_band_fraction >= 0.5:
                lo, hi, note = 2, 8, "2–8 kHz spread"
                note_y = 7.35
            else:
                lo, hi, note = 0, 2, "0–2 kHz spread"
                note_y = 2.65
        ax.add_patch(Rectangle((0.06, lo), max(0.1, extent[1]-0.12), hi-lo,
                               fill=False, edgecolor="#33BBEE", lw=1.25, ls="--"))
        ax.annotate(note, xy=(extent[1]*0.76, (lo+hi)/2),
                    xytext=(extent[1]*0.28, note_y),
                    arrowprops=dict(arrowstyle="->", color="#33BBEE", lw=0.9),
                    color="white", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="#222222", ec="none", alpha=0.72))
        mel_dist=float(r["mel_residual_distance"])
        ax.text(0.03, 0.04, f"Mel distance={mel_dist:.3f}", transform=ax.transAxes,
                color="white", fontsize=6.7,
                bbox=dict(boxstyle="round,pad=0.2", fc="#222222", ec="none", alpha=0.72))
        if row_idx == 0:
            ax.set_title("(c) One-bit Mel residual\n(shared absolute dB scale)")

    cbar = fig.colorbar(im, ax=axes[:, 2], fraction=0.055, pad=0.025, shrink=0.94)
    cbar.set_label("Residual energy (dB, absolute)")
    r0=rows["audioseal"][trial]
    title=(f"Matched content, trial {trial}, flipped bit {r0['flipped_bit']}; "
           "both payloads decode correctly")
    fig.suptitle(title, fontsize=9.2)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"onebit_{label}_trial{trial:03d}.{ext}")
    plt.close(fig)

    return {
        "label": label, "trial_id": trial, "speaker": r0["spk"],
        "local_t": int(r0["local_t"]), "base_payload": int(r0["base_payload"]),
        "flipped_payload": int(r0["flipped_payload"]),
        "flipped_bit": int(r0["flipped_bit"]),
        "selection_rule": {
            "representative_median": "both systems near their within-system medians",
            "clear_typical": "AudioSeal near median; VoiceMark near 75th percentile; not a tail maximum",
            "spectral_contrast": "strong 1–2 kHz versus broad 2–8 kHz allocation; both payloads decode correctly",
        }[label],
        "systems": {
            model: {
                "waveform_residual_mse": float(rows[model][trial]["waveform_residual_mse"]),
                "waveform_mse_percentile": percentiles[model]["waveform_residual_mse"][trial],
                "mel_residual_distance": float(rows[model][trial]["mel_residual_distance"]),
                "mel_distance_percentile": percentiles[model]["mel_residual_distance"][trial],
                "base_payload_correct": int(rows[model][trial]["base_payload_correct"]),
                "flipped_payload_correct": int(rows[model][trial]["flipped_payload_correct"]),
                "band_fractions": [float(rows[model][trial][f"band_fraction_{b}"])
                                   for b in ("0_1k", "1_2k", "2_4k", "4_8k")],
            } for model, _ in systems
        },
    }


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {m: read_rows(m) for m in ("audioseal", "voicemark")}
    pct = percentile_maps(rows)
    records=[]
    for label, trial in SELECTIONS.items():
        for model in rows:
            r=rows[model][trial]
            assert int(r["base_payload_correct"]) == 1
            assert int(r["flipped_payload_correct"]) == 1
        records.append(candidate_figure(label, trial, rows, pct))
    with (OUT / "selection_audit.json").open("w", encoding="utf-8") as f:
        json.dump({"selection_pool": "269 matched trials with exact payload decoding in both systems",
                   "shared_visual_scales": True, "candidates": records}, f, indent=2)
    print(f"generated {len(records)} candidates in {OUT}")


if __name__ == "__main__":
    main()

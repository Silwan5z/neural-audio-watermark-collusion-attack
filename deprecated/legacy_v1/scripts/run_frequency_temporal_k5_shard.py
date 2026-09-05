#!/usr/bin/env python3
"""Frequency and speech/non-speech Mean-collusion ablations (K=5).

The runner is intentionally shard-local: seven independent processes can use
seven GPUs without sharing CSV files.  Every attacked waveform is retained.
PESQ/STOI are backfilled by the CPU finalizer so GPU workers do not stall on
quality metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from registry import (NBITS, cache_path_for, clean_path_v19, coalition_seed,
                      full_registry_bits, get_or_embed,  # noqa: E402
                      int_to_bits, load_clean, sample_coalition, source_record,
                      speaker_trial_index)
from watermarks import detect_many  # noqa: E402

K = 5
N_TRIALS = 300
SR = 16000
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
CONDITIONS = [
    "freq_0_1k", "freq_1_2k", "freq_2_4k", "freq_4_8k",
    "speech_only", "non_speech_only", "full_waveform", "random_mask",
]
FIELDS = [
    "model", "K", "trial_id", "spk", "local_t", "clip_index", "source_path",
    "source_exact_count", "source_payload_attempts",
    "shard_id", "num_shards",
    "condition", "condition_family", "coalition_payloads", "coalition_payload_bits",
    "attack_weights", "reference_payload", "reference_audio_path", "output_audio_path",
    "sample_rate", "n_samples", "frequency_low_hz", "frequency_high_hz",
    "frequency_transition_hz", "vad_method", "vad_top_db", "mask_smoothing_ms",
    "speech_binary_fraction", "mask_effective_fraction", "random_circular_shift_samples",
    "mask_path", "frequency_partition_reconstruction_mse", "temporal_complement_error",
    "decode_valid", "decoded_identity", "decoded_payload_bits", "tracing_failure",
    "NCA_bits", "NCA", "presence", "PESQ", "STOI",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=7)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--models", nargs="+", choices=MODELS,
                   help="optional subset; useful when GPU memory limits exclude a backend")
    p.add_argument("--prepare-only", action="store_true",
                   help="CPU-only: generate masks, attacked WAVs, and metadata without decoding")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "frequency_temporal_k5_20260828")
    return p.parse_args()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_wav(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.wav")
    sf.write(tmp, wav.astype(np.float32), SR, subtype="FLOAT")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_precomputed_trial(out: Path, model: str, trial_id: int,
                           coalition: list[int]) -> tuple[dict[str, np.ndarray], dict] | None:
    marker = out / "precomputed" / model / f"trial_{trial_id:03d}.json"
    if not marker.exists():
        return None
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta["coalition_payloads"] != coalition or meta["conditions"] != CONDITIONS:
        raise ValueError(f"precompute metadata mismatch: {marker}")
    signals = {}
    for condition in CONDITIONS:
        path = out / "audio" / model / f"trial_{trial_id:03d}" / f"{condition}.wav"
        if not path.exists():
            return None
        wav, sr = sf.read(path, dtype="float32")
        if sr != SR or wav.ndim != 1 or len(wav) != int(meta["n_samples"]) or not np.isfinite(wav).all():
            raise ValueError(f"invalid precomputed WAV: {path}")
        signals[condition] = wav
    return signals, meta


def load_rows(path: Path, model: str, shard_id: int) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row["model"] != model or int(row["shard_id"]) != shard_id:
            raise ValueError(f"model/shard mismatch in {path}")
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    complete = {
        trial for trial, rr in grouped.items()
        if len(rr) == len(CONDITIONS) and {r["condition"] for r in rr} == set(CONDITIONS)
    }
    kept = [r for r in rows if int(r["trial_id"]) in complete]
    return kept, complete


def soft_lowpass(freq: np.ndarray, cutoff: float, transition: float) -> np.ndarray:
    low, high = cutoff - transition, cutoff + transition
    out = np.ones_like(freq, dtype=np.float64)
    out[freq >= high] = 0.0
    mid = (freq > low) & (freq < high)
    phase = (freq[mid] - low) / (high - low)
    out[mid] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return out


def frequency_masks(n_fft: int = 1024, transition_hz: float = 100.0) -> dict[str, np.ndarray]:
    freq = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    l1 = soft_lowpass(freq, 1000.0, transition_hz)
    l2 = soft_lowpass(freq, 2000.0, transition_hz)
    l4 = soft_lowpass(freq, 4000.0, transition_hz)
    masks = {
        "freq_0_1k": l1,
        "freq_1_2k": l2 - l1,
        "freq_2_4k": l4 - l2,
        "freq_4_8k": 1.0 - l4,
    }
    total = sum(masks.values())
    if np.max(np.abs(total - 1.0)) > 1e-12 or any(np.min(m) < -1e-12 for m in masks.values()):
        raise RuntimeError("frequency masks are not a non-negative partition of unity")
    return masks


def energy_vad_masks(clean: np.ndarray, trial_id: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic energy VAD with a smooth, duration-matched random control."""
    n = len(clean)
    intervals = librosa.effects.split(clean, top_db=35, frame_length=1024, hop_length=160)
    binary = np.zeros(n, dtype=np.float32)
    for start, end in intervals:
        binary[max(0, int(start)):min(n, int(end))] = 1.0
    # About 20 ms from 2 sigma on each side; smoothing is applied before the
    # circular shift so speech and random masks have exactly equal mass.
    sigma_samples = int(round(0.005 * SR))
    smooth = gaussian_filter1d(binary, sigma=sigma_samples, mode="nearest")
    smooth = np.clip(smooth, 0.0, 1.0).astype(np.float32)
    rng = np.random.default_rng(20260828 + 104729 * trial_id)
    min_shift = min(n - 1, int(0.5 * SR))
    shift = int(rng.integers(min_shift, n)) if n > min_shift else 0
    random_mask = np.roll(smooth, shift)
    meta = {
        "intervals": intervals.astype(int),
        "binary_fraction": float(binary.mean()),
        "effective_fraction": float(smooth.mean()),
        "shift": shift,
        "sigma_samples": sigma_samples,
    }
    return smooth, random_mask, meta


def make_outputs(reference: np.ndarray, mean: np.ndarray, clean: np.ndarray,
                 trial_id: int) -> tuple[dict[str, np.ndarray], dict]:
    n = min(len(reference), len(mean), len(clean))
    reference, mean, clean = reference[:n], mean[:n], clean[:n]
    delta = mean - reference
    n_fft, hop, win = 1024, 256, 1024
    spectrum = librosa.stft(delta, n_fft=n_fft, hop_length=hop, win_length=win,
                            window="hann", center=True)
    fmasks = frequency_masks(n_fft)
    filtered = {}
    for name, mask in fmasks.items():
        band_delta = librosa.istft(spectrum * mask[:, None], hop_length=hop,
                                   win_length=win, window="hann", center=True, length=n)
        filtered[name] = (reference + band_delta).astype(np.float32)
    reconstructed = sum((filtered[name] - reference) for name in fmasks)
    freq_error = float(np.mean((reconstructed - delta) ** 2))

    speech, random_mask, vad = energy_vad_masks(clean, trial_id)
    nonspeech = 1.0 - speech
    outputs = dict(filtered)
    outputs.update({
        "speech_only": (reference + speech * delta).astype(np.float32),
        "non_speech_only": (reference + nonspeech * delta).astype(np.float32),
        "full_waveform": mean.astype(np.float32),
        "random_mask": (reference + random_mask * delta).astype(np.float32),
    })
    temporal_error = float(np.max(np.abs(
        (outputs["speech_only"] - reference) +
        (outputs["non_speech_only"] - reference) - delta
    )))
    if set(outputs) != set(CONDITIONS) or not all(np.isfinite(x).all() for x in outputs.values()):
        raise RuntimeError("invalid condition outputs")
    return outputs, {"vad": vad, "frequency_error": freq_error,
                     "temporal_error": temporal_error,
                     "speech_mask": speech, "random_mask": random_mask}


def payload_json(payloads: list[int], d: int) -> str:
    return json.dumps([int_to_bits(x, d).astype(int).tolist() for x in payloads],
                      separators=(",", ":"))


def source_correct_coalition(model: str, spk: str, local_t: int,
                             rng: np.random.Generator, registry: np.ndarray,
                             max_attempts: int = 1000) -> tuple[list[int], list[np.ndarray], int]:
    """Draw five distinct payloads whose individual marked copies decode exactly."""
    selected: list[int] = []
    wavs: list[np.ndarray] = []
    attempts = 0
    limit = 1 << NBITS[model]
    while len(selected) < K:
        needed = K - len(selected)
        candidates: list[int] = []
        while len(candidates) < needed:
            payload = int(rng.integers(0, limit))
            if payload not in selected and payload not in candidates:
                candidates.append(payload)
        candidate_wavs = [get_or_embed(model, spk, payload, local_t)
                          for payload in candidates]
        decoded = detect_many(model, candidate_wavs, registry)
        attempts += len(candidates)
        for payload, wav, (_, _, hard) in zip(candidates, candidate_wavs, decoded):
            if hard is None:
                continue
            got = int(sum(int(v) << i for i, v in enumerate(hard)))
            if got == payload:
                selected.append(payload)
                wavs.append(wav)
        if attempts >= max_attempts and len(selected) < K:
            raise RuntimeError(
                f"{model} {spk} local_t={local_t}: only {len(selected)}/{K} "
                f"source-correct payloads after {attempts} attempts")
    return selected, wavs, attempts


def decode_metrics(scores: np.ndarray, hard, presence: float,
                   coalition: list[int], d: int) -> dict:
    if hard is None:
        return {"decode_valid": 0, "decoded_identity": "", "decoded_payload_bits": "",
                "tracing_failure": 1, "NCA_bits": "", "NCA": "",
                "presence": "" if not np.isfinite(presence) else float(presence)}
    hard = np.asarray(hard, dtype=np.int8)
    decoded = int(np.argsort(scores, kind="stable")[::-1][0])
    coll_bits = np.stack([int_to_bits(x, d) for x in coalition])
    nca_bits = int(np.max(np.sum(coll_bits == hard[None, :], axis=1)))
    return {
        "decode_valid": 1,
        "decoded_identity": decoded,
        "decoded_payload_bits": json.dumps(hard.astype(int).tolist(), separators=(",", ":")),
        "tracing_failure": int(decoded not in set(coalition)),
        "NCA_bits": nca_bits,
        "NCA": nca_bits / d,
        "presence": "" if not np.isfinite(presence) else float(presence),
    }


def run_model(args: argparse.Namespace, model: str, trial_plan: list[tuple[str, int]]) -> None:
    out = args.output_dir
    stem = f"frequency_temporal_k5_{model}_shard{args.shard_id}of{args.num_shards}"
    partial, final = out / "shards" / f"{stem}.partial.csv", out / "shards" / f"{stem}.csv"
    partial.parent.mkdir(parents=True, exist_ok=True)
    source = final if final.exists() else partial
    rows, complete = load_rows(source, model, args.shard_id) if not args.prepare_only else ([], set())
    registry = None if args.prepare_only else full_registry_bits(model)
    d = NBITS[model]
    assigned = list(range(args.shard_id, args.n_trials, args.num_shards))
    since_save = 0

    for trial_id in assigned:
        if not args.prepare_only and trial_id in complete:
            continue
        spk, local_t = trial_plan[trial_id]
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        if args.prepare_only:
            coalition = sample_coalition(rng, model, K)
            source_wavs, source_attempts = None, ""
        else:
            coalition, source_wavs, source_attempts = source_correct_coalition(
                model, spk, local_t, rng, registry)
        mask_path = out / "masks" / f"trial_{trial_id:03d}.npz"
        cached = load_precomputed_trial(out, model, trial_id, coalition)
        if cached is None:
            source = source_record(spk, local_t)
            wavs = ([get_or_embed(model, spk, identity, local_t) for identity in coalition]
                    if source_wavs is None else source_wavs)
            clean = load_clean(spk, local_t)
            n = min([len(clean)] + [len(w) for w in wavs])
            wavs = [np.asarray(w[:n], dtype=np.float32) for w in wavs]
            clean = np.asarray(clean[:n], dtype=np.float32)
            reference = wavs[0]
            mean = np.mean(np.stack(wavs), axis=0, dtype=np.float64).astype(np.float32)
            signals, raw_meta = make_outputs(reference, mean, clean, trial_id)
            if not mask_path.exists():
                atomic_npz(mask_path, speech_mask=raw_meta["speech_mask"].astype(np.float16),
                           random_mask=raw_meta["random_mask"].astype(np.float16),
                           vad_intervals=raw_meta["vad"]["intervals"].astype(np.int32),
                           random_circular_shift_samples=np.asarray(raw_meta["vad"]["shift"], dtype=np.int64))
            for condition, signal in signals.items():
                audio_path = out / "audio" / model / f"trial_{trial_id:03d}" / f"{condition}.wav"
                atomic_wav(audio_path, signal)
            meta = {
                "model": model, "trial_id": trial_id, "spk": spk, "local_t": local_t,
                "clip_index": source["clip_index"],
                "source_path": str(clean_path_v19(spk, local_t)),
                "source_exact_count": K if not args.prepare_only else "",
                "source_payload_attempts": source_attempts,
                "coalition_payloads": coalition, "conditions": CONDITIONS, "sample_rate": SR,
                "n_samples": n, "speech_binary_fraction": raw_meta["vad"]["binary_fraction"],
                "speech_effective_fraction": raw_meta["vad"]["effective_fraction"],
                "random_circular_shift_samples": raw_meta["vad"]["shift"],
                "frequency_partition_reconstruction_mse": raw_meta["frequency_error"],
                "temporal_complement_error": raw_meta["temporal_error"],
                "mask_path": str(mask_path),
            }
            atomic_json(out / "precomputed" / model / f"trial_{trial_id:03d}.json", meta)
        else:
            signals, meta = cached
            n = int(meta["n_samples"])

        if args.prepare_only:
            print(f"prepared shard={args.shard_id} model={model} trial={trial_id}", flush=True)
            continue

        ordered_signals = [signals[c] for c in CONDITIONS]
        decoded = detect_many(model, ordered_signals, registry)
        trial_rows = []
        for condition, signal, (scores, presence, hard) in zip(CONDITIONS, ordered_signals, decoded):
            audio_path = out / "audio" / model / f"trial_{trial_id:03d}" / f"{condition}.wav"
            if condition.startswith("freq_"):
                family = "frequency"
                lo, hi = {"freq_0_1k": (0, 1000), "freq_1_2k": (1000, 2000),
                          "freq_2_4k": (2000, 4000), "freq_4_8k": (4000, 8000)}[condition]
                mask_fraction, random_shift = "", ""
            else:
                family, lo, hi = "temporal", "", ""
                if condition == "speech_only":
                    mask_fraction = meta["speech_effective_fraction"]
                elif condition == "non_speech_only":
                    mask_fraction = 1.0 - meta["speech_effective_fraction"]
                elif condition == "random_mask":
                    mask_fraction = meta["speech_effective_fraction"]
                else:
                    mask_fraction = 1.0
                random_shift = meta["random_circular_shift_samples"] if condition == "random_mask" else ""
            metrics = decode_metrics(scores, hard, presence, coalition, d)
            trial_rows.append({
                "model": model, "K": K, "trial_id": trial_id, "spk": spk,
                "local_t": local_t,
                "clip_index": meta["clip_index"], "source_path": meta["source_path"],
                "source_exact_count": meta.get("source_exact_count", K),
                "source_payload_attempts": meta.get("source_payload_attempts", source_attempts),
                "shard_id": args.shard_id, "num_shards": args.num_shards,
                "condition": condition, "condition_family": family,
                "coalition_payloads": json.dumps(coalition, separators=(",", ":")),
                "coalition_payload_bits": payload_json(coalition, d),
                "attack_weights": json.dumps([0.2] * K, separators=(",", ":")),
                "reference_payload": coalition[0],
                "reference_audio_path": str(cache_path_for(model, spk, local_t, coalition[0])),
                "output_audio_path": str(audio_path), "sample_rate": SR, "n_samples": n,
                "frequency_low_hz": lo, "frequency_high_hz": hi,
                "frequency_transition_hz": 100 if family == "frequency" else "",
                "vad_method": "librosa.effects.split_energy_vad" if family == "temporal" else "",
                "vad_top_db": 35 if family == "temporal" else "",
                "mask_smoothing_ms": 20 if family == "temporal" else "",
                "speech_binary_fraction": meta["speech_binary_fraction"] if family == "temporal" else "",
                "mask_effective_fraction": mask_fraction,
                "random_circular_shift_samples": random_shift,
                "mask_path": str(mask_path) if family == "temporal" else "",
                "frequency_partition_reconstruction_mse": meta["frequency_partition_reconstruction_mse"] if family == "frequency" else "",
                "temporal_complement_error": meta["temporal_complement_error"] if family == "temporal" else "",
                **metrics, "PESQ": "", "STOI": "",
            })
        rows.extend(trial_rows)
        complete.add(trial_id)
        since_save += 1
        if since_save >= args.checkpoint_every:
            rows.sort(key=lambda r: (int(r["trial_id"]), CONDITIONS.index(r["condition"])))
            atomic_csv(partial, rows)
            print(f"checkpoint shard={args.shard_id} model={model} trials={len(complete)}/{len(assigned)}", flush=True)
            since_save = 0

    if args.prepare_only:
        print(f"prepare complete shard={args.shard_id} model={model} trials={len(assigned)}", flush=True)
        return
    rows.sort(key=lambda r: (int(r["trial_id"]), CONDITIONS.index(r["condition"])))
    expected = len(assigned) * len(CONDITIONS)
    if len(rows) != expected:
        raise ValueError(f"{model} shard {args.shard_id}: expected {expected} rows, got {len(rows)}")
    atomic_csv(partial, rows); atomic_csv(final, rows)
    print(f"complete shard={args.shard_id} model={model} rows={len(rows)}", flush=True)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard id")
    if args.n_trials != N_TRIALS:
        raise ValueError("paper protocol requires exactly 300 trials")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_plan = speaker_trial_index(args.n_trials)
    # Rotate model order so seven workers do not simultaneously initialize the
    # same backend and so heterogeneous model costs are spread over time.
    selected = MODELS if args.models is None else args.models
    offset = args.shard_id % len(selected)
    order = selected[offset:] + selected[:offset]
    for model in order:
        run_model(args, model, trial_plan)


if __name__ == "__main__":
    main()

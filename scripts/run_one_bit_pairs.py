#!/usr/bin/env python3
"""Build valid K=5 payload pairs that differ by exactly one bit.

Each system receives the same 300-trial schedule: speaker/content, base payload,
and flipped bit are identical.  A trial compares two independently embedded
copies whose payloads differ by exactly one bit.  Results are checkpointed
atomically and can be resumed without overwriting existing paper data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registry import (  # noqa: E402
    NBITS,
    coalition_seed,
    get_or_embed,
    int_to_bits,
    load_clean,
    sample_coalition,
    source_record,
    trial_schedule,
)
from watermarks import (  # noqa: E402
    _chunk_logits_to_bit_evidence,
    get_audioseal,
    get_voicemark,
)

SR = 16000
COALITION_SIZE = 5
N_TRIALS = 300
SCHEDULE_SEED = 20260828
BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
FIELDNAMES = [
    "model", "k", "trial_id", "speaker", "clip_index", "source_path",
    "coalition_payloads",
    "coalition_member_index", "base_payload", "flipped_payload", "flipped_bit",
    "pair_attempts", "sample_count", "waveform_mse", "mel_distance",
    "band_energy_0_1k", "band_energy_1_2k", "band_energy_2_4k", "band_energy_4_8k",
    "band_fraction_0_1k", "band_fraction_1_2k", "band_fraction_2_4k", "band_fraction_4_8k",
    "base_decoded_payload", "flipped_decoded_payload",
    "base_payload_correct", "flipped_payload_correct", "base_changed_bit_correct",
    "flipped_changed_bit_correct", "decoded_bit_changes", "base_changed_bit_score",
    "flipped_changed_bit_score", "changed_bit_score_change",
    "unchanged_bit_score_change", "base_bit_scores", "flipped_bit_scores",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("audioseal", "voicemark"))
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument("--trial-end", type=int, default=N_TRIALS)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "one_bit" / "pairs")
    return p.parse_args()


def schedule() -> list[dict]:
    """Balanced, deterministic schedule shared by both 16-bit systems."""
    members = np.tile(
        np.arange(COALITION_SIZE, dtype=int), N_TRIALS // COALITION_SIZE)
    member_rng = np.random.default_rng(SCHEDULE_SEED + 1)
    member_rng.shuffle(members)

    bits = np.tile(np.arange(16, dtype=int), 19)[:N_TRIALS]
    bit_rng = np.random.default_rng(SCHEDULE_SEED)
    bit_rng.shuffle(bits)

    out = []
    for trial_id, (speaker, clip_slot) in enumerate(trial_schedule(N_TRIALS)):
        rng = np.random.default_rng(
            coalition_seed(speaker, COALITION_SIZE, clip_slot))
        coalition = sample_coalition(rng, "audioseal", COALITION_SIZE)
        member_index = int(members[trial_id])
        base = int(coalition[member_index])
        bit = int(bits[trial_id])
        flipped = int(base ^ (1 << bit))
        assert (base ^ flipped).bit_count() == 1
        out.append({
            "trial_id": trial_id,
            "speaker": speaker,
            "clip_slot": clip_slot,
            "coalition_payloads": coalition,
            "coalition_member_index": member_index,
            "base_payload": base,
            "flipped_payload": flipped,
            "flipped_bit": bit,
        })
    assert np.bincount(members, minlength=COALITION_SIZE).tolist() == [60] * COALITION_SIZE
    counts = np.bincount(bits, minlength=16)
    assert counts.min() == 18 and counts.max() == 19 and counts.sum() == N_TRIALS
    return out


def bits_to_int(bits: np.ndarray) -> int:
    return int(sum(int(v) << i for i, v in enumerate(bits)))


def decode_with_evidence(model: str, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One detector pass returning exact native hard bits and soft bit evidence."""
    import torch

    if model == "audioseal":
        m = get_audioseal()
        t = torch.from_numpy(wav).float().to(m["dev"])[None, None]
        with torch.no_grad():
            response = m["det"].detector(t)
        logits = response[:, 2:].mean(dim=-1)[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        hard = (probs > 0.5).astype(np.int8)
        evidence = 2.0 * probs - 1.0
    else:
        m = get_voicemark()
        t = torch.from_numpy(wav).float().to(m["dev"])[None, None]
        with torch.no_grad():
            _, chunk_logits = m["solver"].model.detect_watermark(t, return_logits=True)
        chunk_probs = torch.softmax(chunk_logits, dim=-1)[0].cpu().numpy()
        vals = np.argmax(chunk_probs, axis=1)
        hard = np.asarray(
            [(int(v) >> j) & 1 for v in vals for j in range(4)], dtype=np.int8
        )
        evidence = _chunk_logits_to_bit_evidence(chunk_probs, bit_order="lsb")
    assert hard.shape == (16,) and evidence.shape == (16,)
    return hard, evidence.astype(np.float64)


def valid_pair(model: str, item: dict, max_attempts: int = 200) -> dict:
    """Choose a deterministic one-bit pair whose two marked endpoints decode exactly."""
    bit = int(item["flipped_bit"])
    member_index = int(item["coalition_member_index"])
    speaker, clip_slot = item["speaker"], int(item["clip_slot"])
    original_base = int(item["base_payload"])
    rng = np.random.default_rng(SCHEDULE_SEED + 1000003 * int(item["trial_id"]) + 97)

    for attempt in range(1, max_attempts + 1):
        base = original_base if attempt == 1 else int(rng.integers(0, 1 << NBITS[model]))
        flipped = int(base ^ (1 << bit))
        others = {
            int(v) for j, v in enumerate(item["coalition_payloads"])
            if j != member_index
        }
        if base in others or flipped in others:
            continue
        wav0 = get_or_embed(model, speaker, base, clip_slot)
        wav1 = get_or_embed(model, speaker, flipped, clip_slot)
        hard0, _ = decode_with_evidence(model, wav0)
        hard1, _ = decode_with_evidence(model, wav1)
        bits0 = int_to_bits(base, NBITS[model])
        bits1 = int_to_bits(flipped, NBITS[model])
        if np.array_equal(hard0, bits0) and np.array_equal(hard1, bits1):
            selected = dict(item)
            coalition = list(item["coalition_payloads"])
            coalition[member_index] = base
            selected.update({
                "coalition_payloads": coalition,
                "base_payload": base,
                "flipped_payload": flipped,
                "pair_attempts": attempt,
            })
            return selected
    raise RuntimeError(
        f"{model} trial={item['trial_id']}: no valid one-bit pair "
        f"after {max_attempts} attempts")


def logmel_residual_distance(r0: np.ndarray, r1: np.ndarray) -> float:
    kwargs = dict(sr=SR, n_fft=1024, win_length=1024, hop_length=256,
                  n_mels=80, fmin=0.0, fmax=8000.0, power=2.0)
    m0 = librosa.feature.melspectrogram(y=r0, **kwargs)
    m1 = librosa.feature.melspectrogram(y=r1, **kwargs)
    l0 = np.log10(m0 + 1e-12)
    l1 = np.log10(m1 + 1e-12)
    return float(np.mean((l1 - l0) ** 2))


def band_energies(delta: np.ndarray) -> tuple[list[float], list[float], float]:
    """One-sided Parseval decomposition of mean-square delta energy."""
    n = len(delta)
    spec = np.fft.rfft(delta.astype(np.float64))
    power = np.abs(spec) ** 2 / (n * n)
    if n % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    freq = np.fft.rfftfreq(n, d=1.0 / SR)
    energy = []
    for lo, hi in BANDS:
        keep = (freq >= lo) & (freq < hi if hi < SR / 2 else freq <= hi)
        energy.append(float(power[keep].sum()))
    total = float(np.mean(delta.astype(np.float64) ** 2))
    fractions = [e / total if total > 0 else float("nan") for e in energy]
    return energy, fractions, float(sum(energy) - total)


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_rows(path: Path, model: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [int(r["trial_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate trial_id in {path}")
    if any(r["model"] != model or int(r["k"]) != COALITION_SIZE for r in rows):
        raise ValueError(f"model/k mismatch in {path}")
    return rows


def analyse_trial(model: str, item: dict) -> dict:
    speaker = item["speaker"]
    clip_slot = int(item["clip_slot"])
    base = item["base_payload"]
    flipped = item["flipped_payload"]
    clean = load_clean(speaker, clip_slot)
    wav0 = get_or_embed(model, speaker, base, clip_slot)
    wav1 = get_or_embed(model, speaker, flipped, clip_slot)
    n = min(len(clean), len(wav0), len(wav1))
    if n < SR:
        raise ValueError(f"trial {item['trial_id']} shorter than one second: n={n}")
    clean = np.asarray(clean[:n], dtype=np.float32)
    wav0 = np.asarray(wav0[:n], dtype=np.float32)
    wav1 = np.asarray(wav1[:n], dtype=np.float32)
    if not (np.isfinite(clean).all() and np.isfinite(wav0).all() and np.isfinite(wav1).all()):
        raise ValueError(f"non-finite waveform at trial {item['trial_id']}")

    residual0 = wav0 - clean
    residual1 = wav1 - clean
    delta = wav1 - wav0
    waveform_mse = float(np.mean(delta.astype(np.float64) ** 2))
    mel_distance = logmel_residual_distance(residual0, residual1)
    energy, fractions, _ = band_energies(delta)

    hard0, evidence0 = decode_with_evidence(model, wav0)
    hard1, evidence1 = decode_with_evidence(model, wav1)
    bits0 = int_to_bits(base, NBITS[model])
    bits1 = int_to_bits(flipped, NBITS[model])
    bit = item["flipped_bit"]
    direction = 2 * int(bits1[bit]) - 1
    unchanged = np.ones(16, dtype=bool)
    unchanged[bit] = False

    values = {
        "model": model,
        "k": COALITION_SIZE,
        "trial_id": item["trial_id"],
        "speaker": speaker,
        "clip_index": source_record(speaker, clip_slot)["clip_index"],
        "source_path": source_record(speaker, clip_slot)["path"],
        "coalition_payloads": json.dumps(
            item["coalition_payloads"], separators=(",", ":")),
        "coalition_member_index": item["coalition_member_index"],
        "base_payload": base,
        "flipped_payload": flipped,
        "flipped_bit": bit,
        "pair_attempts": item["pair_attempts"],
        "sample_count": n,
        "waveform_mse": waveform_mse,
        "mel_distance": mel_distance,
        "band_energy_0_1k": energy[0],
        "band_energy_1_2k": energy[1],
        "band_energy_2_4k": energy[2],
        "band_energy_4_8k": energy[3],
        "band_fraction_0_1k": fractions[0],
        "band_fraction_1_2k": fractions[1],
        "band_fraction_2_4k": fractions[2],
        "band_fraction_4_8k": fractions[3],
        "base_decoded_payload": bits_to_int(hard0),
        "flipped_decoded_payload": bits_to_int(hard1),
        "base_payload_correct": int(np.array_equal(hard0, bits0)),
        "flipped_payload_correct": int(np.array_equal(hard1, bits1)),
        "base_changed_bit_correct": int(hard0[bit] == bits0[bit]),
        "flipped_changed_bit_correct": int(hard1[bit] == bits1[bit]),
        "decoded_bit_changes": int(np.count_nonzero(hard0 != hard1)),
        "base_changed_bit_score": float(evidence0[bit]),
        "flipped_changed_bit_score": float(evidence1[bit]),
        "changed_bit_score_change": float(
            direction * (evidence1[bit] - evidence0[bit])),
        "unchanged_bit_score_change": float(
            np.mean(np.abs(evidence1[unchanged] - evidence0[unchanged]))),
        "base_bit_scores": json.dumps(evidence0.tolist(), separators=(",", ":")),
        "flipped_bit_scores": json.dumps(evidence1.tolist(), separators=(",", ":")),
    }
    return values


def main() -> None:
    args = parse_args()
    if not 0 <= args.trial_start < args.trial_end <= N_TRIALS:
        raise ValueError("require 0 <= trial-start < trial-end <= 300")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model
    partial = args.output_dir / f"{stem}.partial.csv"
    final = args.output_dir / f"{stem}.csv"
    source = final if final.exists() else partial
    rows = load_rows(source, args.model)
    done = {int(r["trial_id"]) for r in rows}
    plan = schedule()

    requested = range(args.trial_start, args.trial_end)
    completed_since_save = 0
    for trial_id in requested:
        if trial_id in done:
            continue
        item = valid_pair(args.model, plan[trial_id])
        row = analyse_trial(args.model, item)
        if not (row["base_payload_correct"] and row["flipped_payload_correct"]):
            raise RuntimeError(f"endpoint validation failed after selection: trial={trial_id}")
        rows.append(row)
        done.add(trial_id)
        completed_since_save += 1
        if completed_since_save >= args.checkpoint_every:
            rows.sort(key=lambda r: int(r["trial_id"]))
            atomic_write_csv(partial, rows)
            print(f"checkpoint model={args.model} completed={len(rows)} last={trial_id}", flush=True)
            completed_since_save = 0

    rows.sort(key=lambda r: int(r["trial_id"]))
    requested_ids = set(requested)
    if requested_ids.issubset(done):
        atomic_write_csv(partial, rows)
    if args.trial_start == 0 and args.trial_end == N_TRIALS and set(range(N_TRIALS)).issubset(done):
        if len(rows) != N_TRIALS:
            raise ValueError(f"expected exactly {N_TRIALS} rows, found {len(rows)}")
        atomic_write_csv(final, rows)
        print(f"complete model={args.model} rows={len(rows)} output={final}", flush=True)
    else:
        print(f"range complete model={args.model} rows={len(rows)} output={partial}", flush=True)


if __name__ == "__main__":
    main()

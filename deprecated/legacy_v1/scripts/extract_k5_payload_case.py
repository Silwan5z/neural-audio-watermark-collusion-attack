#!/usr/bin/env python3
"""Extract one audited K=5 equal-mean payload/confidence case per watermark."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from watermarks import extract_evidence, get_wavmark  # noqa: E402

MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
NBITS = {"audioseal": 16, "wavmark": 16, "timbrewm": 10,
         "voicemark": 16, "wmcodec": 16}
OUT = ROOT / "data" / "k5_payload_case_trial000_20260829"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--trial-id", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=OUT)
    return p.parse_args()


def load_wav(path: Path) -> np.ndarray:
    sr, wav = wavfile.read(path)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz, got {sr}: {path}")
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        wav = wav.astype(np.float32) / max(abs(info.min), info.max)
    return wav.astype(np.float32, copy=False)


def int_bits(value: int, d: int) -> list[int]:
    return [(value >> bit) & 1 for bit in range(d)]


def bits_int(bits: list[int]) -> int:
    return int(sum(int(value) << bit for bit, value in enumerate(bits)))


def wavmark_vote_probability(wav: np.ndarray) -> tuple[np.ndarray, int, int]:
    """P(bit=1) from official valid-window votes; not a neural posterior."""
    import torch
    from wavmark.utils import wm_add_util

    model = get_wavmark()
    start = np.asarray(wm_add_util.fix_pattern[:16], dtype=np.int8)
    windows = np.stack([wav[pos*800:pos*800+16000]
                        for pos in range((len(wav)-16000)//800)])
    decoded = []
    batch = int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE", "400"))
    for offset in range(0, len(windows), batch):
        with torch.no_grad():
            bits = (model["model"].decode(
                torch.from_numpy(windows[offset:offset+batch]).to(model["dev"])) >= .5)
        decoded.append(bits.int().cpu().numpy())
    decoded = np.concatenate(decoded)
    keep = np.all(decoded[:, :16] == start[None, :], axis=1)
    if not keep.any():
        raise RuntimeError("WavMark: no start-pattern-valid window")
    return decoded[keep, 16:32].mean(axis=0), int(keep.sum()), int(len(decoded))


def main() -> None:
    a = args(); d = NBITS[a.model]
    source = ROOT / "data" / "frequency_temporal_k5_20260828" / f"frequency_temporal_k5_{a.model}.csv"
    with source.open(newline="", encoding="utf-8") as f:
        candidates = [row for row in csv.DictReader(f)
                      if int(row["trial_id"]) == a.trial_id and row["condition"] == "full_waveform"]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one full_waveform row: {a.model}, trial {a.trial_id}")
    row = candidates[0]
    payloads = [int(x) for x in json.loads(row["coalition_payloads"])]
    coalition_bits = np.asarray([int_bits(x, d) for x in payloads], dtype=np.int8)
    native_hard = [int(x) for x in json.loads(row["decoded_payload_bits"])]
    if len(payloads) != 5 or len(native_hard) != d:
        raise ValueError("invalid coalition or decoded payload dimensions")
    wav = load_wav(Path(row["output_audio_path"]))

    confidence_kind = "native bit posterior"
    extra = {}
    if a.model == "wavmark":
        p1, valid_windows, total_windows = wavmark_vote_probability(wav)
        confidence_kind = "valid-window vote fraction (confidence proxy; WavMark has no official soft posterior)"
        extra = {"valid_start_pattern_windows": valid_windows,
                 "total_sliding_windows": total_windows}
    else:
        evidence = np.asarray(extract_evidence(a.model, wav), dtype=float)
        p1 = (evidence + 1.0) / 2.0
    if p1.shape != (d,) or not np.isfinite(p1).all():
        raise ValueError(f"invalid evidence for {a.model}: {p1.shape}")

    bit_rows = []
    for bit in range(d):
        ones = int(coalition_bits[:, bit].sum())
        state = "unanimous_0" if ones == 0 else "unanimous_1" if ones == 5 else "disputed"
        hard = int(native_hard[bit])
        bit_rows.append({
            "bit_index_lsb_first": bit,
            "colluder_bits": coalition_bits[:, bit].astype(int).tolist(),
            "ones_among_5": ones,
            "agreement": state,
            "p_bit_1": float(p1[bit]),
            "native_decoded_bit": hard,
            "confidence_native_decoded_bit": float(p1[bit] if hard else 1.0-p1[bit]),
            "marginal_threshold_bit": int(p1[bit] >= .5),
            "native_vs_marginal_threshold_mismatch": int(hard != int(p1[bit] >= .5)),
        })

    decoded_identity = int(row["decoded_identity"])
    native_identity = bits_int(native_hard)
    if decoded_identity != native_identity:
        raise ValueError(f"decoded identity mismatch: CSV={decoded_identity}, bits={native_identity}")
    result = {
        "model": a.model, "K": 5, "trial_id": a.trial_id,
        "spk": row["spk"], "local_t": int(row["local_t"]),
        "condition": "full_waveform", "mixing": "equal waveform mean",
        "attack_weights": [0.2]*5, "payload_nbits": d,
        "payload_bit_order": "LSB-first (b0 is the least significant bit)",
        "coalition_payloads": payloads,
        "coalition_payload_bits": coalition_bits.astype(int).tolist(),
        "output_audio_path": row["output_audio_path"],
        "confidence_kind": confidence_kind,
        "decoded_identity": decoded_identity,
        "decoded_payload_bits": native_hard,
        "tracing_failure": int(row["tracing_failure"]),
        "NCA_bits": int(row["NCA_bits"]), "NCA": float(row["NCA"]),
        "presence": None if row["presence"] == "" else float(row["presence"]),
        "unanimous_0_bits": [x["bit_index_lsb_first"] for x in bit_rows if x["agreement"] == "unanimous_0"],
        "unanimous_1_bits": [x["bit_index_lsb_first"] for x in bit_rows if x["agreement"] == "unanimous_1"],
        "disputed_bits": [x["bit_index_lsb_first"] for x in bit_rows if x["agreement"] == "disputed"],
        "bit_results": bit_rows,
        **extra,
    }
    out = a.output_dir / "raw"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.model}_trial{a.trial_id:03d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"created {path}")


if __name__ == "__main__":
    main()

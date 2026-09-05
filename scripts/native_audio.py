"""Native-rate embedding and decoding helpers for K=8 experiments."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registry import full_registry_bits  # noqa: E402
from watermarks import (  # noqa: E402
    _chunk_logits_to_bit_evidence,
    detect_many,
    embed,
    extract_evidence,
    get_timbrewm,
    get_wavmark,
    get_wmcodec,
    resample_to,
)

NATIVE_SAMPLE_RATE = {
    "audioseal": 16000,
    "wavmark": 16000,
    "timbrewm": 22050,
    "voicemark": 16000,
    "wmcodec": 24000,
}


def bits_to_int(bits: np.ndarray) -> int:
    """Convert an LSB-first bit vector to its integer payload."""
    return int(sum(int(value) << index for index, value in enumerate(bits)))


def embed_native(model: str, clean_16khz: np.ndarray,
                 bits: list[int]) -> tuple[np.ndarray, int]:
    """Embed one payload at the model's native sample rate."""
    if model not in {"timbrewm", "wmcodec"}:
        return embed(model, clean_16khz, bits), 16000

    import torch

    if model == "timbrewm":
        loaded = get_timbrewm()
        clean = resample_to(clean_16khz, 16000, 22050)
        waveform = torch.from_numpy(clean).float().to(loaded["dev"])[None, None]
        message = torch.tensor(
            [[bits]], dtype=torch.float32, device=loaded["dev"]
        ) * 2 - 1
        with torch.no_grad():
            marked, _ = loaded["enc"].test_forward(waveform, message)
        return marked[0, 0].cpu().numpy().astype(np.float32), 22050

    loaded = get_wmcodec()
    clean = resample_to(clean_16khz, 16000, 24000)
    waveform = torch.from_numpy(clean).float().to(loaded["dev"])[None, None]
    digits = [int("".join(map(str, bits[index * 4:(index + 1) * 4])), 2)
              for index in range(4)]
    message = torch.tensor([digits], dtype=torch.long, device=loaded["dev"])
    with torch.no_grad():
        encoded_message = loaded["wm_enc"](message)
        encoded_audio = loaded["enc"](waveform, encoded_message)
        quantized, _, _ = loaded["quant"](encoded_audio)
        marked = loaded["gen"](quantized)
    return marked[0, 0].cpu().numpy().astype(np.float32), 24000


def wavmark_vote_probability(waveform: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return WavMark bit vote fractions from valid decoding windows."""
    import torch
    from wavmark.utils import wm_add_util

    loaded = get_wavmark()
    start_pattern = np.asarray(wm_add_util.fix_pattern[:16], dtype=np.int8)
    windows = np.stack([
        waveform[position * 800:position * 800 + 16000]
        for position in range((len(waveform) - 16000) // 800)
    ])
    decoded = []
    batch_size = int(os.environ.get("WAVMARK_WINDOW_BATCH_SIZE", "600"))
    for offset in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[offset:offset + batch_size]).to(
            loaded["dev"])
        with torch.no_grad():
            bits = loaded["model"].decode(batch) >= 0.5
        decoded.append(bits.int().cpu().numpy())
    decoded_bits = np.concatenate(decoded)
    valid = np.all(decoded_bits[:, :16] == start_pattern[None, :], axis=1)
    if not valid.any():
        raise RuntimeError("WavMark produced no valid decoding window")
    return (decoded_bits[valid, 16:32].mean(axis=0), int(valid.sum()),
            int(len(decoded_bits)))


def decode_native(model: str, waveform: np.ndarray, sample_rate: int):
    """Return hard bits, bit probabilities, presence, and decoder details."""
    import torch

    details: dict = {}
    if sample_rate != NATIVE_SAMPLE_RATE[model]:
        raise ValueError(
            f"{model} expects {NATIVE_SAMPLE_RATE[model]} Hz, got {sample_rate} Hz")

    if model == "timbrewm":
        loaded = get_timbrewm()
        tensor = torch.from_numpy(waveform).float().to(loaded["dev"])[None, None]
        with torch.no_grad():
            soft = loaded["dec"].test_forward(tensor)[0, 0]
        soft = soft.cpu().numpy()
        probability = 1.0 / (1.0 + np.exp(-soft))
        return (soft >= 0).astype(np.int8), probability, float("nan"), details

    if model == "wmcodec":
        from third_party.wmcodec.meldataset import mel_spectrogram

        loaded = get_wmcodec()
        config = loaded["h"]
        tensor = torch.from_numpy(waveform).float().to(loaded["dev"])[None]
        with torch.no_grad():
            mel = mel_spectrogram(
                tensor, config.n_fft, config.num_mels, config.sampling_rate,
                config.hop_size, config.win_size, config.fmin,
                config.fmax_for_loss,
            )
            scores, prediction = loaded["wm_dec"](mel)
        digit_probabilities = torch.softmax(
            torch.stack(scores, dim=1), dim=-1
        )[0].cpu().numpy()
        digits = prediction[0].cpu().numpy().astype(int).tolist()
        hard_bits = []
        for value in digits:
            hard_bits.extend([(value >> (3 - index)) & 1 for index in range(4)])
        evidence = _chunk_logits_to_bit_evidence(
            digit_probabilities, bit_order="msb")
        details = {
            "decoded_digits": digits,
            "digit_probabilities": digit_probabilities.tolist(),
            "digit_confidences": [
                float(digit_probabilities[index, digits[index]])
                for index in range(4)
            ],
        }
        return (np.asarray(hard_bits, dtype=np.int8), (evidence + 1.0) / 2.0,
                float("nan"), details)

    registry = full_registry_bits(model)
    _, presence, hard = detect_many(model, [waveform], registry)[0]
    if hard is None:
        return None, None, presence, details
    if model == "wavmark":
        probability, valid, total = wavmark_vote_probability(waveform)
        details = {
            "valid_start_pattern_windows": valid,
            "total_sliding_windows": total,
        }
    else:
        probability = (
            np.asarray(extract_evidence(model, waveform), dtype=float) + 1.0
        ) / 2.0
    return (np.asarray(hard, dtype=np.int8),
            np.asarray(probability, dtype=float), presence, details)

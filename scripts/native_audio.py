"""Native-rate embedding, caching, and decoding helpers."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registry import (  # noqa: E402
    NBITS, full_registry_bits, get_or_embed, int_to_bits, load_clean,
    source_record,
)
from watermarks import (  # noqa: E402
    _chunk_logits_to_bit_evidence,
    _loglik_bits,
    _loglik_chunks,
    detect,
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
NATIVE_CACHE_DIR = ROOT / "cache" / "marked_native"


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


def native_cache_path(model: str, speaker: str, clip_slot: int,
                      payload: int) -> Path:
    clip_index = int(source_record(speaker, clip_slot)["clip_index"])
    return (NATIVE_CACHE_DIR / model / speaker / f"clip_{clip_index:02d}" /
            f"{payload}.wav")


def get_or_embed_native(model: str, speaker: str, payload: int,
                        clip_slot: int = 0) -> tuple[np.ndarray, int]:
    """Load or create a marked copy without a post-embedding rate conversion."""
    sample_rate = NATIVE_SAMPLE_RATE[model]
    if sample_rate == 16000:
        return (np.asarray(
            get_or_embed(model, speaker, payload, clip_slot), dtype=np.float32),
            sample_rate)

    import soundfile as sf

    path = native_cache_path(model, speaker, clip_slot, payload)
    if path.exists():
        try:
            cached, observed_rate = sf.read(str(path), dtype="float32")
            if (observed_rate == sample_rate and len(cached) >= sample_rate
                    and np.isfinite(cached).all()):
                return np.asarray(cached, dtype=np.float32), sample_rate
        except RuntimeError:
            pass

    clean = np.asarray(load_clean(speaker, clip_slot), dtype=np.float32)
    bits = int_to_bits(payload, NBITS[model]).astype(int).tolist()
    waveform, observed_rate = embed_native(model, clean, bits)
    if observed_rate != sample_rate:
        raise RuntimeError(
            f"{model}: expected {sample_rate} Hz embedding, got {observed_rate} Hz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{payload}.{uuid.uuid4().hex}.wav"
    sf.write(str(temporary), waveform, sample_rate, subtype="FLOAT")
    os.replace(temporary, path)
    return waveform, sample_rate


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


def detect_native(model: str, waveform: np.ndarray, sample_rate: int,
                  registry_bits: np.ndarray | None = None):
    """Return registry scores, presence, and hard bits at the native rate."""
    registry = (full_registry_bits(model)
                if registry_bits is None else registry_bits)
    if sample_rate == 16000:
        return detect(model, np.asarray(waveform, dtype=np.float32), registry)

    hard, probability, presence, details = decode_native(
        model, np.asarray(waveform, dtype=np.float32), sample_rate)
    if hard is None or probability is None:
        return np.zeros(len(registry)), presence, None
    if model == "timbrewm":
        scores = _loglik_bits(np.asarray(probability), registry)
    elif model == "wmcodec":
        scores = _loglik_chunks(
            np.asarray(details["digit_probabilities"]), registry,
            nchunk=4, bit_order="msb")
    else:
        raise RuntimeError(f"unsupported non-16-kHz model: {model}")
    return scores, presence, hard


def detect_many_native(model: str, waveforms: list[np.ndarray],
                       sample_rate: int,
                       registry_bits: np.ndarray | None = None):
    """Decode a waveform batch without converting away from the native rate."""
    registry = (full_registry_bits(model)
                if registry_bits is None else registry_bits)
    if sample_rate == 16000:
        return detect_many(model, waveforms, registry)
    return [detect_native(model, waveform, sample_rate, registry)
            for waveform in waveforms]


def bit_probabilities_native(model: str, waveform: np.ndarray,
                             sample_rate: int) -> tuple[np.ndarray, str]:
    """Return decoded-one probabilities used by confidence screening."""
    _, probability, _, _ = decode_native(
        model, np.asarray(waveform, dtype=np.float32), sample_rate)
    if probability is None:
        raise RuntimeError(f"missing soft evidence for {model}")
    source = ("valid-window vote fraction" if model == "wavmark"
              else "native decoder bit marginal")
    return np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0), source

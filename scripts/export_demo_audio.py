#!/usr/bin/env python3
"""Export the five-system trial-150 listening demo from validated caches."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from watermarks import resample_to  # noqa: E402
from run_alignment_stress_test import shift_fixed_length  # noqa: E402
from run_codec_stress_test import independently_code  # noqa: E402


EXAMPLES = {
    "audioseal": (16000, 24199, 31849, "marked"),
    "wavmark": (16000, 24199, 31849, "marked"),
    "timbrewm": (22050, 378, 497, "marked_native"),
    "voicemark": (16000, 24199, 31849, "marked"),
    "wmcodec": (24000, 24199, 31849, "marked_native"),
}

AUDIOSEAL_COALITIONS = {
    2: [24199, 31849],
    3: [52255, 17777, 62914],
    5: [40973, 50527, 54258, 54171, 40406],
    8: [2322, 3134, 14407, 21174, 50027, 50834, 51485, 62890],
}
OFFSETS_MS = (-50, -20, 0, 20, 50)
CODECS = ("none", "mp3_128k", "opus_64k")


def load_mono(path: Path, expected_rate: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1 or rate != expected_rate:
        raise ValueError(
            f"{path}: expected mono {expected_rate} Hz audio, got "
            f"shape={audio.shape}, rate={rate}")
    if len(audio) != expected_rate * 10:
        raise ValueError(f"{path}: expected ten seconds, got {len(audio)} samples")
    return audio


def write_demo(path: Path, audio: np.ndarray) -> None:
    audio = np.asarray(audio[:160000], dtype=np.float32)
    if len(audio) != 160000 or not np.isfinite(audio).all():
        raise ValueError(f"invalid exported audio: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 16000, subtype="PCM_16")


def write_mp3(wav_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(wav_path.with_suffix(".mp3")),
    ], check=True)


def write_pair(path: Path, audio: np.ndarray) -> None:
    write_demo(path, audio)
    write_mp3(path)


def load_audioseal_members(cache_root: Path,
                           payloads: list[int]) -> list[np.ndarray]:
    directory = cache_root / "marked" / "audioseal" / \
        "english:103" / "clip_01"
    return [load_mono(directory / f"{payload}.wav", 16000)
            for payload in payloads]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "demos")
    args = parser.parse_args()

    source = ROOT / "dataset" / "collusion_300" / "english" / "103" / \
        "english_103_01.wav"
    shutil.copyfile(source, args.output / "source_reference.wav")
    write_mp3(args.output / "source_reference.wav")

    for model, (rate, payload_a, payload_b, cache_name) in EXAMPLES.items():
        cache_dir = args.cache_root / cache_name / model / "english:103" / "clip_01"
        member_a = load_mono(cache_dir / f"{payload_a}.wav", rate)
        member_b = load_mono(cache_dir / f"{payload_b}.wav", rate)
        average = ((member_a.astype(np.float64) + member_b) / 2).astype(np.float32)

        if rate != 16000:
            member_a = resample_to(member_a, rate, 16000)
            member_b = resample_to(member_b, rate, 16000)
            average = resample_to(average, rate, 16000)

        directory = args.output / model
        write_pair(directory / f"member_{payload_a}.wav", member_a)
        write_pair(directory / f"member_{payload_b}.wav", member_b)
        write_pair(directory / "uniform_average.wav", average)
        print(f"exported {model}: {payload_a} + {payload_b}")

    condition_root = args.output / "conditions"
    coalition_signals = {}
    for k, payloads in AUDIOSEAL_COALITIONS.items():
        members = load_audioseal_members(args.cache_root, payloads)
        signal = np.mean(np.stack(members), axis=0,
                         dtype=np.float64).astype(np.float32)
        coalition_signals[k] = signal
        write_pair(condition_root / "coalition_size" / f"k{k}.wav", signal)

    k5_members = load_audioseal_members(
        args.cache_root, AUDIOSEAL_COALITIONS[5])
    for offset_ms in OFFSETS_MS:
        members = list(k5_members)
        if offset_ms:
            members[0] = shift_fixed_length(
                members[0], int(round(16000 * offset_ms / 1000)))
        signal = np.mean(np.stack(members), axis=0,
                         dtype=np.float64).astype(np.float32)
        label = (f"minus{abs(offset_ms)}" if offset_ms < 0 else
                 f"plus{offset_ms}" if offset_ms > 0 else "aligned")
        write_pair(condition_root / "offset" / f"{label}.wav", signal)

    for codec in CODECS:
        transformed = independently_code(k5_members, 16000, codec)
        signal = np.mean(np.stack(transformed), axis=0,
                         dtype=np.float64).astype(np.float32)
        write_pair(condition_root / "codec" / f"{codec}.wav", signal)
    print("exported AudioSeal coalition-size, offset, and codec comparisons")


if __name__ == "__main__":
    main()

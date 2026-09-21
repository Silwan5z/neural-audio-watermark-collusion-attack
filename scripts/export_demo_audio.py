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


EXAMPLES = {
    "audioseal": (16000, 24199, 31849, "marked"),
    "wavmark": (16000, 24199, 31849, "marked"),
    "timbrewm": (22050, 378, 497, "marked_native"),
    "voicemark": (16000, 24199, 31849, "marked"),
    "wmcodec": (24000, 24199, 31849, "marked_native"),
}


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
        write_demo(directory / f"member_{payload_a}.wav", member_a)
        write_demo(directory / f"member_{payload_b}.wav", member_b)
        write_demo(directory / "uniform_average.wav", average)
        write_mp3(directory / f"member_{payload_a}.wav")
        write_mp3(directory / f"member_{payload_b}.wav")
        write_mp3(directory / "uniform_average.wav")
        print(f"exported {model}: {payload_a} + {payload_b}")


if __name__ == "__main__":
    main()

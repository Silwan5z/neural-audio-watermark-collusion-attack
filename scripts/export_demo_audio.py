#!/usr/bin/env python3
"""Export complete trial-150 listening comparisons for the web demo."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from native_audio import NATIVE_SAMPLE_RATE, get_or_embed_native  # noqa: E402
from run_alignment_stress_test import SHIFTS_MS, shift_fixed_length  # noqa: E402
from run_codec_stress_test import CODECS, independently_code  # noqa: E402
from watermarks import resample_to, si_sdr  # noqa: E402


MODELS = ("audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec")
K_VALUES = (2, 3, 5, 8)
TRIAL_ID = 150
SPEAKER = "english:103"
CLIP_SLOT = 0


def coalitions_for(model: str) -> dict[int, list[int]]:
    with (ROOT / "data" / "supplementary" / "quality" /
          "all_trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    coalitions: dict[int, list[int]] = {}
    for k in (2, 3, 5):
        row = next(
            item for item in rows
            if item["model"] == model and int(item["k"]) == k
            and int(item["trial_id"]) == TRIAL_ID)
        coalitions[k] = [int(value) for value in
                         json.loads(row["coalition_payloads"])]
    record = json.loads((
        ROOT / "data" / "average" / "k8" / model /
        f"trial_{TRIAL_ID}.json").read_text(encoding="utf-8"))
    coalitions[8] = [int(value) for value in record["coalition_payloads"]]
    return coalitions


def load_members(model: str, coalition: list[int]) -> list[np.ndarray]:
    members = [
        get_or_embed_native(model, SPEAKER, payload, CLIP_SLOT)[0]
        for payload in coalition
    ]
    n = min(map(len, members))
    return [np.asarray(member[:n], dtype=np.float32) for member in members]


def mean_signal(members: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(members), axis=0,
                   dtype=np.float64).astype(np.float32)


def native_conditions(model: str) -> tuple[dict[int, list[int]], dict, float]:
    """Build every native-rate signal shown for one system."""
    coalitions = coalitions_for(model)
    sizes: dict[int, np.ndarray] = {}
    members_by_k: dict[int, list[np.ndarray]] = {}
    for k in K_VALUES:
        members = load_members(model, coalitions[k])
        members_by_k[k] = members
        sizes[k] = mean_signal(members)

    k5_members = members_by_k[5]
    shifted_index = TRIAL_ID % 5
    offsets: dict[int, np.ndarray] = {}
    for shift_ms in SHIFTS_MS:
        members = list(k5_members)
        if shift_ms:
            samples = int(round(
                NATIVE_SAMPLE_RATE[model] * shift_ms / 1000.0))
            members[shifted_index] = shift_fixed_length(
                members[shifted_index], samples)
        offsets[shift_ms] = mean_signal(members)

    codecs: dict[str, np.ndarray] = {}
    for codec in CODECS:
        codecs[codec] = mean_signal(independently_code(
            k5_members, NATIVE_SAMPLE_RATE[model], codec))
    reference_16 = browser_audio(model, members_by_k[8][0])
    k8_si_sdr = float(si_sdr(reference_16, browser_audio(model, sizes[8])))
    return coalitions, {
        "coalition_size": sizes,
        "offset": offsets,
        "codec": codecs,
    }, k8_si_sdr


def released_metrics(model: str, family: str, condition: str,
                     k8_si_sdr: float) -> tuple[float, float, float]:
    """Read the metrics for the exact released trial shown in the demo."""
    if family == "coalition_size":
        k = int(condition.split("=")[1])
        if k == 8:
            record = json.loads((
                ROOT / "data" / "average" / "k8" / model /
                f"trial_{TRIAL_ID}.json").read_text(encoding="utf-8"))
            return float(record["pesq"]), float(record["stoi"]), k8_si_sdr
        path = ROOT / "data" / "average" / f"k{k}" / f"{model}.csv"
        match = {"trial_id": str(TRIAL_ID)}
    elif family == "offset":
        path = ROOT / "data" / "supplementary" / "alignment" / "all_trials.csv"
        match = {"trial_id": str(TRIAL_ID), "model": model,
                 "shift_ms": condition}
    elif family == "codec":
        path = ROOT / "data" / "supplementary" / "codec" / "all_trials.csv"
        match = {"trial_id": str(TRIAL_ID), "model": model,
                 "codec": condition}
    else:
        raise ValueError(f"unknown demo family: {family}")
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(item for item in csv.DictReader(handle)
                   if all(item[key] == value for key, value in match.items()))
    return float(row["pesq"]), float(row["stoi"]), float(row["si_sdr"])


def browser_audio(model: str, waveform: np.ndarray) -> np.ndarray:
    rate = NATIVE_SAMPLE_RATE[model]
    output = (waveform if rate == 16000 else
              resample_to(waveform, rate, 16000))
    output = np.asarray(output[:160000], dtype=np.float32)
    if len(output) < 160000:
        output = np.pad(output, (0, 160000 - len(output)))
    if len(output) != 160000 or not np.isfinite(output).all():
        raise ValueError(f"invalid browser audio for {model}")
    return output


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 16000, subtype="PCM_16")


def write_mp3(wav_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(wav_path.with_suffix(".mp3")),
    ], check=True)


def write_pair(path: Path, audio: np.ndarray) -> None:
    write_wav(path, audio)
    write_mp3(path)


def offset_slug(value: int) -> str:
    if value < 0:
        return f"minus{abs(value)}"
    if value > 0:
        return f"plus{value}"
    return "aligned"


def atomic_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "family", "condition", "system", "k", "coalition_payloads",
        "audio_path", "evidence_path", "pesq", "stoi", "si_sdr_db",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=list(MODELS))
    parser.add_argument("--output", type=Path, default=ROOT / "demos")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "dataset" / "collusion_300" / "english" / "103" / \
        "english_103_01.wav"
    shutil.copyfile(source, args.output / "source_reference.wav")
    write_mp3(args.output / "source_reference.wav")

    rows = []
    metadata_path = args.output / "conditions" / "metadata.csv"
    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["system"] not in set(args.models)]

    for model in args.models:
        coalitions, conditions, k8_si_sdr = native_conditions(model)
        for k, waveform in conditions["coalition_size"].items():
            relative = Path("conditions") / "coalition_size" / model / f"k{k}.wav"
            write_pair(args.output / relative, browser_audio(model, waveform))
            metrics = released_metrics(
                model, "coalition_size", f"K={k}", k8_si_sdr)
            rows.append({
                "family": "coalition_size", "condition": f"K={k}",
                "system": model, "k": k,
                "coalition_payloads": json.dumps(coalitions[k]),
                "audio_path": relative.as_posix(),
                "evidence_path": (
                    f"data/average/k{k}/{model}.csv" if k < 8 else
                    f"data/average/k8/{model}/trial_{TRIAL_ID}.json"),
                "pesq": f"{metrics[0]:.6f}",
                "stoi": f"{metrics[1]:.6f}",
                "si_sdr_db": f"{metrics[2]:.6f}",
            })
        for shift_ms, waveform in conditions["offset"].items():
            relative = Path("conditions") / "offset" / model / \
                f"{offset_slug(shift_ms)}.wav"
            write_pair(args.output / relative, browser_audio(model, waveform))
            metrics = released_metrics(
                model, "offset", str(shift_ms), k8_si_sdr)
            rows.append({
                "family": "offset", "condition": str(shift_ms),
                "system": model, "k": 5,
                "coalition_payloads": json.dumps(coalitions[5]),
                "audio_path": relative.as_posix(),
                "evidence_path": "data/supplementary/alignment/all_trials.csv",
                "pesq": f"{metrics[0]:.6f}",
                "stoi": f"{metrics[1]:.6f}",
                "si_sdr_db": f"{metrics[2]:.6f}",
            })
        for codec, waveform in conditions["codec"].items():
            relative = Path("conditions") / "codec" / model / f"{codec}.wav"
            write_pair(args.output / relative, browser_audio(model, waveform))
            metrics = released_metrics(
                model, "codec", codec, k8_si_sdr)
            rows.append({
                "family": "codec", "condition": codec,
                "system": model, "k": 5,
                "coalition_payloads": json.dumps(coalitions[5]),
                "audio_path": relative.as_posix(),
                "evidence_path": "data/supplementary/codec/all_trials.csv",
                "pesq": f"{metrics[0]:.6f}",
                "stoi": f"{metrics[1]:.6f}",
                "si_sdr_db": f"{metrics[2]:.6f}",
            })
        print(f"exported {model}: 4 coalition sizes, 7 offsets, 3 codecs",
              flush=True)

    order = {model: index for index, model in enumerate(MODELS)}
    family_order = {"coalition_size": 0, "offset": 1, "codec": 2}
    rows.sort(key=lambda row: (
        order[row["system"]], family_order[row["family"]], row["condition"]))
    atomic_csv(metadata_path, rows)
    print(json.dumps({"models": args.models, "conditions": len(rows),
                      "metadata": str(metadata_path)}), flush=True)


if __name__ == "__main__":
    main()

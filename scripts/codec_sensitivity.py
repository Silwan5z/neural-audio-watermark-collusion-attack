"""Independent pre-collusion codec sensitivity for mean and FWP (K=5)."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (coalition_seed, full_registry_bits, get_or_embed,  # noqa: E402
                      sample_coalition, speaker_trial_index)
from watermarks import detect_many, pesq_wb, si_sdr, stoi  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
K = 5
CODECS = ["none", "mp3_128k", "opus_64k"]
FIELDS = [
    "model", "K", "trial_id", "spk", "local_t", "method", "codec",
    "codec_setting", "ASR", "attribution_margin", "PESQ", "STOI", "SI_SDR",
]


def write_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    expected = len(CODECS) * 2
    completed = {trial for trial, rr in grouped.items() if len(rr) == expected}
    return [r for r in rows if int(r["trial_id"]) in completed], completed


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {error}")


def codec_roundtrip(wav: np.ndarray, codec: str) -> np.ndarray:
    if codec == "none":
        return wav.copy()
    # File containers preserve encoder-delay / skip-sample metadata.  A raw
    # pipe round-trip silently drops that metadata for MP3 and introduces a
    # spurious temporal shift, confounding codec and misalignment effects.
    with tempfile.TemporaryDirectory(prefix="collusion_codec_") as temp_dir:
        work = Path(temp_dir)
        source = work / "input.wav"
        encoded = work / ("encoded.mp3" if codec == "mp3_128k" else "encoded.ogg")
        decoded = work / "decoded.wav"
        sf.write(source, wav, 16000, subtype="FLOAT")
        if codec == "mp3_128k":
            codec_args = ["-c:a", "libmp3lame", "-b:a", "128k"]
        elif codec == "opus_64k":
            codec_args = ["-c:a", "libopus", "-b:a", "64k"]
        else:
            raise ValueError(codec)
        run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-threads", "1", "-i", str(source), *codec_args, str(encoded)])
        run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-threads", "1", "-i", str(encoded), "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_f32le", str(decoded)])
        out, sample_rate = sf.read(decoded, dtype="float32")
        if sample_rate != 16000:
            raise RuntimeError(f"unexpected decoded sample rate: {sample_rate}")
    if len(out) < len(wav):
        out = np.pad(out, (0, len(wav) - len(out)))
    return out[:len(wav)].astype(np.float32)


def independently_code(wavs: list[np.ndarray], codec: str) -> list[np.ndarray]:
    if codec == "none":
        return [wav.copy() for wav in wavs]
    with ThreadPoolExecutor(max_workers=len(wavs)) as pool:
        return list(pool.map(lambda wav: codec_roundtrip(wav, codec), wavs))


def fwp(wavs: list[np.ndarray]) -> np.ndarray:
    best_pair, best_dist = (0, 1), -np.inf
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            dist = float(np.mean((wavs[i] - wavs[j]) ** 2))
            if dist > best_dist:
                best_pair, best_dist = (i, j), dist
    weights = np.zeros(len(wavs))
    weights[list(best_pair)] = 0.5
    return weights


def attack_metrics(scores: np.ndarray, coalition: list[int]) -> tuple[int, float]:
    coll = np.asarray(coalition, dtype=np.int64)
    top1 = int(np.argsort(scores, kind="stable")[::-1][0])
    coll_max = float(np.max(scores[coll]))
    mask = np.ones(len(scores), dtype=bool)
    mask[coll] = False
    non_max = float(np.max(scores[mask]))
    return int(top1 not in set(coalition)), non_max - coll_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"])
    parser.add_argument("--n_trials", type=int, default=300)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"codec_sensitivity_{args.model}_K5.csv"
    partial = RESULTS / f"codec_sensitivity_{args.model}_K5.partial.csv"
    rows, completed = load_completed(partial)
    trials = speaker_trial_index(n_total=args.n_trials)
    registry = full_registry_bits(args.model)
    start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coalition = sample_coalition(rng, args.model, K)
        wavs = [get_or_embed(args.model, spk, identity) for identity in coalition]
        length = min(map(len, wavs))
        wavs = [wav[:length] for wav in wavs]
        reference = wavs[0]
        specs, signals = [], []
        for codec in CODECS:
            transformed = independently_code(wavs, codec)
            methods = [("mean", np.full(K, 1 / K)), ("fwp", fwp(transformed))]
            for method, weights in methods:
                signal = sum(weights[i] * transformed[i] for i in range(K)).astype(np.float32)
                specs.append((codec, method))
                signals.append(signal)
        decoded = detect_many(args.model, signals, registry)
        for (codec, method), signal, (scores, _, _) in zip(specs, signals, decoded):
            asr, margin = attack_metrics(scores, coalition)
            setting = {"none": "none", "mp3_128k": "MP3 128 kbps",
                       "opus_64k": "Opus 64 kbps"}[codec]
            rows.append({
                "model": args.model, "K": K, "trial_id": trial_id, "spk": spk,
                "local_t": local_t, "method": method, "codec": codec,
                "codec_setting": setting, "ASR": asr,
                "attribution_margin": f"{margin:.8f}",
                "PESQ": f"{pesq_wb(reference, signal):.4f}",
                "STOI": f"{stoi(reference, signal):.4f}",
                "SI_SDR": f"{si_sdr(reference, signal):.2f}",
            })
        if (trial_id + 1) % 10 == 0:
            write_atomic(partial, rows)
            print(f"{args.model} codec: {trial_id + 1}/{len(trials)} "
                  f"({time.time() - start:.0f}s)", flush=True)

    write_atomic(partial, rows)
    write_atomic(out, rows)
    print(f"completed {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

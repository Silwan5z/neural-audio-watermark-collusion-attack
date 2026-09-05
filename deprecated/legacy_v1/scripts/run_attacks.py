"""Run single-copy watermark evasion baselines.

Every trial starts from exactly one legitimately watermarked audio signal.  The
attacked signal is generated once and the same detector output is restricted to
both the native registry and the matched N=1024 registry.  There is no coalition
size K in sampling, attack parameters, metrics, or filenames.

Supported evasion attacks:
  overwrite_same   same-model re-embedding with a different payload
  overwrite_cross  re-embedding by a different watermarking model
  encodec          EnCodec reconstruction
  vocoder          Vocos resynthesis
  hsja             HopSkipJump black-box removal
  fgsm             one-step white-box removal

Example:
  python scripts/run_attacks.py --attack fgsm --model audioseal --n_trials 5

Output:
  results/evaluation/attack_evasion_{attack}_{model}[__to_{cross_model}].csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from registry import (  # noqa: E402
    NBITS, full_registry_bits, full_registry_size, get_or_embed, int_to_bits,
    speaker_trial_index,
)
from registry_size_control import active_registries  # noqa: E402
from watermarks import detect, pesq_wb, si_sdr, stoi  # noqa: E402

RESULTS = ROOT / "results" / "evaluation"
DEVICE = os.environ.get("WATERMARK_DEVICE", "cuda:0")
MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
ATTACKS = ["overwrite_same", "overwrite_cross", "encodec", "vocoder", "hsja", "fgsm"]

FIELDS = [
    "model", "attack", "mode", "trial_id", "spk", "local_t", "seed",
    "source_id", "source_bits", "N_registry", "top1_identity", "source_rank",
    "source_top1", "identity_escape", "presence", "presence_removed",
    "decoded_bits", "source_bit_accuracy", "attacker_model", "attacker_target_id",
    "attacker_bits", "attacker_top1", "attacker_target_hit",
    "attacker_bit_accuracy", "attacker_presence", "PESQ", "STOI", "SI_SDR",
    "SNR", "query_count", "attack_parameters", "elapsed_sec",
]


def single_sample_seed(spk: str, model: str, local_t: int) -> int:
    """Stable seed independent of coalition size and registry size."""
    material = f"single-copy-evasion-v1|{spk}|{model}|{local_t}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def bits_to_int(bits: np.ndarray) -> int:
    return sum(int(bit) << i for i, bit in enumerate(bits))


def format_float(value: float, digits: int) -> str:
    if np.isnan(value):
        return ""
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.{digits}f}"


def snr(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    clean = ref[:n].astype(np.float64)
    noise = deg[:n].astype(np.float64) - clean
    signal_power = float(np.dot(clean, clean))
    noise_power = float(np.dot(noise, noise))
    if noise_power < 1e-12:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_completed(path: Path, expected_rows: int) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    completed = {
        trial_id for trial_id, trial_rows in grouped.items()
        if len(trial_rows) == expected_rows
    }
    kept = [row for row in rows if int(row["trial_id"]) in completed]
    return kept, completed


def parse_registry_sizes(spec: str, full_size: int) -> list[int]:
    sizes = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        size = full_size if token == "native" else int(token)
        if size < 1 or size > full_size:
            raise ValueError(f"registry size {size} is outside [1, {full_size}]")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one registry size is required")
    return sorted(set(sizes))


def restricted_source_metrics(scores: np.ndarray, active: np.ndarray,
                              source_id: int) -> dict[str, int]:
    active_scores = np.asarray(scores)[active]
    order = np.argsort(active_scores, kind="stable")[::-1]
    ranked = active[order]
    source_score = float(scores[source_id])
    source_rank = 1 + int(np.sum(active_scores > source_score))
    top1 = int(ranked[0])
    return {
        "top1_identity": top1,
        "source_rank": source_rank,
        "source_top1": int(top1 == source_id),
        "identity_escape": int(top1 != source_id),
    }


def decode_auxiliary(model: str, audio: np.ndarray, target_bits: np.ndarray) -> dict:
    registry = full_registry_bits(model)
    scores, presence, hard = detect(model, audio.astype(np.float32), registry)
    top1 = int(np.argsort(scores, kind="stable")[-1])
    target_id = bits_to_int(target_bits)
    bit_accuracy = ""
    if hard is not None:
        bit_accuracy = format_float(float(np.mean(np.asarray(hard) == target_bits)), 4)
    return {
        "attacker_model": model,
        "attacker_target_id": target_id,
        "attacker_bits": json.dumps(target_bits.astype(int).tolist()),
        "attacker_top1": top1,
        "attacker_target_hit": int(top1 == target_id),
        "attacker_bit_accuracy": bit_accuracy,
        "attacker_presence": format_float(float(presence), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=ATTACKS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--output_tag", default="",
                        help="Optional filename tag, intended for smoke tests or shards.")
    parser.add_argument("--registry_sizes", default="1024,native",
                        help="Comma-separated candidate counts; 'native' means full codebook.")
    parser.add_argument("--cross_model", choices=MODELS, default=None)
    parser.add_argument("--bandwidth", type=float, default=24.0,
                        choices=[1.5, 3.0, 6.0, 12.0, 24.0])
    parser.add_argument("--hsja_max_iter", type=int, default=64)
    parser.add_argument("--hsja_max_eval", type=int, default=3000)
    parser.add_argument("--hsja_init_eval", type=int, default=100)
    parser.add_argument("--fgsm_epsilon", type=float, default=0.01)
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n_trials must be positive")
    if args.attack == "overwrite_cross" and args.cross_model == args.model:
        parser.error("--cross_model must differ from --model")

    model = args.model
    attack_name = args.attack
    bit_count = NBITS[model]
    full_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)
    try:
        registry_sizes = parse_registry_sizes(args.registry_sizes, full_size)
    except ValueError as exc:
        parser.error(str(exc))

    cross_model = args.cross_model
    if attack_name == "overwrite_cross" and cross_model is None:
        cross_model = MODELS[(MODELS.index(model) + 1) % len(MODELS)]

    suffix = f"__to_{cross_model}" if attack_name == "overwrite_cross" else ""
    if args.output_tag:
        suffix += f".{args.output_tag}"
    stem = f"attack_evasion_{attack_name}_{model}{suffix}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / f"{stem}.csv"
    partial_path = RESULTS / f"{stem}.partial.csv"
    rows, completed = load_completed(partial_path, len(registry_sizes))
    if completed:
        print(f"resuming {stem}: {len(completed)}/{args.n_trials}", flush=True)

    from attacks import (  # noqa: E402
        encodec_reconstruct, fgsm, hsja, overwrite_cross, overwrite_same,
        vocoder_resynth,
    )

    trials = speaker_trial_index(n_total=args.n_trials)
    failures: list[tuple[int, str]] = []
    run_start = time.time()

    for trial_id, (spk, local_t) in enumerate(trials):
        if trial_id in completed:
            continue
        trial_start = time.time()
        seed = single_sample_seed(spk, model, local_t)
        rng = np.random.default_rng(seed)
        source_id = int(rng.integers(0, full_size))
        source_bits = int_to_bits(source_id, bit_count)
        reference = get_or_embed(model, spk, source_id).astype(np.float32)
        attacker_bits = None
        auxiliary = {
            "attacker_model": "", "attacker_target_id": "", "attacker_bits": "",
            "attacker_top1": "", "attacker_target_hit": "",
            "attacker_bit_accuracy": "", "attacker_presence": "",
        }
        query_count = ""
        parameters: dict[str, object] = {}

        try:
            if attack_name == "overwrite_same":
                attacked, attacker_bits = overwrite_same(
                    reference, model, original_bits=source_bits, rng=rng)
                auxiliary = decode_auxiliary(model, attacked, attacker_bits)
                parameters = {"payload_policy": "random_distinct"}
            elif attack_name == "overwrite_cross":
                attacked, attacker_bits = overwrite_cross(
                    reference, model, cross_model, rng=rng)
                auxiliary = decode_auxiliary(cross_model, attacked, attacker_bits)
                parameters = {"cross_model": cross_model}
            elif attack_name == "encodec":
                attacked = encodec_reconstruct(
                    reference, bandwidth=args.bandwidth, device=DEVICE)
                parameters = {"bandwidth_kbps": args.bandwidth}
            elif attack_name == "vocoder":
                attacked = vocoder_resynth(reference, device=DEVICE)
                parameters = {"vocoder": "charactr/vocos-mel-24khz"}
            elif attack_name == "hsja":
                attacked, info = hsja(
                    reference, model, source_bits, mode="removal",
                    max_iter=args.hsja_max_iter, max_eval=args.hsja_max_eval,
                    init_eval=args.hsja_init_eval, device=DEVICE)
                query_count = int(info["query_count"])
                parameters = {
                    "max_iter": args.hsja_max_iter,
                    "max_eval": args.hsja_max_eval,
                    "init_eval": args.hsja_init_eval,
                }
            elif attack_name == "fgsm":
                attacked, _ = fgsm(
                    reference, model, source_bits, mode="removal",
                    epsilon=args.fgsm_epsilon, device=DEVICE)
                parameters = {"epsilon_relative_peak": args.fgsm_epsilon, "steps": 1}
            else:  # pragma: no cover - argparse enforces the choices
                raise AssertionError(attack_name)
        except Exception as exc:
            failures.append((trial_id, f"{type(exc).__name__}: {exc}"))
            print(f"{stem}: trial={trial_id} FAILED: {failures[-1][1]}", flush=True)
            continue

        attacked = np.asarray(attacked, dtype=np.float32)
        scores, presence, hard = detect(model, attacked, registry_bits)
        required_ids = [source_id]
        if attack_name == "overwrite_same" and attacker_bits is not None:
            required_ids.append(bits_to_int(attacker_bits))
        registries = active_registries(
            full_size, required_ids, registry_sizes, spk, 1, local_t)

        source_bit_accuracy = ""
        decoded_bits = ""
        if hard is not None:
            hard_array = np.asarray(hard, dtype=np.int8)
            source_bit_accuracy = format_float(
                float(np.mean(hard_array == source_bits)), 4)
            decoded_bits = json.dumps(hard_array.astype(int).tolist())

        quality = {
            "PESQ": format_float(float(pesq_wb(reference, attacked)), 4),
            "STOI": format_float(float(stoi(reference, attacked)), 4),
            "SI_SDR": format_float(float(si_sdr(reference, attacked)), 2),
            "SNR": format_float(float(snr(reference, attacked)), 2),
        }
        elapsed = time.time() - trial_start
        for registry_size in registry_sizes:
            identity = restricted_source_metrics(
                scores, registries[registry_size], source_id)
            rows.append({
                "model": model,
                "attack": attack_name,
                "mode": "evasion",
                "trial_id": trial_id,
                "spk": spk,
                "local_t": local_t,
                "seed": seed,
                "source_id": source_id,
                "source_bits": json.dumps(source_bits.astype(int).tolist()),
                "N_registry": registry_size,
                **identity,
                "presence": format_float(float(presence), 6),
                "presence_removed": "" if np.isnan(presence) else int(presence < 0.5),
                "decoded_bits": decoded_bits,
                "source_bit_accuracy": source_bit_accuracy,
                **auxiliary,
                **quality,
                "query_count": query_count,
                "attack_parameters": json.dumps(parameters, sort_keys=True),
                "elapsed_sec": f"{elapsed:.3f}",
            })

        write_rows_atomic(partial_path, rows)
        if (trial_id + 1) % 10 == 0 or args.n_trials <= 5:
            print(
                f"{stem}: {trial_id + 1}/{args.n_trials} "
                f"elapsed={time.time() - run_start:.0f}s",
                flush=True,
            )

    write_rows_atomic(partial_path, rows)
    write_rows_atomic(output_path, rows)
    successful_trials = len({int(row["trial_id"]) for row in rows})
    print(
        f"completed {output_path} trials={successful_trials}/{args.n_trials} "
        f"rows={len(rows)} failures={len(failures)}",
        flush=True,
    )
    if failures:
        for trial_id, message in failures:
            print(f"failure trial={trial_id}: {message}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

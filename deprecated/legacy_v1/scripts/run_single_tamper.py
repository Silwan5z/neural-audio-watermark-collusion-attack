"""Single-copy targeted identity-tampering baselines.

Each trial starts from one watermarked source copy.  The candidate targets are
the ten most reachable non-source identities in the matched N=1024 registry,
ranked with the same softmin-v2 bit-margin score specialized to K=1.  Each
target is attacked independently and success is reported both per target and
as the number of successful targets among the selected ten.  The attacked
waveform is evaluated against both N=1024 and the native registry.
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
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from registry import (  # noqa: E402
    NBITS, full_registry_bits, full_registry_size, get_or_embed, int_to_bits,
    speaker_trial_index,
)
from registry_size_control import active_registries  # noqa: E402
from watermarks import (  # noqa: E402
    detect_many, detect_wavmark_many, get_wavmark, pesq_wb, si_sdr, stoi,
)

RESULTS = ROOT / "results" / "evaluation"
DEVICE = os.environ.get("WATERMARK_DEVICE", "cuda:0")
MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
ATTACKS = ["overwrite", "ifgsm", "hsja"]
N_CAND = 10
BETA = 8.0
FIELDS = [
    "model", "attack", "mode", "trial_id", "spk", "local_t", "seed",
    "source_id", "target_id", "candidate_rank", "selection_N_registry",
    "reachability_score", "source_bits", "target_bits", "N_registry",
    "top1_identity", "source_rank", "target_rank", "source_top1", "target_hit",
    "successful_targets_in_top10", "any_target_hit",
    "presence", "decoded_bits", "source_bit_accuracy", "target_bit_accuracy",
    "quality_reference", "PESQ", "STOI", "SI_SDR", "SNR",
    "query_count", "attack_parameters",
    "elapsed_sec",
]


def tamper_seed(spk: str, model: str, local_t: int) -> int:
    material = f"single-copy-targeted-tamper-v2-top10|{spk}|{model}|{local_t}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


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


def parse_registry_sizes(spec: str, full_size: int) -> list[int]:
    sizes = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        size = full_size if token == "native" else int(token)
        if size < 2 or size > full_size:
            raise ValueError(f"registry size {size} is outside [2, {full_size}]")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one registry size is required")
    return sorted(set(sizes))


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
        trial for trial, trial_rows in grouped.items()
        if len(trial_rows) == expected_rows
    }
    return [r for r in rows if int(r["trial_id"]) in completed], completed


def ranking_metrics(scores: np.ndarray, active: np.ndarray,
                    source_id: int, target_id: int) -> dict[str, int]:
    active_scores = np.asarray(scores)[active]
    ranked = active[np.argsort(active_scores, kind="stable")[::-1]]
    source_rank = 1 + int(np.sum(active_scores > float(scores[source_id])))
    target_rank = 1 + int(np.sum(active_scores > float(scores[target_id])))
    top1 = int(ranked[0])
    return {
        "top1_identity": top1,
        "source_rank": source_rank,
        "target_rank": target_rank,
        "source_top1": int(top1 == source_id),
        "target_hit": int(top1 == target_id),
    }


def most_reachable_targets(source_bits: np.ndarray, candidate_ids: np.ndarray,
                           registry_bits: np.ndarray) -> list[tuple[int, float]]:
    """Rank K=1 targets by the softmin-v2 bit-margin reachability score."""
    target_bits = registry_bits[candidate_ids]
    margins = np.where(target_bits == source_bits[None, :], 0.5, -0.5)
    scores = -(1.0 / BETA) * logsumexp(-BETA * margins, axis=1)
    order = np.lexsort((candidate_ids, -scores))[:N_CAND]
    return [(int(candidate_ids[index]), float(scores[index])) for index in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=ATTACKS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--registry_sizes", default="1024,native")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_end", type=int, default=None)
    parser.add_argument("--output_tag", default="")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hsja_max_iter", type=int, default=1)
    parser.add_argument("--hsja_max_eval", type=int, default=20)
    parser.add_argument("--hsja_init_eval", type=int, default=10)
    parser.add_argument("--hsja_timeout", type=int, default=600)
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n_trials must be positive")
    trial_end = args.n_trials if args.trial_end is None else args.trial_end
    if not 0 <= args.trial_start < trial_end <= args.n_trials:
        parser.error("trial range must satisfy 0 <= trial_start < trial_end <= n_trials")
    if args.attack == "ifgsm" and args.model == "wavmark":
        parser.error("I-FGSM is unavailable for non-differentiable WavMark")

    full_size = full_registry_size(args.model)
    try:
        registry_sizes = parse_registry_sizes(args.registry_sizes, full_size)
    except ValueError as exc:
        parser.error(str(exc))

    tag = f".{args.output_tag}" if args.output_tag else ""
    stem = f"single_tamper_{args.attack}_{args.model}{tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / f"{stem}.csv"
    partial_path = RESULTS / f"{stem}.partial.csv"
    rows, completed = load_completed(
        partial_path, N_CAND * len(registry_sizes))

    from attacks import fgsm, hsja, overwrite_same  # noqa: E402

    registry_bits = full_registry_bits(args.model)
    bit_count = NBITS[args.model]
    failures = []
    run_start = time.time()

    for trial_id, (spk, local_t) in enumerate(
            speaker_trial_index(n_total=args.n_trials)):
        if trial_id < args.trial_start or trial_id >= trial_end:
            continue
        if trial_id in completed:
            continue
        start = time.time()
        seed = tamper_seed(spk, args.model, local_t)
        rng = np.random.default_rng(seed)
        source_id = int(rng.integers(full_size))
        source_bits = int_to_bits(source_id, bit_count)
        source_audio = get_or_embed(args.model, spk, source_id).astype(np.float32)
        selection_size = min(1024, full_size)
        registries = active_registries(
            full_size, [source_id], sorted(set([selection_size, *registry_sizes])),
            spk, 1, local_t)
        selection_registry = registries[selection_size]
        candidate_ids = selection_registry[selection_registry != source_id]
        selected = most_reachable_targets(
            source_bits, candidate_ids, registry_bits)

        attacks_for_trial = []
        for candidate_rank, (target_id, reachability_score) in enumerate(selected, 1):
            target_bits = int_to_bits(target_id, bit_count)
            query_count = ""
            try:
                if args.attack == "overwrite":
                    attacked, _ = overwrite_same(
                        source_audio, args.model, original_bits=source_bits,
                        target_bits=target_bits, rng=rng)
                    parameters = {"target_payload": "registered_target"}
                elif args.attack == "ifgsm":
                    attacked, info = fgsm(
                        source_audio, args.model, source_bits, mode="tamper",
                        target_bits=target_bits, epsilon=args.epsilon,
                        steps=args.steps, device=DEVICE)
                    parameters = {
                        "epsilon_relative_peak": args.epsilon,
                        "steps": args.steps,
                        "step_size": info["step_size"],
                    }
                else:
                    target_audio = get_or_embed(
                        args.model, spk, target_id).astype(np.float32)
                    length = min(len(source_audio), len(target_audio))
                    attacked, info = hsja(
                        source_audio[:length], args.model, source_bits,
                        mode="tamper", target_bits=target_bits,
                        target_audio=target_audio[:length],
                        max_iter=args.hsja_max_iter, max_eval=args.hsja_max_eval,
                        init_eval=args.hsja_init_eval, timeout_s=args.hsja_timeout,
                        device=DEVICE)
                    query_count = int(info["query_count"])
                    parameters = {
                        "max_iter": args.hsja_max_iter,
                        "max_eval": args.hsja_max_eval,
                        "init_eval": args.hsja_init_eval,
                    }
            except Exception as exc:
                failures.append((trial_id, candidate_rank,
                                 f"{type(exc).__name__}: {exc}"))
                print(f"{stem}: trial={trial_id} target_rank={candidate_rank} "
                      f"FAILED: {failures[-1][-1]}", flush=True)
                break
            attacks_for_trial.append({
                "candidate_rank": candidate_rank,
                "target_id": target_id,
                "target_bits": target_bits,
                "reachability_score": reachability_score,
                "attacked": np.asarray(attacked, dtype=np.float32),
                "query_count": query_count,
                "parameters": parameters,
            })
        if len(attacks_for_trial) != N_CAND:
            continue

        attacked_batch = [item["attacked"] for item in attacks_for_trial]
        if args.model == "wavmark":
            decoded = detect_wavmark_many(get_wavmark(), attacked_batch, registry_bits)
        else:
            decoded = detect_many(args.model, attacked_batch, registry_bits)

        trial_rows = []
        for item, (scores, presence, hard) in zip(attacks_for_trial, decoded):
            target_id = item["target_id"]
            target_bits = item["target_bits"]
            attacked = item["attacked"]
            decoded_bits = ""
            source_accuracy = ""
            target_accuracy = ""
            if hard is not None:
                hard_array = np.asarray(hard, dtype=np.int8)
                decoded_bits = json.dumps(hard_array.astype(int).tolist())
                source_accuracy = format_float(
                    float(np.mean(hard_array == source_bits)), 4)
                target_accuracy = format_float(
                    float(np.mean(hard_array == target_bits)), 4)
            quality = {
                "PESQ": format_float(float(pesq_wb(source_audio, attacked)), 4),
                "STOI": format_float(float(stoi(source_audio, attacked)), 4),
                "SI_SDR": format_float(float(si_sdr(source_audio, attacked)), 2),
                "SNR": format_float(float(snr(source_audio, attacked)), 2),
            }
            for size in registry_sizes:
                trial_rows.append({
                "model": args.model,
                "attack": args.attack,
                "mode": "targeted_single_copy_top10_reachable",
                "trial_id": trial_id,
                "spk": spk,
                "local_t": local_t,
                "seed": seed,
                "source_id": source_id,
                "target_id": target_id,
                "candidate_rank": item["candidate_rank"],
                "selection_N_registry": selection_size,
                "reachability_score": f"{item['reachability_score']:.8f}",
                "source_bits": json.dumps(source_bits.astype(int).tolist()),
                "target_bits": json.dumps(target_bits.astype(int).tolist()),
                "N_registry": size,
                **ranking_metrics(scores, registries[size], source_id, target_id),
                "successful_targets_in_top10": "",
                "any_target_hit": "",
                "presence": format_float(float(presence), 6),
                "decoded_bits": decoded_bits,
                "source_bit_accuracy": source_accuracy,
                "target_bit_accuracy": target_accuracy,
                "quality_reference": "source_watermarked_audio_before_tamper",
                **quality,
                "query_count": item["query_count"],
                "attack_parameters": json.dumps(item["parameters"], sort_keys=True),
                "elapsed_sec": "",
                })
        elapsed = time.time() - start
        hits_by_size = {
            size: sum(int(row["target_hit"]) for row in trial_rows
                      if int(row["N_registry"]) == size)
            for size in registry_sizes
        }
        for row in trial_rows:
            hits = hits_by_size[int(row["N_registry"])]
            row["successful_targets_in_top10"] = hits
            row["any_target_hit"] = int(hits > 0)
            row["elapsed_sec"] = f"{elapsed:.3f}"
        rows.extend(trial_rows)
        write_rows_atomic(partial_path, rows)
        print(
            f"{stem}: {trial_id + 1}/{args.n_trials} elapsed={elapsed:.1f}s",
            flush=True,
        )

    write_rows_atomic(partial_path, rows)
    write_rows_atomic(output_path, rows)
    print(
        f"completed {output_path} rows={len(rows)} "
        f"wall={time.time() - run_start:.1f}s failures={len(failures)}",
        flush=True,
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

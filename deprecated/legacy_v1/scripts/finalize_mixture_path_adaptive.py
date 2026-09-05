#!/usr/bin/env python3
"""Validate, merge, and summarize adaptive mixture-path shards."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "mixture_path_k5_adaptive_20260829"
MODELS = ("audioseal", "voicemark")
COARSE = set(range(0, 101, 10))
SEED = 20260829
BOOTSTRAP = 10000


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=DEFAULT)
    p.add_argument("--n-trials", type=int, default=300)
    p.add_argument("--num-shards", type=int, default=7)
    return p.parse_args()


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2 or np.max(y) - np.min(y) <= 1e-15:
        return float("nan")
    fit = np.polyval(np.polyfit(x, y, 1), x)
    ss_res = float(np.sum((y-fit)**2)); ss_tot = float(np.sum((y-y.mean())**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")


def monotone(y: np.ndarray, tol: float = 1e-6) -> bool:
    direction = float(y[-1] - y[0])
    diffs = np.diff(y)
    return bool(np.all(diffs >= -tol) if direction >= 0 else np.all(diffs <= tol))


def bootstrap_speaker(rows: list[dict], field: str) -> tuple[float, float, float]:
    by_spk: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if np.isfinite(value):
            by_spk[row["spk"]].append(value)
    speakers = sorted(by_spk)
    observed = float(np.mean([x for s in speakers for x in by_spk[s]]))
    rng = np.random.default_rng(SEED)
    reps = np.empty(BOOTSTRAP, dtype=np.float64)
    for b in range(BOOTSTRAP):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        reps[b] = np.mean([x for s in sampled for x in by_spk[s]])
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return observed, float(lo), float(hi)


def summarize_trial(rr: list[dict]) -> dict:
    rr = sorted(rr, key=lambda r: int(r["lambda_index"]))
    idx = np.array([int(r["lambda_index"]) for r in rr], dtype=int)
    lam = idx / 100.0
    probs = np.stack([json.loads(r["bit_probabilities"]) for r in rr]).astype(float)
    logits = np.stack([json.loads(r["bit_logits"]) for r in rr]).astype(float)
    hard = np.stack([json.loads(r["native_hard_bits"]) for r in rr]).astype(int)
    identities = np.array([int(r["decoded_identity"]) for r in rr], dtype=np.int64)
    target = int(rr[0]["flipped_bit"])
    # The adaptive points intentionally over-sample transition neighborhoods.
    # Use the uniform coarse grid for the primary path R^2, and report the
    # adaptive-grid result separately so local refinement does not reweight R^2.
    coarse = np.array([value in COARSE for value in idx], dtype=bool)
    prob_r2 = np.array([r2(lam[coarse], probs[coarse, bit])
                        for bit in range(probs.shape[1])])
    logit_r2 = np.array([r2(lam[coarse], logits[coarse, bit])
                         for bit in range(logits.shape[1])])
    prob_r2_adaptive = np.array([r2(lam, probs[:, bit])
                                 for bit in range(probs.shape[1])])
    logit_r2_adaptive = np.array([r2(lam, logits[:, bit])
                                  for bit in range(logits.shape[1])])
    mono = np.array([monotone(probs[:, bit]) for bit in range(probs.shape[1])], dtype=bool)
    endpoint_changing = hard[0] != hard[-1]
    transitions = []
    for pos in range(1, len(rr)):
        for bit in np.flatnonzero(hard[pos] != hard[pos-1]):
            transitions.append({"bit": int(bit), "lambda_low": float(lam[pos-1]),
                                "lambda_high": float(lam[pos]),
                                "from": int(hard[pos-1, bit]), "to": int(hard[pos, bit])})
    identity_transitions = [
        {"lambda_low": float(lam[pos-1]), "lambda_high": float(lam[pos]),
         "from": int(identities[pos-1]), "to": int(identities[pos])}
        for pos in range(1, len(rr)) if identities[pos] != identities[pos-1]
    ]
    target_transitions = [x for x in transitions if x["bit"] == target]
    return {
        "model": rr[0]["model"], "K": rr[0]["K"], "trial_id": rr[0]["trial_id"],
        "spk": rr[0]["spk"], "local_t": rr[0]["local_t"],
        "base_payload": rr[0]["base_payload"], "flipped_payload": rr[0]["flipped_payload"],
        "flipped_bit": target, "n_path_points": len(rr),
        "n_refined_intervals": len({r["refined_interval_low"] for r in rr
                                    if r["sampling_stage"] == "refined"}),
        "refinement_reason": next(r["refinement_reason"] for r in rr
                                  if r["sampling_stage"] == "refined"),
        "target_bit_probability_r2": float(prob_r2[target]),
        "target_bit_logit_r2": float(logit_r2[target]),
        "target_bit_probability_r2_adaptive": float(prob_r2_adaptive[target]),
        "target_bit_logit_r2_adaptive": float(logit_r2_adaptive[target]),
        "mean_bit_probability_r2": float(np.nanmean(prob_r2)),
        "mean_bit_logit_r2": float(np.nanmean(logit_r2)),
        "mean_bit_probability_r2_adaptive": float(np.nanmean(prob_r2_adaptive)),
        "mean_bit_logit_r2_adaptive": float(np.nanmean(logit_r2_adaptive)),
        "target_bit_monotonic": int(mono[target]),
        "monotonic_bit_fraction": float(mono.mean()),
        "endpoint_changing_bit_count": int(endpoint_changing.sum()),
        "endpoint_changing_monotonic_fraction": (
            float(mono[endpoint_changing].mean()) if endpoint_changing.any() else float("nan")),
        "target_transition_count": len(target_transitions),
        "target_transition_brackets": json.dumps(target_transitions, separators=(",", ":")),
        "all_bit_transitions": json.dumps(transitions, separators=(",", ":")),
        "identity_transition_count": len(identity_transitions),
        "identity_transitions": json.dumps(identity_transitions, separators=(",", ":")),
        "per_bit_probability_r2": json.dumps(prob_r2.tolist(), separators=(",", ":")),
        "per_bit_logit_r2": json.dumps(logit_r2.tolist(), separators=(",", ":")),
        "per_bit_probability_r2_adaptive": json.dumps(prob_r2_adaptive.tolist(), separators=(",", ":")),
        "per_bit_logit_r2_adaptive": json.dumps(logit_r2_adaptive.tolist(), separators=(",", ":")),
        "per_bit_monotonic": json.dumps(mono.astype(int).tolist(), separators=(",", ":")),
    }


def main() -> None:
    a = args(); point_rows = []
    expected_trials = set(range(a.n_trials))
    for model in MODELS:
        model_rows = []
        for shard in range(a.num_shards):
            path = a.input_dir / "shards" / f"mixture_path_{model}_shard{shard}of{a.num_shards}.partial.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8") as f:
                model_rows.extend(csv.DictReader(f))
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in model_rows:
            grouped[int(row["trial_id"])].append(row)
        if set(grouped) != expected_trials:
            raise ValueError(f"{model}: trial IDs mismatch ({len(grouped)}/{a.n_trials})")
        for trial, rr in grouped.items():
            idx = [int(r["lambda_index"]) for r in rr]
            if len(idx) != len(set(idx)) or not COARSE.issubset(idx):
                raise ValueError(f"{model} trial {trial}: duplicate/missing coarse points")
            if not any(r["sampling_stage"] == "refined" for r in rr):
                raise ValueError(f"{model} trial {trial}: no refined points")
            for row in rr:
                for key in ("bit_probabilities", "bit_logits", "native_hard_bits"):
                    values = np.asarray(json.loads(row[key]), dtype=float)
                    if values.shape != (16,) or not np.isfinite(values).all():
                        raise ValueError(f"{model} trial {trial}: invalid {key}")
                if not Path(row["audio_path"]).exists():
                    raise FileNotFoundError(row["audio_path"])
        model_rows.sort(key=lambda r: (int(r["trial_id"]), int(r["lambda_index"])))
        point_rows.extend(model_rows)
        atomic_csv(a.input_dir / f"mixture_path_{model}_points.csv",
                   list(model_rows[0]), model_rows)

    grouped_all: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in point_rows:
        grouped_all[(row["model"], int(row["trial_id"]))].append(row)
    summaries = [summarize_trial(rr) for rr in grouped_all.values()]
    summaries.sort(key=lambda r: (r["model"], int(r["trial_id"])))
    atomic_csv(a.input_dir / "mixture_path_trial_summary.csv", list(summaries[0]), summaries)

    system_rows = []
    metrics = ("target_bit_probability_r2", "target_bit_logit_r2",
               "target_bit_probability_r2_adaptive", "target_bit_logit_r2_adaptive",
               "target_bit_monotonic", "monotonic_bit_fraction",
               "endpoint_changing_monotonic_fraction")
    for model in MODELS:
        selected = [r for r in summaries if r["model"] == model]
        for metric in metrics:
            mean, lo, hi = bootstrap_speaker(selected, metric)
            system_rows.append({"model": model, "metric": metric, "mean": mean,
                                "ci95_low": lo, "ci95_high": hi,
                                "n_trials": len(selected), "n_speakers": len({r['spk'] for r in selected})})
    atomic_csv(a.input_dir / "mixture_path_system_summary.csv", list(system_rows[0]), system_rows)
    with (a.input_dir / "analysis_audit.json").open("w", encoding="utf-8") as f:
        json.dump({
            "experiment": "adaptive one-bit mixture path", "K": 5,
            "models": list(MODELS), "trials_per_model": a.n_trials,
            "coarse_lambda": [x/100 for x in sorted(COARSE)],
            "refinement": "split every target-bit-changing coarse interval into ten; fallback to largest target probability change",
            "r2_definition": "primary R2 uses the uniformly spaced 11-point coarse grid; adaptive-grid R2 is separately reported",
            "point_confidence_fields": ["bit_probabilities", "bit_logits", "identity_confidence",
                                        "identity_log_confidence", "identity_margin",
                                        "presence_probability", "presence_logits",
                                        "chunk_probabilities", "chunk_logits"],
            "audio": "lambda endpoints reference existing float audio; all interior mixtures stored as float32 WAV",
            "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAP,
            "bootstrap_unit": "speaker cluster (three trials)",
            "total_point_rows": len(point_rows),
        }, f, indent=2)
    (a.input_dir / ".complete").touch()
    print(f"complete: trials={len(summaries)} points={len(point_rows)} output={a.input_dir}")


if __name__ == "__main__":
    main()

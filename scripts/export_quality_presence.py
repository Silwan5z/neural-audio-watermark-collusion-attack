"""Create the unified mean/FWP quality and native-presence export.

Quality metrics come from the completed attack runs, where the reference is the
first legitimate watermarked coalition copy.  Native presence is joined from
the matched evidence-chain trial/output, so no neural inference is repeated.
"""
from __future__ import annotations

import csv
import os
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (coalition_seed, full_registry_bits, get_or_embed,  # noqa: E402
                      sample_coalition)
from watermarks import detect_many  # noqa: E402

MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
KS = [2, 3, 5, 8]
METHODS = {"mean", "fwp"}
PRESENCE_MODELS = {"audioseal", "voicemark", "wavmark"}
ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"
FIELDS = [
    "model", "K", "trial_id", "spk", "local_t", "method", "quality_reference",
    "PESQ", "STOI", "SI_SDR", "presence_supported", "presence_score",
    "presence_decision",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


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


def recompute_presence(model: str, K: int, spk: str, local_t: int) -> dict[str, float]:
    rng = np.random.default_rng(coalition_seed(spk, K, local_t))
    coalition = sample_coalition(rng, model, K)
    wavs = [get_or_embed(model, spk, identity) for identity in coalition]
    length = min(map(len, wavs))
    wavs = [wav[:length] for wav in wavs]
    methods = [("mean", np.full(K, 1 / K)), ("fwp", fwp(wavs))]
    outputs = [sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
               for _, a in methods]
    decoded = detect_many(model, outputs, full_registry_bits(model))
    return {method: float(result[1]) for (method, _), result in zip(methods, decoded)}


def main() -> None:
    for model in MODELS:
        for K in KS:
            attack = [r for r in read_csv(EVAL / f"attack_{model}_K{K}.csv")
                      if r["method"] in METHODS]
            evidence = [r for r in read_csv(EVAL / f"evidence_chain_{model}_K{K}.csv")
                        if r["method"] in METHODS]
            evidence_map = {(r["spk"], r["local_t"], r["method"]): r for r in evidence}
            backfill: dict[tuple[str, str], dict[str, float]] = {}
            rows = []
            for row in attack:
                key = (row["spk"], row["local_t"], row["method"])
                matched = evidence_map.get(key)
                supported = model in PRESENCE_MODELS
                score_text = matched["presence"] if supported and matched is not None else ""
                if supported and not score_text:
                    context = (row["spk"], row["local_t"])
                    if context not in backfill:
                        backfill[context] = recompute_presence(
                            model, K, row["spk"], int(row["local_t"]))
                    score_text = f"{backfill[context][row['method']]:.4f}"
                decision = "" if not score_text else int(float(score_text) >= 0.5)
                rows.append({
                    "model": model, "K": K, "trial_id": row["gi"],
                    "spk": row["spk"], "local_t": row["local_t"],
                    "method": row["method"],
                    "quality_reference": "first_legitimate_watermarked_coalition_copy",
                    "PESQ": row["PESQ"], "STOI": row["STOI"],
                    "SI_SDR": row["SI_SDR"], "presence_supported": int(supported),
                    "presence_score": score_text, "presence_decision": decision,
                })
            if len(rows) != 600:
                raise RuntimeError(f"expected 600 rows for {model} K={K}, got {len(rows)}")
            out = EVAL / f"quality_presence_{model}_K{K}.csv"
            write_atomic(out, rows)
            print(f"wrote {out.name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()

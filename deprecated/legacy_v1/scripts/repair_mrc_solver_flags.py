#!/usr/bin/env python3
"""Revalidate failed MRC solver flags without changing attack results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_mrc_ablation as runner  # noqa: E402
from registry import (NBITS, coalition_seed, int_to_bits, sample_coalition)  # noqa: E402


def write_atomic(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(runner.VARIANTS))
    args = parser.parse_args()
    config = runner.VARIANTS[args.variant]
    results = ROOT / "results" / "evaluation"
    paths = sorted(results.glob(f"mrc_ablation_{args.variant}_N1024_*.csv"))
    finals = [path for path in paths if ".partial." not in path.name]
    if len(finals) != 20:
        raise ValueError(f"expected 20 final CSVs, found {len(finals)}")

    audit: list[dict[str, object]] = []
    for final in finals:
        with final.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        changed = 0
        for row in rows:
            if str(row["solver_success"]).lower() in {"1", "1.0", "true"}:
                continue
            model, K = row["model"], int(row["K"])
            spk, local_t = row["spk"], int(row["local_t"])
            rng = np.random.default_rng(coalition_seed(spk, K, local_t))
            coalition = sample_coalition(rng, model, K)
            C = np.stack([int_to_bits(identity, NBITS[model]) for identity in coalition])
            keff_min = (None if config["keff_frac"] is None
                        else max(1.0, float(config["keff_frac"]) * K))
            weights, score, success = runner.optimize_softmin(
                C, int_to_bits(int(row["target"]), NBITS[model]),
                float(config["beta"]), keff_min, float(config["entropy"]))
            recorded = np.asarray(json.loads(row["selection_weights"]), dtype=np.float64)
            max_delta = float(np.max(np.abs(weights - recorded)))
            if not success or max_delta > 1e-10:
                raise RuntimeError(
                    f"failed solver revalidation: {final.name} gi={row['gi']} "
                    f"target={row['target']} success={success} max_delta={max_delta}")
            row["solver_success"] = "1"
            changed += 1
            audit.append({
                "file": final.name, "gi": int(row["gi"]),
                "target": int(row["target"]), "max_weight_delta": max_delta,
                "recomputed_score": score,
            })
        if changed:
            write_atomic(final, rows, fields)
            partial = final.with_name(final.name.replace(".csv", ".partial.csv"))
            if partial.exists():
                write_atomic(partial, rows, fields)

    audit_path = results / f"mrc_ablation_{args.variant}_solver_revalidation.json"
    audit_path.write_text(json.dumps({"variant": args.variant, "revalidated": audit},
                                     indent=2) + "\n")
    print(f"SOLVER_REVALIDATION_COMPLETE corrected={len(audit)} audit={audit_path}")


if __name__ == "__main__":
    main()

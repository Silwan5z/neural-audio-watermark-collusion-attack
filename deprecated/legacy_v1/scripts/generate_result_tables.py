#!/usr/bin/env python3
"""Generate publication-facing Markdown tables from completed evaluation CSVs.

The script never modifies raw experiment outputs.  It writes one Markdown
document under ``results/tables/`` and deliberately keeps zero-valued metrics,
ties, and every completed method row.  Within a fixed (model, K), every tied
best value is bolded: ASR/R3/R5/PESQ/STOI/SI-SDR are maximised, while NAC is
minimised.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"
OUT_DIR = ROOT / "results" / "tables"
MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
KS = [2, 3, 5, 8]
ATTACK_SOURCES = [
    ("attack", ["mean", "fwp"]),
    ("rp", ["rp"]),
    ("eep", ["eep"]),
    ("dm", ["dm"]),
    ("bdb", ["bdb"]),
    ("pgr", ["pgr"]),  # explicitly labelled oracle in the rendered table
    ("baselines", ["median", "minimum", "maximum", "rand_minmax", "copy_paste"]),
]
METHOD_LABELS = {
    "mean": "mean", "fwp": "FWP", "rp": "RP", "eep": "EEP", "dm": "DM",
    "bdb": "BDB", "pgr": "PGR (oracle)", "median": "median",
    "minimum": "minimum", "maximum": "maximum", "rand_minmax": "rand_minmax",
    "copy_paste": "copy_paste", "tct": "TCT",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric_mean(rows: list[dict[str, str]], column: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value in ("", "nan", "NaN", None):
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else None


def metric_cell(value: float | None, best: float | None) -> str:
    if value is None:
        return "—"
    text = f"{value:.3f}"  # Never turns 0 into an empty cell.
    if best is not None and np.isclose(value, best, rtol=0.0, atol=1e-12):
        return f"**{text}**"
    return text


def best_value(values: list[float | None], direction: str) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return max(present) if direction == "max" else min(present)


def collect_attack(model: str, k: int) -> list[dict]:
    grouped: list[dict] = []
    for prefix, methods in ATTACK_SOURCES:
        path = EVAL / f"{prefix}_{model}_K{k}.csv"
        if not path.exists():
            continue
        rows = read_rows(path)
        for method in methods:
            trial_rows = [row for row in rows if row.get("method") == method]
            # A complete method has 300 trials.  Partial files are never used.
            if len(trial_rows) != 300:
                continue
            grouped.append({
                "method": METHOD_LABELS[method],
                "ASR": numeric_mean(trial_rows, "ASR"),
                "NAC": numeric_mean(trial_rows, "ACC_near_norm"),
                "PESQ": numeric_mean(trial_rows, "PESQ"),
                "STOI": numeric_mean(trial_rows, "STOI"),
                "SI_SDR": numeric_mean(trial_rows, "SI_SDR"),
            })
    return grouped


def render_attack(lines: list[str]) -> None:
    lines.extend([
        "## Attack: ASR, NAC, and audio quality",
        "",
        "NAC is `ACC_near_norm`; lower is better. PGR is an oracle diagnostic and is shown separately within the same table for reference only.",
        "",
        "| Watermark | K | Method | ASR ↑ | NAC ↓ | PESQ ↑ | STOI ↑ | SI-SDR ↑ |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        for k in KS:
            rows = collect_attack(model, k)
            if not rows:
                continue
            best = {
                "ASR": best_value([row["ASR"] for row in rows], "max"),
                "NAC": best_value([row["NAC"] for row in rows], "min"),
                "PESQ": best_value([row["PESQ"] for row in rows], "max"),
                "STOI": best_value([row["STOI"] for row in rows], "max"),
                "SI_SDR": best_value([row["SI_SDR"] for row in rows], "max"),
            }
            for row in rows:
                lines.append(
                    f"| {model} | {k} | {row['method']} | "
                    f"{metric_cell(row['ASR'], best['ASR'])} | {metric_cell(row['NAC'], best['NAC'])} | "
                    f"{metric_cell(row['PESQ'], best['PESQ'])} | {metric_cell(row['STOI'], best['STOI'])} | "
                    f"{metric_cell(row['SI_SDR'], best['SI_SDR'])} |"
                )
    lines.append("")


def collect_framing(model: str, k: int) -> list[dict]:
    path = EVAL / f"tamper_{model}_K{k}.csv"
    if not path.exists():
        return []
    rows = read_rows(path)
    result: list[dict] = []
    for method in ("mean", "tct"):
        method_rows = [row for row in rows if row.get("method") == method]
        # 300 trials × 10 selected targets.
        if len(method_rows) != 3000:
            continue
        per_trial: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for row in method_rows:
            per_trial[(row["spk"], row["local_t"], row["gi"])].append(int(row["target_top1"]))
        if len(per_trial) != 300 or any(len(v) != 10 for v in per_trial.values()):
            continue
        # Uniformly sampling one of the 10 preselected targets versus allowing
        # the evaluator to take the easiest one of the same 10.
        random_success = float(np.mean([int(row["target_top1"]) for row in method_rows]))
        easiest_success = float(np.mean([max(hits) for hits in per_trial.values()]))
        result.append({"method": METHOD_LABELS[method], "random": random_success, "easiest": easiest_success})
    return result


def render_framing(lines: list[str]) -> None:
    lines.extend([
        "## Framing: target success",
        "",
        "`Random target` samples uniformly from the 10 preselected opportunistic targets; `any of 10` succeeds if at least one of those targets is hit. Both target sets are the same geometrically easiest 10 candidates per trial.",
        "",
        "| Watermark | K | Method | Random target success ↑ | Any of 10 success ↑ |",
        "|---|---:|---|---:|---:|",
    ])
    for model in MODELS:
        for k in KS:
            rows = collect_framing(model, k)
            if not rows:
                continue
            best_random = best_value([row["random"] for row in rows], "max")
            best_easiest = best_value([row["easiest"] for row in rows], "max")
            for row in rows:
                lines.append(
                    f"| {model} | {k} | {row['method']} | {metric_cell(row['random'], best_random)} | "
                    f"{metric_cell(row['easiest'], best_easiest)} |"
                )
    lines.append("")


def collect_ecc(model: str, k: int) -> list[dict]:
    path = EVAL / f"attack_ecc_{model}_K{k}.csv"
    if not path.exists():
        return []
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(path):
        groups[(row.get("scheme", ""), row.get("method", ""))].append(row)
    result: list[dict] = []
    for (scheme, method), rows in sorted(groups.items()):
        if len(rows) != 300:
            continue
        result.append({
            "method": f"{scheme}:{METHOD_LABELS.get(method, method)}",
            "ASR": numeric_mean(rows, "ASR"),
            "R3": numeric_mean(rows, "R3_escape"),
            "R5": numeric_mean(rows, "R5_escape"),
        })
    return result


def render_ecc(lines: list[str]) -> None:
    lines.extend([
        "## ECC: escape rates",
        "",
        "| Watermark | K | Method | ASR ↑ | R@3 escape ↑ | R@5 escape ↑ |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for model in MODELS:
        for k in KS:
            rows = collect_ecc(model, k)
            if not rows:
                continue
            best_asr = best_value([row["ASR"] for row in rows], "max")
            best_r3 = best_value([row["R3"] for row in rows], "max")
            best_r5 = best_value([row["R5"] for row in rows], "max")
            for row in rows:
                lines.append(
                    f"| {model} | {k} | {row['method']} | {metric_cell(row['ASR'], best_asr)} | "
                    f"{metric_cell(row['R3'], best_r3)} | {metric_cell(row['R5'], best_r5)} |"
                )
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_DIR / "result_tables.md")
    args = ap.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment result tables",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "All completed rows are retained. `**bold**` denotes every tie for the best value within the same watermark and K; numerical zero is rendered as `0.000`.",
        "",
    ]
    render_attack(lines)
    render_framing(lines)
    render_ecc(lines)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

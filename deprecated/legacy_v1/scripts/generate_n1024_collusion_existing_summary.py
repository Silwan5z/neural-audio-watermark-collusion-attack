#!/usr/bin/env python3
"""Summarize currently available N=1024 collusion results without rerunning models."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_CSV = DATA / "summaries" / "n1024_collusion_existing.csv"
OUT_MD = ROOT / "paper" / "identity_attribution_icassp_v8_source" / "N1024_COLLUSION_EXISTING_SUMMARY.md"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}
KS = [2, 3, 5, 8]
METHODS = ["fwp", "mean", "maximum"]
METHOD_LABEL = {"fwp": "FWP（本文）", "mean": "Mean", "maximum": "Maximum"}


def wilson(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def mean_or_nan(frame: pd.DataFrame, field: str) -> float:
    if field not in frame or frame[field].dropna().empty:
        return math.nan
    return float(frame[field].mean())


def matched_rows(model: str, k: int) -> pd.DataFrame:
    if model == "timbrewm":
        path = DATA / "registry_control" / f"registry_control_{model}_K{k}.csv"
    else:
        path = DATA / "registry_control" / f"registry_control_nested_{model}_K{k}.csv"
    frame = pd.read_csv(path)
    return frame[frame.N_registry.eq(1024)].copy()


def build() -> pd.DataFrame:
    records = []
    for model in MODELS:
        for k in KS:
            matched = matched_rows(model, k)
            attack = pd.read_csv(DATA / "attack" / f"attack_{model}_K{k}.csv")
            baselines = pd.read_csv(DATA / "baselines" / f"baselines_{model}_K{k}.csv")
            presence = pd.read_csv(DATA / "quality_presence" / f"quality_presence_{model}_K{k}.csv")
            for method in METHODS:
                source = attack[attack.method.eq(method)] if method != "maximum" else baselines[baselines.method.eq(method)]
                if len(source) != 300:
                    raise ValueError(f"expected 300 source rows: {model} K={k} {method}; got {len(source)}")

                if method in ("mean", "fwp"):
                    attr = matched[matched.method.eq(method)]
                    if len(attr) != 300:
                        raise ValueError(f"expected 300 matched rows: {model} K={k} {method}; got {len(attr)}")
                    asr_1024 = mean_or_nan(attr, "ASR")
                    r3_1024 = mean_or_nan(attr, "R3_escape")
                    r5_1024 = mean_or_nan(attr, "R5_escape")
                    margin = mean_or_nan(attr, "attribution_margin")
                    rank = mean_or_nan(attr, "best_colluder_rank")
                    lo, hi = wilson(float(attr.ASR.sum()), len(attr))
                    p = presence[presence.method.eq(method)]
                    presence_supported = bool(p.presence_supported.eq(1).all())
                    presence_rate = mean_or_nan(p, "presence_decision") if presence_supported else math.nan
                    note = "完整 N=1,024"
                elif model == "timbrewm":
                    # TimbreWM's complete native registry is exactly N=1,024.
                    asr_1024 = mean_or_nan(source, "ASR")
                    r3_1024 = mean_or_nan(source, "R3_escape")
                    r5_1024 = mean_or_nan(source, "R5_escape")
                    margin = rank = math.nan
                    lo, hi = wilson(float(source.ASR.sum()), len(source))
                    presence_supported = False
                    presence_rate = math.nan
                    note = "完整 N=1,024（等于 native）"
                else:
                    asr_1024 = r3_1024 = r5_1024 = margin = rank = lo = hi = math.nan
                    presence_supported = False
                    presence_rate = math.nan
                    note = "Maximum 未保存 N=1,024 identity scores；仅有 native ASR/NAC/音质"

                records.append({
                    "model": model, "system": LABEL[model], "K": k,
                    "method": method, "method_label": METHOD_LABEL[method], "n_trials": 300,
                    "N_registry": 1024, "ASR_1024": asr_1024,
                    "ASR_1024_wilson95_low": lo, "ASR_1024_wilson95_high": hi,
                    "R3_escape_1024": r3_1024, "R5_escape_1024": r5_1024,
                    "attribution_margin_1024": margin, "best_colluder_rank_1024": rank,
                    "ASR_native": mean_or_nan(source, "ASR"),
                    "NAC": mean_or_nan(source, "ACC_near_norm"),
                    "AggResid": mean_or_nan(source, "AggResid"),
                    "PESQ": mean_or_nan(source, "PESQ"), "STOI": mean_or_nan(source, "STOI"),
                    "SI_SDR": mean_or_nan(source, "SI_SDR"),
                    "presence_supported": presence_supported,
                    "presence_decision_rate": presence_rate, "availability_note": note,
                })
    return pd.DataFrame(records)


def pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{100 * value:.2f}"


def num(value: float, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def write_markdown(frame: pd.DataFrame) -> None:
    lines = [
        "# N=1,024 合谋攻击已有结果汇总",
        "",
        "仅使用现有最终数据，不包含任何新补跑。ASR/R3/R5 为纠正后的 common-prefix matched registry 结果；NAC 与音质来自相同攻击波形，因此不随注册库大小变化。",
        "",
        "## 1. ASR 快速对比（%）",
        "",
        "| 水印系统 | 方法 | K=2 | K=3 | K=5 | K=8 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for method in METHODS:
            q = frame[(frame.model == model) & (frame.method == method)].set_index("K")
            cells = [pct(q.loc[k, "ASR_1024"]) for k in KS]
            lines.append(f"| {LABEL[model]} | {METHOD_LABEL[method]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 2. 全部 N=1,024 指标",
        "",
        "| 水印系统 | K | 方法 | ASR | 95% CI | R3 escape | R5 escape | NAC ↓ | Attr. margin ↑ | Best colluder rank ↑ | PESQ ↑ | STOI ↑ | SI-SDR ↑ | Presence |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        ci = "—" if pd.isna(row.ASR_1024) else f"[{100*row.ASR_1024_wilson95_low:.2f}, {100*row.ASR_1024_wilson95_high:.2f}]"
        presence = pct(row.presence_decision_rate) if row.presence_supported else "—"
        lines.append(
            f"| {row.system} | {row.K} | {row.method_label} | {pct(row.ASR_1024)} | {ci} | "
            f"{pct(row.R3_escape_1024)} | {pct(row.R5_escape_1024)} | {num(row.NAC)} | "
            f"{num(row.attribution_margin_1024)} | {num(row.best_colluder_rank_1024, 2)} | "
            f"{num(row.PESQ)} | {num(row.STOI, 4)} | {num(row.SI_SDR, 2)} | {presence} |"
        )

    lines += [
        "",
        "## 3. 跨 K 汇总",
        "",
        "每个方法–系统对四个等规模 K 单元直接汇总，共 1,200 trials；Maximum 的 N=1,024 ASR 仅 TimbreWM 可汇总。",
        "",
        "| 水印系统 | 方法 | ASR@1024 | ASR native | NAC ↓ | PESQ ↑ | STOI ↑ | SI-SDR ↑ | Presence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for method in METHODS:
            q = frame[(frame.model == model) & (frame.method == method)]
            presence = q.presence_decision_rate.mean() if q.presence_supported.all() else math.nan
            lines.append(
                f"| {LABEL[model]} | {METHOD_LABEL[method]} | {pct(q.ASR_1024.mean())} | "
                f"{pct(q.ASR_native.mean())} | {num(q.NAC.mean())} | {num(q.PESQ.mean())} | "
                f"{num(q.STOI.mean(), 4)} | {num(q.SI_SDR.mean(), 2)} | {pct(presence)} |"
            )

    lines += [
        "",
        "## 4. 数据可用性",
        "",
        "- Mean/FWP：5 个系统 × 4 个 K，N=1,024 完整。",
        "- Maximum：TimbreWM 的 native 注册库就是 N=1,024，因此 4 个 K 完整。AudioSeal、WavMark、VoiceMark、WMCodec 只有 native ASR/NAC/音质，缺少可离线截取的 identity score，未填充 N=1,024 ASR。",
        "- Presence：现有 presence 数据只覆盖 Mean/FWP；TimbreWM、WMCodec 接口不提供 presence。",
        "",
        f"逐行机器可读数据：`{OUT_CSV.relative_to(ROOT)}`。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)
    write_markdown(frame)
    print(f"wrote {OUT_CSV} rows={len(frame)}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

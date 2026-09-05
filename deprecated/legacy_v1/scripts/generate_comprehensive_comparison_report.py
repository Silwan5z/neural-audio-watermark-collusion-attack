#!/usr/bin/env python3
"""Generate a data-dense comparison report from the canonical ``data/`` tree."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "paper" / "identity_attribution_icassp_v8_source" / "EXPERIMENT_COMPARISON_SUMMARY.md"
MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}
KS = [2, 3, 5, 8]

METHODS = [
    ("fwp", "FWP（本文）", "本文：盲波形选择", "attack", "attack"),
    ("mean", "Mean", "经典平均", "attack", "attack"),
    ("rp", "RP", "盲配对对照", "rp", "rp"),
    ("eep", "EEP", "盲能量配对对照", "eep", "eep"),
    ("dm", "DM", "盲波形优化", "dm", "dm"),
    ("bdb", "BDB", "盲波形优化", "bdb", "bdb"),
    ("median", "Median", "经典逐采样点方法", "baselines", "baselines"),
    ("minimum", "Minimum", "经典逐采样点方法", "baselines", "baselines"),
    ("maximum", "Maximum", "经典逐采样点方法", "baselines", "baselines"),
    ("rand_minmax", "Random min-max", "经典逐采样点变体", "baselines", "baselines"),
    ("copy_paste", "Copy-paste", "信号域对照", "baselines", "baselines"),
    ("pgr", "PGR（payload oracle）", "特权 oracle", "pgr", "pgr"),
]


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict[str, str]], field: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(field, "")
        if value not in ("", "nan", "NaN", None):
            out.append(float(value))
    return out


def avg(rows: list[dict[str, str]], field: str) -> float | None:
    present = values(rows, field)
    return float(np.mean(present)) if present else None


def f(value: float | None, digits: int = 3, scale: float = 1.0) -> str:
    return "—" if value is None else f"{value * scale:.{digits}f}"


def pct(value: float | None) -> str:
    return f(value, 2, 100.0)


def collect_collusion_attack() -> dict[tuple[str, int, str], list[dict[str, str]]]:
    collected = {}
    for model in MODELS:
        for k in KS:
            file_cache: dict[str, list[dict[str, str]]] = {}
            for method, _, _, directory, prefix in METHODS:
                if directory not in file_cache:
                    file_cache[directory] = read(DATA / directory / f"{prefix}_{model}_K{k}.csv")
                rows = [row for row in file_cache[directory] if row.get("method") == method]
                if len(rows) == 300:
                    collected[(model, k, method)] = rows
    return collected


def target_metrics(rows: list[dict[str, str]], trial_field: str,
                   hit_field: str) -> tuple[float, float, float]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        grouped[int(row[trial_field])].append(int(row[hit_field]))
    counts = np.asarray([sum(hits) for hits in grouped.values()], dtype=float)
    return float(np.mean([hit for hits in grouped.values() for hit in hits])), float(counts.mean()), float(np.mean(counts > 0))


def append_system_config(lines: list[str]) -> None:
    lines.extend([
        "## 1. 固定实验配置",
        "",
        "| 水印系统 | Payload 位数 | Native 注册库 | Matched 注册库 | 水印采样率 | Native presence 输出 |",
        "|---|---:|---:|---:|---:|---|",
        "| AudioSeal | 16 | 65,536 | 1,024 | 16 kHz | 分数 |",
        "| WavMark | 16 | 65,536 | 1,024 | 16 kHz | 同步判定 |",
        "| TimbreWM | 10 | 1,024 | 1,024 | 22.05 kHz | 不提供 |",
        "| VoiceMark | 16 | 65,536 | 1,024 | 16 kHz | 分数 |",
        "| WMCodec | 16 | 65,536 | 1,024 | 24 kHz | 不提供 |",
        "",
        "| 项目 | 设置 |",
        "|---|---|",
        "| 说话人 / 载体音频 | 100 名说话人；每人取最长的 10 秒、16 kHz 音频 |",
        "| 重复次数 | 每个模型–K–条件 300 次 |",
        "| 合谋规模 | K = {2, 3, 5, 8} |",
        "| 候选篡改目标 | 每个 coalition/trial 10 个目标 |",
        "| 音质参考 | 第一条合法水印合谋音频；单样本 baseline 使用攻击前的源水印音频 |",
        "| 音质指标 | PESQ-WB、STOI、SI-SDR；单样本 baseline 另报 SNR |",
        "| 盲攻击 ASR | top-1 注册身份不属于合谋集合/源身份 |",
        "| NAC | CSV 中的 `ACC_near_norm`：与任一合谋者 payload 的最大解码比特一致率；越低越好 |",
        "| 定向篡改成功 | 返回的 top-1 身份等于指定目标 |",
        "| Native 评估 | 使用全部 2^L 个注册 payload；无拒识选项 |",
        "",
    ])


def append_capability_table(lines: list[str]) -> None:
    rows = [
        ["FWP（本文）", "盲逃逸", "K≥2", "否", "否", "否", "否", "否", "否", "是", "选最远波形对；权重 0.5/0.5"],
        ["Softmin-v2（本文）", "定向篡改", "K≥2", "是", "是", "否", "否", "否", "否", "检测器黑盒", "仅基于 payload 选 top-10 并优化 softmin 权重"],
        ["Mean", "盲逃逸/对照", "K≥2", "否", "否", "否", "否", "否", "否", "是", "均匀凸平均"],
        ["RP / EEP", "盲对照", "K≥2", "否", "否", "否", "否", "否", "否", "是", "随机配对 / 能量极值配对"],
        ["DM / BDB", "盲逃逸", "K≥2", "否", "否", "否", "否", "否", "否", "是", "波形距离优化；capped simplex"],
        ["Median / min / max / random min-max", "经典逃逸", "K≥2", "否", "否", "否", "否", "否", "否", "是", "逐采样点顺序统计量"],
        ["Copy-paste", "信号域对照", "K≥2", "否", "否", "否", "否", "否", "否", "是", "从合谋成员复制 20 ms 音频块"],
        ["PGR oracle", "逃逸诊断", "K≥2", "是", "否", "否", "否", "否", "否", "检测器黑盒", "真实 payload Gram minimax"],
        ["TCT v1", "定向篡改", "K≥2", "是", "是", "否", "否", "否", "否", "检测器黑盒", "d-hull 目标筛选；目标 payload 距离；α≤0.5"],
        ["EnCodec", "单样本逃逸", "1", "否", "否", "否", "否", "否", "是", "源检测器黑盒", "预训练神经 codec 重建"],
        ["Vocos", "单样本逃逸", "1", "否", "否", "否", "否", "否", "是", "源检测器黑盒", "预训练神经 vocoder 重合成"],
        ["Cross overwrite", "单样本逃逸", "1", "否", "否", "否", "否", "是（其他水印）", "是", "源检测器黑盒", "嵌入另一种水印系统"],
        ["Same-model overwrite", "单样本逃逸/篡改", "1", "仅定向时", "仅定向时", "否", "否", "是", "否", "对源模型非黑盒", "重新嵌入新/目标 payload"],
        ["FGSM / I-FGSM", "单样本逃逸/篡改", "1", "是", "I-FGSM 需要", "是", "否", "否", "否", "否（白盒）", "1 步移除 / 10 步定向波形梯度"],
    ]
    lines.extend([
        "## 2. 攻击者能力与配置对比",
        "",
        "按对本文方法最有利的顺序排列：本文方法在前，随后是需要逐渐更强模型访问权限的方法。黑盒均相对于被测水印检测器定义。",
        "",
        "| 方法 | 目标 | 输入水印副本数 | 知道源 payload | 知道目标 payload | 检测器梯度/权重 | 查询检测器 | 访问嵌入器 | 辅助神经网络 | 黑盒状态 | 核心操作 |",
        "|---|---|---:|---|---|---|---|---|---|---|---|",
    ])
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", "以上方法均不需要针对本任务从头训练；神经 codec/vocoder baseline 使用预训练辅助模型。", ""])


def append_collusion_attack(lines: list[str], attack: dict) -> None:
    method_meta = {method: (label, category) for method, label, category, _, _ in METHODS}
    lines.extend([
        "## 3. 盲多样本攻击：全部方法",
        "",
        "### 3.1 Native 注册库汇总",
        "",
        "ASR 按 K 分列；NAC 与音质对四个等规模 K 单元汇总（每个模型–方法共 1,200 次）。",
        "",
        "| 水印系统 | 方法 | 类别 | ASR K2 | ASR K3 | ASR K5 | ASR K8 | NAC ↓ | PESQ ↑ | STOI ↑ | SI-SDR ↑ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        for method, _, _, _, _ in METHODS:
            groups = [attack.get((model, k, method), []) for k in KS]
            if not any(groups):
                continue
            all_rows = [row for group in groups for row in group]
            label, category = method_meta[method]
            asr = [avg(group, "ASR") for group in groups]
            lines.append(
                f"| {LABEL[model]} | {label} | {category} | "
                + " | ".join(pct(value) for value in asr)
                + f" | {f(avg(all_rows, 'ACC_near_norm'))} | {f(avg(all_rows, 'PESQ'))} | "
                  f"{f(avg(all_rows, 'STOI'), 4)} | {f(avg(all_rows, 'SI_SDR'), 2)} |"
            )
    lines.extend(["", "### 3.2 全部模型–K 详细结果", "",
                  "| 水印系统 | K | 方法 | ASR ↑ | NAC ↓ | PESQ ↑ | STOI ↑ | SI-SDR ↑ |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"])
    for model in MODELS:
        for k in KS:
            for method, label, _, _, _ in METHODS:
                rows = attack.get((model, k, method), [])
                if not rows:
                    continue
                lines.append(
                    f"| {LABEL[model]} | {k} | {label} | {pct(avg(rows, 'ASR'))} | "
                    f"{f(avg(rows, 'ACC_near_norm'))} | {f(avg(rows, 'PESQ'))} | "
                    f"{f(avg(rows, 'STOI'), 4)} | {f(avg(rows, 'SI_SDR'), 2)} |"
                )
    lines.append("")


def append_neural_attack(lines: list[str]) -> None:
    files = sorted((DATA / "attack_evasion").glob("attack_evasion_*.csv"))
    items = []
    order = {"overwrite_same": 0, "overwrite_cross": 1, "encodec": 2, "vocoder": 3, "fgsm": 4}
    for path in files:
        rows = read(path)
        model, attack_name = rows[0]["model"], rows[0]["attack"]
        registries = {int(row["N_registry"]) for row in rows}
        grouped = {n: [row for row in rows if int(row["N_registry"]) == n] for n in registries}
        reference = grouped[1024]
        native_n = full_n = 1024 if model == "timbrewm" else 65536
        native = grouped[full_n]
        label = attack_name
        if attack_name == "overwrite_same":
            label = "Overwrite-same"
        elif attack_name == "overwrite_cross":
            target = next((row["attacker_model"] for row in reference if row.get("attacker_model")), "?")
            label = f"Overwrite-cross → {LABEL.get(target, target)}"
        elif attack_name == "encodec":
            label = "EnCodec"
        elif attack_name == "vocoder":
            label = "Vocos"
        elif attack_name == "fgsm":
            label = "FGSM (1 step)"
        items.append((order[attack_name], model, label, reference, native))
    lines.extend([
        "## 4. 单样本神经网络攻击 baseline",
        "",
        "K=1 时，NAC 为解码 payload 与唯一源身份的比特一致率（`source_bit_accuracy`）。WavMark 无有效解码比特的单元记为不可用。",
        "",
        "| 水印系统 | Baseline | ASR N=1,024 | ASR native | NAC ↓ | Presence 移除率 | PESQ | STOI | SI-SDR | SNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, model, label, reference, native in sorted(items):
        lines.append(
            f"| {LABEL[model]} | {label} | {pct(avg(reference, 'identity_escape'))} | "
            f"{pct(avg(native, 'identity_escape'))} | {f(avg(reference, 'source_bit_accuracy'))} | "
            f"{pct(avg(reference, 'presence_removed'))} | {f(avg(reference, 'PESQ'))} | "
            f"{f(avg(reference, 'STOI'), 4)} | {f(avg(reference, 'SI_SDR'), 2)} | "
            f"{f(avg(reference, 'SNR'), 2)} |"
        )
    lines.append("")


def condition_file(condition: str, model: str, k: int) -> Path | None:
    if condition == "Favorable, native":
        return DATA / "tamper" / f"tamper_{model}_K{k}.csv"
    if condition == "Favorable, N=1,024":
        if model == "timbrewm":
            return DATA / "tamper" / f"tamper_{model}_K{k}.csv"
        return DATA / "tamper" / f"tamper_N1024_{model}_K{k}.csv"
    if condition == "Arbitrary, native":
        return DATA / "tamper_arbitrary" / f"tamper_arbitrary_{model}_K{k}.csv"
    if condition == "Arbitrary, N=1,024":
        return DATA / "tamper_arbitrary_matched_n1024" / f"tamper_arbitrary_N1024_{model}_K{k}.csv"
    return None


def append_tct(lines: list[str]) -> None:
    conditions = ["Favorable, N=1,024", "Favorable, native", "Arbitrary, N=1,024", "Arbitrary, native"]
    lines.extend([
        "## 5. 定向篡改：TCT-v1 与 Mean 对照",
        "",
        "每格为 `十个中至少一个成功率 % / 单目标成功率 %`。Favorable TCT-v1 使用 d-hull 筛选；Arbitrary 从非合谋者中均匀抽样。",
        "",
        "| 条件 | 水印系统 | 方法 | K2 | K3 | K5 | K8 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        for model in MODELS:
            for method in ("tct", "mean"):
                cells = []
                any_present = False
                for k in KS:
                    path = condition_file(condition, model, k)
                    rows = [] if path is None else [row for row in read(path) if row.get("method") == method]
                    if len(rows) != 3000:
                        cells.append("—")
                        continue
                    per_target, _, any_ten = target_metrics(rows, "gi", "target_top1")
                    cells.append(f"{100 * any_ten:.2f}/{100 * per_target:.2f}")
                    any_present = True
                if any_present:
                    lines.append(f"| {condition} | {LABEL[model]} | {method.upper()} | " + " | ".join(cells) + " |")
    lines.append("")


def softmin_rows() -> list[dict]:
    result = []
    for model in MODELS:
        for k in KS:
            paths = [(1024, DATA / "tamper_softmin_v2_n1024" / f"margin_reachability_v2_{model}_K{k}.csv")]
            if model != "timbrewm":
                paths.append((65536, DATA / "tamper_softmin_v2_native" / f"margin_reachability_v2_native_{model}_K{k}.csv"))
            for n, path in paths:
                rows = read(path)
                if len(rows) != 3000:
                    continue
                per_target, mean_hits, any_ten = target_metrics(rows, "gi", "target_top1")
                result.append({
                    "model": model, "K": k, "N": n, "per": per_target,
                    "hits": mean_hits, "any": any_ten, "PESQ": avg(rows, "PESQ"),
                    "STOI": avg(rows, "STOI"), "SI_SDR": avg(rows, "SI_SDR"),
                })
    return result


def append_softmin(lines: list[str], softmin: list[dict]) -> None:
    lines.extend([
        "## 6. 定向篡改：Softmin-v2（本文方法）",
        "",
        "目标为 N=1,024 下 Softmin-v2 可达性得分最高的十个身份。Native 评估保持目标、权重和攻击音频完全不变，仅切换归因注册库。",
        "",
        "| 水印系统 | K | 评估 N | 单目标成功率 | 平均成功目标数 /10 | 十个中至少一个成功率 | PESQ | STOI | SI-SDR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in softmin:
        lines.append(
            f"| {LABEL[row['model']]} | {row['K']} | {row['N']:,} | {pct(row['per'])} | "
            f"{row['hits']:.3f} | {pct(row['any'])} | {f(row['PESQ'])} | "
            f"{f(row['STOI'], 4)} | {f(row['SI_SDR'], 2)} |"
        )
    lines.append("")


def single_tamper_rows() -> list[dict]:
    result = []
    for path in sorted((DATA / "tamper_single_copy").glob("single_tamper_*.csv")):
        rows = read(path)
        attack, model = rows[0]["attack"], rows[0]["model"]
        for n in sorted({int(row["N_registry"]) for row in rows}):
            subset = [row for row in rows if int(row["N_registry"]) == n]
            per_target, mean_hits, any_ten = target_metrics(subset, "trial_id", "target_hit")
            result.append({
                "attack": attack, "model": model, "N": n, "per": per_target,
                "hits": mean_hits, "any": any_ten, "PESQ": avg(subset, "PESQ"),
                "STOI": avg(subset, "STOI"), "SI_SDR": avg(subset, "SI_SDR"),
                "SNR": avg(subset, "SNR"),
            })
    return result


def append_single_tamper(lines: list[str], rows: list[dict]) -> None:
    lines.extend([
        "## 7. 单样本定向篡改 baseline",
        "",
        "十个目标由 K=1 特化的 Softmin-v2 可达性选择。I-FGSM 使用 10 步；WavMark 流水线不可微，因此不运行 I-FGSM。",
        "",
        "| Baseline | 水印系统 | 评估 N | 单目标成功率 | 平均成功目标数 /10 | 十个中至少一个成功率 | PESQ | STOI | SI-SDR | SNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        name = "I-FGSM (10 steps)" if row["attack"] == "ifgsm" else "Targeted overwrite"
        lines.append(
            f"| {name} | {LABEL[row['model']]} | {row['N']:,} | {pct(row['per'])} | "
            f"{row['hits']:.3f} | {pct(row['any'])} | {f(row['PESQ'])} | "
            f"{f(row['STOI'], 4)} | {f(row['SI_SDR'], 2)} | {f(row['SNR'], 2)} |"
        )
    lines.append("")


def tct_any(condition: str, model: str, k: int) -> float | None:
    path = condition_file(condition, model, k)
    rows = [] if path is None else [row for row in read(path) if row.get("method") == "tct"]
    return target_metrics(rows, "gi", "target_top1")[2] if len(rows) == 3000 else None


def append_k8_summary(lines: list[str], softmin: list[dict], single: list[dict]) -> None:
    soft = {(row["model"], row["N"]): row for row in softmin if row["K"] == 8}
    one = {(row["attack"], row["model"], row["N"]): row for row in single}
    lines.extend([
        "## 8. 定向篡改核心汇总",
        "",
        "多样本方法取 K=8，单样本 baseline 取 K=1；表中均为十个目标中至少一个成功的比例。",
        "",
        "| 水印系统 | TCT-v1 favorable N=1,024 | Softmin-v2 N=1,024 | TCT-v1 favorable native | Softmin-v2 native | Overwrite K=1 native | I-FGSM K=1 native |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        native = 1024 if model == "timbrewm" else 65536
        lines.append(
            f"| {LABEL[model]} | {pct(tct_any('Favorable, N=1,024', model, 8))} | "
            f"{pct(soft.get((model, 1024), {}).get('any'))} | "
            f"{pct(tct_any('Favorable, native', model, 8))} | "
            f"{pct(soft.get((model, native), {}).get('any'))} | "
            f"{pct(one.get(('overwrite', model, native), {}).get('any'))} | "
            f"{pct(one.get(('ifgsm', model, native), {}).get('any'))} |"
        )
    lines.append("")


def append_integrity(lines: list[str]) -> None:
    lines.extend([
        "## 9. 数据来源与完整性",
        "",
        "| 数据集 | 已完成单元 | 行数 | 最终目录 |",
        "|---|---:|---:|---|",
        "| Blind Mean/FWP | 20 files | 12,000 | `data/attack/` |",
        "| Classical signal baselines | 20 files | 30,000 | `data/baselines/` |",
        "| RP / EEP / DM / BDB / PGR | 100 files | 30,000 | corresponding `data/{rp,eep,dm,bdb,pgr}/` |",
        "| Single-copy neural evasion | 24 files | 12,900 | `data/attack_evasion/` |",
        "| TCT-v1 favorable（native + matched） | 32 files | 192,000 | `data/tamper/` |",
        "| TCT-v1 arbitrary native | 20 files | 120,000 | `data/tamper_arbitrary/` |",
        "| TCT-v1 arbitrary N=1,024 | 20 files | 120,000 | `data/tamper_arbitrary_matched_n1024/` |",
        "| Softmin-v2 N=1,024 | 20 files | 60,000 | `data/tamper_softmin_v2_n1024/` |",
        "| Softmin-v2 native | 16 files | 48,000 | `data/tamper_softmin_v2_native/` |",
        "| Single-copy targeted tamper | 9 files | 48,000 | `data/tamper_single_copy/` |",
        "",
        "最终验证：24 个神经逃逸单元、20 个 Softmin-v2 N=1,024 单元、16 个 Softmin-v2 native 单元、9 个单样本篡改单元全部 PASS。`data/INDEX.csv` 保存行数与 SHA-256。",
        "",
        "可比性约束：多样本与单样本结果分表；TCT-v1、Softmin-v2 和 K=1 baseline 的 favorable 目标集合由各自方法产生，不作为配对样本比较。",
        "",
    ])


def main() -> None:
    attack = collect_collusion_attack()
    softmin = softmin_rows()
    single = single_tamper_rows()
    lines = [
        "# 攻击与篡改实验完整汇总",
        "",
        "数据源：`data/` 下已完成并归档的最终 CSV。除特别说明外，每个实验单元均为 300 次试验的经验统计。",
        "",
    ]
    append_system_config(lines)
    append_capability_table(lines)
    append_collusion_attack(lines, attack)
    append_neural_attack(lines)
    append_tct(lines)
    append_softmin(lines, softmin)
    append_single_tamper(lines, single)
    append_k8_summary(lines, softmin, single)
    append_integrity(lines)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} lines={len(lines)}")


if __name__ == "__main__":
    main()

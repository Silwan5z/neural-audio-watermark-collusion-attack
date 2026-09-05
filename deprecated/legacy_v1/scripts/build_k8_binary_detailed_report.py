#!/usr/bin/env python3
"""Build a binary-only, detailed report for the native-rate K=8 case study.

The original JSON files are treated as immutable provenance.  All newly
published payload labels, tables, figures, and audio aliases use fixed-width
binary strings in the canonical b0-to-b(L-1) order.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "k8_constructed_payload_case_native_20260829"
RAW = DATA / "raw"
OUT = DATA / "detailed_document_binary"
FIG = OUT / "figures"
TABLES = OUT / "tables"
AUDIO_ALIAS = OUT / "audio_binary"

MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABEL = {
    "audioseal": "AudioSeal",
    "wavmark": "WavMark",
    "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark",
    "wmcodec": "WMCodec",
}
COLOR = {
    "audioseal": "#3977A8",
    "wavmark": "#4E9B66",
    "timbrewm": "#8D6FA8",
    "voicemark": "#C85A5A",
    "wmcodec": "#D89235",
}
SEED = 20260829


def bits_text(bits: list[int], grouped: bool = True) -> str:
    """Return the stored canonical b0->b(L-1) bit sequence, never an integer."""
    s = "".join(str(int(x)) for x in bits)
    if not grouped:
        return s
    return " ".join(s[i : i + 4] for i in range(0, len(s), 4))


def bit_class(ones: int) -> str:
    if ones == 0:
        return "unanimous 0"
    if ones == 8:
        return "unanimous 1"
    if ones == 4:
        return "4/4 tie"
    return "strict majority 1" if ones > 4 else "strict majority 0"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def link_audio_binary_names(raw: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for model in MODELS:
        d = raw[model]
        dst_dir = AUDIO_ALIAS / model
        dst_dir.mkdir(parents=True, exist_ok=True)
        entries = [
            ("clean", Path(d["clean_audio_path"]), "clean.wav", ""),
            (
                "mixture",
                Path(d["mixed_audio_path"]),
                f"equal_mean_K8_decoded_{bits_text(d['decoded_bits'], False)}.wav",
                bits_text(d["decoded_bits"]),
            ),
        ]
        for i, (src, payload_bits) in enumerate(zip(d["member_audio_paths"], d["payload_bits"]), 1):
            entries.append(
                (
                    f"colluder_{i:02d}",
                    Path(src),
                    f"colluder_{i:02d}_bits_b0_to_b{len(payload_bits)-1}_{bits_text(payload_bits, False)}.wav",
                    bits_text(payload_bits),
                )
            )
        for role, src, alias_name, payload in entries:
            dst = dst_dir / alias_name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.link(src, dst)
                link_type = "hardlink"
            except OSError:
                shutil.copy2(src, dst)
                link_type = "copy"
            rows.append(
                {
                    "system": LABEL[model],
                    "role": role,
                    "payload_bits_b0_to_last": payload,
                    "native_sample_rate_hz": d["native_sample_rate"],
                    "relative_audio_path": str(dst.relative_to(OUT)),
                    "storage": link_type,
                }
            )
    return rows


def export_binary_tables(raw: dict[str, dict]) -> dict[str, list[dict]]:
    payload_rows: list[dict] = []
    for i, bits in enumerate(raw["audioseal"]["payload_bits"], 1):
        payload_rows.append(
            {
                "colluder": f"C{i}",
                "payload_16bit_b0_to_b15": bits_text(bits),
                "timbrewm_10bit_b0_to_b9": bits_text(bits[:10]),
            }
        )

    system_rows: list[dict] = []
    clean_rows: list[dict] = []
    bit_rows: list[dict] = []
    wm_digit_rows: list[dict] = []
    for model in MODELS:
        d = raw[model]
        system_rows.append(
            {
                "system": LABEL[model],
                "native_sample_rate_hz": d["native_sample_rate"],
                "payload_bits": d["payload_nbits"],
                "clean_exact": f"{d['clean_attribution_exact']}/8",
                "mixture_decoded_bits_b0_to_last": bits_text(d["decoded_bits"]),
                "tracing_failure": d["tracing_failure"],
                "NCA": f"{d['NCA']:.4f}",
                "strict_majority_departure_bits": ",".join(f"b{x}" for x in d["strict_majority_departures"]) or "none",
                "unanimous_violation_bits": ",".join(f"b{x}" for x in d["unanimous_violations"]) or "none",
                "PESQ_vs_first_colluder": f"{d['PESQ_vs_first_copy']:.4f}",
                "STOI_vs_first_colluder": f"{d['STOI_vs_first_copy']:.4f}",
            }
        )
        for i, r in enumerate(d["clean_member_decodes"], 1):
            target = d["payload_bits"][i - 1]
            decoded_int = r["decoded_payload"]
            decoded = [(decoded_int >> j) & 1 for j in range(d["payload_nbits"])]
            clean_rows.append(
                {
                    "system": LABEL[model],
                    "colluder": f"C{i}",
                    "target_bits_b0_to_last": bits_text(target),
                    "clean_decoded_bits_b0_to_last": bits_text(decoded),
                    "exact": r["exact"],
                    "presence_score_if_available": "NA" if r.get("presence") is None else f"{r['presence']:.6f}",
                }
            )
            if model == "wmcodec":
                for h, (digit, conf) in enumerate(zip(r["decoded_digits"], r["digit_confidences"]), 1):
                    wm_digit_rows.append(
                        {
                            "audio": f"C{i}",
                            "classification_head": f"H{h}",
                            "decoded_class_as_4bits": format(int(digit), "04b"),
                            "class_confidence": f"{conf:.8f}",
                        }
                    )
        if model == "wmcodec":
            for h, (digit, conf) in enumerate(zip(d["decoded_digits"], d["digit_confidences"]), 1):
                wm_digit_rows.append(
                    {
                        "audio": "K8 mixture",
                        "classification_head": f"H{h}",
                        "decoded_class_as_4bits": format(int(digit), "04b"),
                        "class_confidence": f"{conf:.8f}",
                    }
                )
        for r in d["bit_results"]:
            bit_rows.append(
                {
                    "system": LABEL[model],
                    "bit": f"b{r['bit_index_lsb_first']}",
                    "colluder_bits_C1_to_C8": bits_text(r["colluder_bits"], False),
                    "ones_among_8": r["ones_among_8"],
                    "coalition_state": bit_class(r["ones_among_8"]),
                    "p_bit_1": f"{r['p_bit_1']:.8f}",
                    "decoded_bit": r["native_decoded_bit"],
                    "decoded_bit_confidence": f"{r['confidence_native_decoded_bit']:.8f}",
                    "strict_majority_departure": int(r["bit_index_lsb_first"] in d["strict_majority_departures"]),
                }
            )
    return {
        "payload_matrix_binary.csv": payload_rows,
        "system_summary_binary.csv": system_rows,
        "clean_member_decodes_binary.csv": clean_rows,
        "bit_confidence_binary.csv": bit_rows,
        "wmcodec_head_confidence_binary.csv": wm_digit_rows,
    }


def plot_system_payload_and_confidence(model: str, d: dict) -> None:
    nbits = d["payload_nbits"]
    matrix = np.asarray(d["payload_bits"] + [d["decoded_bits"]], dtype=int)
    fig = plt.figure(figsize=(12.2, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.imshow(matrix, cmap=ListedColormap(["#DCEAF4", "#F4C98B"]), vmin=0, vmax=1, aspect="auto")
    for r in range(matrix.shape[0]):
        for c in range(nbits):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=8.5, fontweight="bold")
    for c in range(nbits):
        ax.add_patch(plt.Rectangle((c - 0.5, 7.5), 1, 1, fill=False, edgecolor="#17202A", linewidth=1.4))
        if c in d["strict_majority_departures"]:
            ax.add_patch(plt.Rectangle((c - 0.46, 7.54), .92, .92, fill=False, edgecolor="#C0392B", linewidth=2.4))
    ax.axhline(7.5, color="#17202A", lw=2.0)
    ax.set_yticks(range(9), [f"C{i}" for i in range(1, 9)] + ["MIX"])
    ax.set_xticks(range(nbits), [f"b{i}\n{d['ones_per_bit'][i]}/8" for i in range(nbits)])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(
        f"{LABEL[model]}: eight colluder payloads and final decoded payload (b0 to b{nbits-1})",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Bit index and number of colluders carrying 1")
    ax.set_ylabel("Input copies and mixture output")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    br = d["bit_results"]
    x = np.arange(nbits)
    p = np.asarray([r["p_bit_1"] for r in br])
    bars = ax2.bar(x, p, color=COLOR[model], width=.72, alpha=.9)
    ax2.axhline(.5, color="#444", ls="--", lw=1.0)
    for i, r in enumerate(br):
        hard = r["native_decoded_bit"]
        ax2.scatter(i, hard, marker="o", s=22, color="#111", zorder=4)
        ax2.text(i, min(1.03, p[i] + .045), f"{p[i]:.2f}", ha="center", va="bottom", fontsize=6.7, rotation=90)
        if r["ones_among_8"] == 4:
            bars[i].set_edgecolor("#8A4B08")
            bars[i].set_linewidth(1.8)
    ax2.set_xlim(-.7, nbits - .3)
    ax2.set_ylim(-.08, 1.15)
    ax2.set_xticks(x, [f"b{i}" for i in range(nbits)])
    ax2.set_ylabel("Decoder P(bit=1)")
    ax2.set_xlabel("Black dot: final hard bit; outlined bar: 4/4 coalition bit")
    ax2.grid(axis="y", color="#E5E7E9", lw=.6)
    ax2.spines[["top", "right"]].set_visible(False)

    input_lines = [f"C{i}: {bits_text(bits)}" for i, bits in enumerate(d["payload_bits"], 1)]
    output_line = f"MIX: {bits_text(d['decoded_bits'])}"
    fig.text(.995, .88, "\n".join(input_lines + ["", output_line]), ha="right", va="top", family="monospace", fontsize=8)
    fig.suptitle(
        f"Equal mean, K=8 | TF={d['tracing_failure']} | NCA={d['NCA']:.4f} | "
        f"PESQ={d['PESQ_vs_first_copy']:.4f} | STOI={d['STOI_vs_first_copy']:.4f}",
        y=.995,
        fontsize=10,
    )
    fig.subplots_adjust(right=.75, top=.93)
    save_figure(fig, f"fig_{model}_colluders_output_confidence")


def plot_direct_summary(raw: dict[str, dict]) -> None:
    labels = [LABEL[m] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    vals = [raw[m]["NCA"] for m in MODELS]
    axes[0].bar(x, vals, color=[COLOR[m] for m in MODELS])
    axes[0].set_ylim(.68, .93)
    axes[0].set_ylabel("NCA")
    for i, v in enumerate(vals): axes[0].text(i, v + .006, f"{v:.3f}", ha="center", fontsize=8)
    vals = [raw[m]["PESQ_vs_first_copy"] for m in MODELS]
    axes[1].bar(x, vals, color=[COLOR[m] for m in MODELS])
    axes[1].set_ylim(4.25, 4.66)
    axes[1].set_ylabel("PESQ vs first colluder")
    for i, v in enumerate(vals): axes[1].text(i, v + .012, f"{v:.3f}", ha="center", fontsize=8)
    vals = [raw[m]["STOI_vs_first_copy"] for m in MODELS]
    axes[2].bar(x, vals, color=[COLOR[m] for m in MODELS])
    axes[2].set_ylim(.965, 1.003)
    axes[2].set_ylabel("STOI vs first colluder")
    for i, v in enumerate(vals): axes[2].text(i, v + .001, f"{v:.3f}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
        ax.grid(axis="y", color="#E5E7E9", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Direct observed metrics for the pre-registered K=8 case", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_direct_metric_summary")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def build_markdown(raw: dict[str, dict], binary_tables: dict[str, list[dict]]) -> str:
    pbits = raw["audioseal"]["payload_bits"]
    lines = [
        "# K=8 构造型二进制 Payload 合谋案例：完整结果与逐 Bit 置信度分析",
        "",
        "> 本文档只使用二进制 payload 表示。所有 bit 串均按原实验的规范顺序 `b0 → b(L−1)` 展示，不能按常见的高位在前整数写法解读。",
        "",
        "## 1. 核心结论",
        "",
        "- 五个系统的 8 个单副本水印均能在各自原生采样率下精确解码（均为 8/8）。",
        "- 五个等权均值混合音频均被解码为不属于 8 个合谋者集合的身份，因此本案例中五个系统的 tracing failure 均为 1。",
        "- AudioSeal、WavMark、TimbreWM 保留了所有严格多数 bit；VoiceMark 在 b3、b6、b7 偏离严格多数；WMCodec 在 b3 偏离严格多数。",
        "- 所有系统都保留了两个全体一致锚点：b0 为 0/8，b9 为 8/8，没有 unanimous violation。",
        "- 4/4 并不意味着解码置信度接近 0.5：不同系统会以完全不同的置信度将同一平票构成解析为 0 或 1。",
        "- 这是一个预先构造、未按结果筛选的单案例，用于展示 bit-level 机制；不能替代多 trial 总体统计，也不能据此给系统作总体排序。",
        "",
        "## 2. 实验配置",
        "",
        markdown_table(
            ["项目", "配置"],
            [
                ["联盟规模", "K=8"],
                ["混合方式", "8 条水印音频全波形等权均值，每条权重 1/8"],
                ["随机种子", str(SEED)],
                ["内容", "同一语音内容，speaker=chinese:SSB0005"],
                ["16-bit 构成", "每 bit 的 1 数量依次为 0,1,2,3,4,4,5,6,7,8,4,4,4,4,4,4"],
                ["TimbreWM 构成", "使用前 10 bit，覆盖 0/8 到 8/8，并包含两个 4/4 bit"],
                ["解码采样率", "AudioSeal/WavMark/VoiceMark: 16 kHz；TimbreWM: 22.05 kHz；WMCodec: 24 kHz"],
                ["音质指标", "PESQ、STOI；与第一个合谋副本比较；统一在 16 kHz 计算"],
                ["payload 顺序", "LSB-first canonical labels；本文统一写作 b0→b(L−1)"],
            ],
        ),
        "",
        "## 3. 八个合谋者的二进制 Payload",
        "",
        markdown_table(
            ["合谋者", "16-bit（b0→b15）", "TimbreWM 10-bit（b0→b9）"],
            [[f"C{i}", bits_text(bits), bits_text(bits[:10])] for i, bits in enumerate(pbits, 1)],
        ),
        "",
        "## 4. 总体结果",
        "",
        markdown_table(
            ["系统", "原生采样率", "clean exact", "混合后解码（b0→最后一 bit）", "TF", "NCA", "严格多数偏离", "PESQ", "STOI"],
            [
                [
                    LABEL[m], str(raw[m]["native_sample_rate"]), "8/8", bits_text(raw[m]["decoded_bits"]),
                    str(raw[m]["tracing_failure"]), f"{raw[m]['NCA']:.4f}",
                    ", ".join(f"b{x}" for x in raw[m]["strict_majority_departures"]) or "无",
                    f"{raw[m]['PESQ_vs_first_copy']:.4f}", f"{raw[m]['STOI_vs_first_copy']:.4f}",
                ] for m in MODELS
            ],
        ),
        "",
        "![五系统直接观测指标](figures/fig_direct_metric_summary.png)",
        "",
        "## 5. 逐系统：合谋者、最终输出与逐 Bit 置信度",
        "",
        "每张图上半部分逐行给出 C1–C8 的输入 bit 和 MIX 的最终 hard payload；下半部分是解码器原始 `P(bit=1)`。黑点表示最终 hard bit，描边柱表示 4/4 平票 bit，红框表示最终 bit 偏离严格多数。",
        "",
    ]
    for m in MODELS:
        d = raw[m]
        lines += [
            f"### 5.{MODELS.index(m)+1} {LABEL[m]}",
            "",
            f"![{LABEL[m]} 合谋者与输出置信度](figures/fig_{m}_colluders_output_confidence.png)",
            "",
            markdown_table(
                ["bit", "C1…C8", "1 的数量", "构成", "P(bit=1)", "hard bit", "hard-bit confidence"],
                [[
                    f"b{r['bit_index_lsb_first']}", bits_text(r["colluder_bits"], False), str(r["ones_among_8"]),
                    bit_class(r["ones_among_8"]), f"{r['p_bit_1']:.6f}", str(r["native_decoded_bit"]),
                    f"{r['confidence_native_decoded_bit']:.6f}",
                ] for r in d["bit_results"]],
            ),
            "",
        ]
    lines += [
        "## 6. Clean 单副本解码核验",
        "",
        "所有系统均为 8/8 exact。presence score 只在系统原始接口实际提供时列出；TimbreWM 和 WMCodec 不用人为构造替代分数。WMCodec 的官方四个分类头置信度单独列在 binary-only CSV 中。",
        "",
        markdown_table(
            ["系统", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            [[LABEL[m]] + ["NA" if r.get("presence") is None else f"{r['presence']:.6f}" for r in raw[m]["clean_member_decodes"]] for m in MODELS],
        ),
        "",
        "## 7. 音质",
        "",
        markdown_table(
            ["系统", "PESQ vs C1", "STOI vs C1", "指标采样率"],
            [[LABEL[m], f"{raw[m]['PESQ_vs_first_copy']:.4f}", f"{raw[m]['STOI_vs_first_copy']:.4f}", "16 kHz"] for m in MODELS],
        ),
        "",
        "## 8. WMCodec 原生协议核验",
        "",
        "WMCodec 官方推理在 24 kHz 工作。早期统一 16 kHz wrapper 会先将 24 kHz 水印音频降采样到 16 kHz，检测时再升采样回 24 kHz；该往返使 clean exact 从 8/8 降为 6/8。因此，本报告只采用 native 24 kHz 结果。原生混合输出的四个分类头预测与置信度如下；分类类别也只用 4-bit 二进制表示。",
        "",
        markdown_table(
            ["分类头", "预测类别（4-bit）", "置信度"],
            [[f"H{i}", format(int(x), "04b"), f"{c:.8f}"] for i, (x, c) in enumerate(zip(raw['wmcodec']['decoded_digits'], raw['wmcodec']['digit_confidences']), 1)],
        ),
        "",
        "## 9. 数据完整性与证据边界",
        "",
        "- 5 个系统均具有 clean、8 个 watermarked colluder copies 和 1 个 mixture，共 50 条 WAV。",
        "- 构造矩阵在查看输出前固定，不按系统结果筛选案例。",
        "- 所有 payload、最终 hard bit、P(bit=1)、NCA、tracing failure、PESQ、STOI 均直接来自逐样本 JSON。",
        "- 文档没有引入新的性能指标；图中只显示输入 bit、输出 bit、系统给出的 bit 概率及论文已有指标。",
        "- 单案例不能给出置信区间，也不能支持总体优劣或统计显著性结论。",
        "- WMCodec 的 16-bit 展开是项目身份层对官方四个 16 类分类头的双射表示；模型与 checkpoint 已核对为官方实现，正式推理保持 24 kHz。",
        "",
        "## 10. 文件与复现入口",
        "",
        "- 原始逐系统结果：`../raw/`",
        "- 二进制音频别名：`audio_binary/`",
        "- 二进制逐样本/逐 bit 表：`tables/`",
        "- 论文图（PDF/SVG/PNG）：`figures/`",
        "- 生成脚本：`scripts/build_k8_binary_detailed_report.py`",
        "- 固定随机种子：`20260829`",
        "",
        "### Binary-only 数据表",
        "",
        *[f"- `tables/{name}`" for name in binary_tables],
        "- `tables/audio_manifest_binary.csv`",
        "",
    ]
    return "\n".join(lines) + "\n"


def pdf_table(data, widths=None, font_size=7.2, header=True, repeat_rows=1):
    t = Table(data, colWidths=widths, repeatRows=repeat_rows if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#B9C2C9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6ED")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17202A")),
        ]
    for r in range(1 if header else 0, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F6F8F9")))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(raw: dict[str, dict], binary_tables: dict[str, list[dict]]) -> Path:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf_path = OUT / "K8_constructed_payload_binary_detailed_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, rightMargin=1.35 * cm, leftMargin=1.35 * cm,
        topMargin=1.35 * cm, bottomMargin=1.35 * cm, title="K=8 Binary Payload Collusion Case Study",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CJKTitle", parent=styles["Title"], fontName="STSong-Light", fontSize=19, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"))
    h1 = ParagraphStyle("CJKH1", parent=styles["Heading1"], fontName="STSong-Light", fontSize=14, leading=18, spaceBefore=8, spaceAfter=7, textColor=colors.HexColor("#17324D"))
    h2 = ParagraphStyle("CJKH2", parent=styles["Heading2"], fontName="STSong-Light", fontSize=11.5, leading=15, spaceBefore=7, spaceAfter=5, textColor=colors.HexColor("#315A73"))
    body = ParagraphStyle("CJKBody", parent=styles["BodyText"], fontName="STSong-Light", fontSize=9.2, leading=14, alignment=TA_LEFT, spaceAfter=5)
    small = ParagraphStyle("CJKSmall", parent=body, fontSize=7.3, leading=10)
    note = ParagraphStyle("CJKNote", parent=body, fontSize=8.6, leading=13, leftIndent=10, rightIndent=10, borderColor=colors.HexColor("#AAB7B8"), borderWidth=.5, borderPadding=7, backColor=colors.HexColor("#F4F6F7"))
    story = []

    story += [Paragraph("K=8 构造型二进制 Payload 合谋案例", title), Spacer(1, 6), Paragraph("完整结果、逐 Bit 置信度、音质与协议审计", h1), Spacer(1, 5), Paragraph("所有 payload 均仅以二进制表示；bit 串按 b0→b(L−1) 顺序展示。", note), Spacer(1, 12)]
    story.append(Paragraph("执行摘要", h1))
    for text in [
        "五个系统的 8 个单副本水印均在原生采样率下精确解码；五个等权均值混合均产生不属于八名合谋者集合的身份，因此本构造案例的 tracing failure 均为 1。",
        "AudioSeal、WavMark 和 TimbreWM 保留全部严格多数 bit；VoiceMark 在 b3、b6、b7 偏离严格多数，WMCodec 在 b3 偏离严格多数。所有系统均保持 b0=0/8 与 b9=8/8 两个全体一致锚点。",
        "4/4 平票不会自然带来低置信度。TimbreWM 的两个平票 bit 接近 0.5，而 WavMark、VoiceMark、WMCodec 的部分平票 bit 接近确定性输出，说明最终离散身份不等同于逐 bit 多数表决。",
        "本实验是一个预先固定的机制案例，不是总体统计；不得据此声称某系统总体更脆弱或更稳健。",
    ]:
        story.append(Paragraph("• " + text, body))
    story += [Spacer(1, 5), Image(str(FIG / "fig_direct_metric_summary.png"), width=17.4 * cm, height=5.15 * cm), PageBreak()]

    story += [Paragraph("1. 实验配置", h1)]
    protocol = [
        ["项目", "配置"],
        ["联盟规模", "K=8"],
        ["混合", "全波形等权均值，每个副本权重 1/8"],
        ["种子", str(SEED)],
        ["语音", "同一内容；speaker=chinese:SSB0005"],
        ["16-bit 构成", "0,1,2,3,4,4,5,6,7,8,4,4,4,4,4,4 个 1"],
        ["TimbreWM", "前 10 bit；覆盖 0/8 至 8/8，含两个 4/4"],
        ["native SR", "AudioSeal/WavMark/VoiceMark 16 kHz；TimbreWM 22.05 kHz；WMCodec 24 kHz"],
        ["音质", "PESQ/STOI 相对第一个合谋副本；16 kHz 计算"],
        ["bit 顺序", "LSB-first canonical labels；展示顺序 b0→b(L−1)"],
    ]
    story += [pdf_table(protocol, [3.5 * cm, 13.6 * cm], font_size=8.2), Spacer(1, 9), Paragraph("2. 八个合谋者的 Payload", h1)]
    payload_data = [["合谋者", "16-bit: b0→b15", "TimbreWM 10-bit: b0→b9"]] + [[f"C{i}", bits_text(b), bits_text(b[:10])] for i, b in enumerate(raw["audioseal"]["payload_bits"], 1)]
    story += [pdf_table(payload_data, [2 * cm, 8.5 * cm, 6.6 * cm], font_size=8.1), Spacer(1, 8), Paragraph("构成设计不是从结果中挑选：b0 与 b9 是全体一致锚点；b1–b3 与 b6–b8 覆盖逐级多数；其余为重复的 4/4 平票位。", note), PageBreak()]

    story += [Paragraph("3. 五系统总体结果", h1)]
    summary = [["系统", "SR", "clean", "混合后 binary payload", "TF", "NCA", "多数偏离", "PESQ", "STOI"]]
    for m in MODELS:
        d = raw[m]
        summary.append([LABEL[m], str(d["native_sample_rate"]), "8/8", bits_text(d["decoded_bits"]), str(d["tracing_failure"]), f"{d['NCA']:.4f}", ",".join(f"b{x}" for x in d["strict_majority_departures"]) or "无", f"{d['PESQ_vs_first_copy']:.4f}", f"{d['STOI_vs_first_copy']:.4f}"])
    story += [pdf_table(summary, [1.8*cm,1.2*cm,1.1*cm,5.7*cm,0.7*cm,1.1*cm,1.6*cm,1.3*cm,1.3*cm], font_size=6.7), Spacer(1, 8), Paragraph("TF=1 的判断只依赖最终 decoded binary payload 是否不属于 C1–C8 集合。NCA 是最终 payload 与最近合谋者的 bit agreement；PESQ/STOI 均与 C1 水印音频比较。", note), Spacer(1, 10)]
    story.append(Image(str(FIG / "fig_direct_metric_summary.png"), width=17.3 * cm, height=5.1 * cm))
    story.append(PageBreak())

    for mi, m in enumerate(MODELS, 1):
        d = raw[m]
        story += [Paragraph(f"4.{mi} {LABEL[m]}：输入、最终输出与置信度", h1), Image(str(FIG / f"fig_{m}_colluders_output_confidence.png"), width=17.8 * cm, height=10.2 * cm), Spacer(1, 5)]
        desc = f"最终输出：{bits_text(d['decoded_bits'])}；TF={d['tracing_failure']}；NCA={d['NCA']:.4f}；严格多数偏离：" + (", ".join(f"b{x}" for x in d["strict_majority_departures"]) if d["strict_majority_departures"] else "无") + "；全体一致位偏离：无。"
        story.append(Paragraph(desc, note))
        br_header = ["bit", "C1…C8", "ones", "状态", "P(bit=1)", "hard", "hard conf."]
        br_rows = [br_header] + [[f"b{r['bit_index_lsb_first']}", bits_text(r["colluder_bits"], False), str(r["ones_among_8"]), bit_class(r["ones_among_8"]), f"{r['p_bit_1']:.6f}", str(r["native_decoded_bit"]), f"{r['confidence_native_decoded_bit']:.6f}"] for r in d["bit_results"]]
        story += [Spacer(1, 7), pdf_table(br_rows, [1.1*cm,2.3*cm,1.1*cm,3.2*cm,2.3*cm,1.2*cm,2.3*cm], font_size=6.6), PageBreak()]

    story += [Paragraph("5. Clean 单副本核验", h1), Paragraph("下表报告各系统原始接口提供的 presence score。TimbreWM 与 WMCodec 没有同口径 presence，因此记为 NA，不构造替代指标。所有目标与 clean decoded payload 完全一致。", body)]
    clean_table = [["系统", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "exact"]]
    for m in MODELS:
        vals = ["NA" if r.get("presence") is None else f"{r['presence']:.6f}" for r in raw[m]["clean_member_decodes"]]
        clean_table.append([LABEL[m]] + vals + ["8/8"])
    story += [pdf_table(clean_table, [1.8*cm] + [1.65*cm]*8 + [1.2*cm], font_size=6.3), Spacer(1, 10), Paragraph("6. WMCodec 官方四分类头置信度", h1), Paragraph("WMCodec 输出四个 16 类分类头。为满足 binary-only 要求，下表将每个预测类别写成 4-bit；这是分类头原始置信度，不是新指标。", body)]
    wm_rows = [["音频", "H1: class/conf.", "H2: class/conf.", "H3: class/conf.", "H4: class/conf."]]
    wm = raw["wmcodec"]
    for i, r in enumerate(wm["clean_member_decodes"], 1):
        wm_rows.append([f"C{i}"] + [f"{format(int(x),'04b')} / {c:.6f}" for x,c in zip(r["decoded_digits"], r["digit_confidences"])])
    wm_rows.append(["MIX"] + [f"{format(int(x),'04b')} / {c:.6f}" for x,c in zip(wm["decoded_digits"], wm["digit_confidences"])])
    story += [pdf_table(wm_rows, [1.4*cm] + [3.9*cm]*4, font_size=6.6), Spacer(1, 8), Paragraph("WMCodec protocol control", h2), Paragraph("官方 WMCodec 推理使用 24 kHz。早期 24→16→24 kHz 往返使 clean exact 由 8/8 降至 6/8，因此正式案例只采用 24 kHz native inference。该差异是采样率处理造成的协议效应，不是 payload 构造失败。", note), PageBreak()]

    story += [Paragraph("7. 完整性、文件与结论边界", h1)]
    for x in [
        "音频完整性：5 个系统 ×（clean + 8 colluder copies + mixture）= 50 条 WAV。",
        "图表字段：只使用 colluder bits、decoded hard bits、P(bit=1)、系统输出置信度、tracing failure、NCA、PESQ 和 STOI。",
        "没有将 bit 当作独立 trial，没有计算或声称置信区间；本页是单个构造案例的机制可视化。",
        "所有负向结果均保留：VoiceMark 的三个严格多数偏离、WMCodec 的一个严格多数偏离，以及 WavMark 较低的 NCA 均未隐藏。",
        "二进制音频别名位于 detailed_document_binary/audio_binary；CSV 位于 tables；图的 PDF/SVG/PNG 位于 figures。",
        "原始 JSON 不覆盖、不重写；新包只提供 binary-only 的论文接口。",
    ]:
        story.append(Paragraph("• " + x, body))
    story += [Spacer(1, 8), Paragraph("可复核输出", h2)]
    for name in list(binary_tables) + ["audio_manifest_binary.csv"]:
        story.append(Paragraph(f"tables/{name}", small))
    story += [Spacer(1, 8), Paragraph("复现命令", h2), Paragraph("python scripts/build_k8_binary_detailed_report.py", note), Spacer(1, 8), Paragraph("随机种子：20260829", body)]

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return pdf_path


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#657786"))
    canvas.drawRightString(A4[0] - 1.35 * cm, .7 * cm, f"K=8 binary payload case | page {doc.page}")
    canvas.restoreState()


def validate_binary_only(deliverables: list[Path], raw: dict[str, dict]) -> None:
    forbidden = set()
    for d in raw.values():
        forbidden.update(str(x) for x in d["payloads"])
        forbidden.add(str(d["decoded_identity"]))
    violations = []
    for p in deliverables:
        # SVG path coordinates contain arbitrary standalone integers, so token
        # scanning them produces false positives.  Payload labels embedded in
        # the vector are already generated exclusively from bits_text().
        if p.suffix.lower() not in {".md", ".csv", ".json"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Probabilities can naturally contain the same digit substring (for
        # example a payload label inside 0.9996).  Only reject a standalone
        # integer token, not digits adjacent to another digit or a decimal dot.
        hits = sorted(
            x for x in forbidden
            if re.search(rf"(?<![\d.]){re.escape(x)}(?![\d.])", text)
        )
        if hits:
            violations.append((str(p), hits))
    if violations:
        raise RuntimeError(f"Decimal payload leakage in binary-only package: {violations}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    raw = {m: json.loads((RAW / f"{m}.json").read_text()) for m in MODELS}

    for m in MODELS:
        plot_system_payload_and_confidence(m, raw[m])
    plot_direct_summary(raw)

    binary_tables = export_binary_tables(raw)
    for name, rows in binary_tables.items():
        write_csv(TABLES / name, rows)
    manifest = link_audio_binary_names(raw)
    write_csv(TABLES / "audio_manifest_binary.csv", manifest)

    md = build_markdown(raw, binary_tables)
    md_path = OUT / "K8_constructed_payload_binary_detailed_report.md"
    md_path.write_text(md, encoding="utf-8")
    pdf_path = build_pdf(raw, binary_tables)

    audit = {
        "title": "K=8 constructed binary payload case",
        "seed": SEED,
        "source": "native-rate raw trial JSON",
        "payload_display": "binary only; canonical b0-to-b(L-1) order",
        "selection": "pre-registered constructed matrix; not selected by outcome",
        "systems": MODELS,
        "audio_files": len(manifest),
        "clean_exact": {LABEL[m]: raw[m]["clean_attribution_exact"] for m in MODELS},
        "tracing_failure": {LABEL[m]: raw[m]["tracing_failure"] for m in MODELS},
        "reported_metrics_only": ["tracing failure", "NCA", "PESQ", "STOI", "P(bit=1)", "native decoder confidence"],
        "new_metric_created": False,
        "evidence_boundary": "single pre-registered constructed case; no population CI or system ranking",
        "command": "python scripts/build_k8_binary_detailed_report.py",
    }
    audit_path = OUT / "binary_report_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    deliverables = [md_path, audit_path] + list(TABLES.glob("*.csv")) + list(FIG.glob("*.svg"))
    validate_binary_only(deliverables, raw)
    print(json.dumps({"markdown": str(md_path), "pdf": str(pdf_path), "figures": len(list(FIG.glob('*.pdf'))), "tables": len(list(TABLES.glob('*.csv'))), "audio_aliases": len(list(AUDIO_ALIAS.rglob('*.wav')))}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build figures and a detailed Chinese report for the aligned K=5 payload case."""
from __future__ import annotations

import ast
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/tmp/mixture_report_deps")
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "k5_payload_case_trial000_20260829"
OUT = DATA / "detailed_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["audioseal", "wavmark", "timbrewm", "voicemark", "wmcodec"]
LABELS = {
    "audioseal": "AudioSeal", "wavmark": "WavMark", "timbrewm": "TimbreWM",
    "voicemark": "VoiceMark", "wmcodec": "WMCodec",
}
COLORS = {
    "audioseal": "#2878B5", "wavmark": "#54A24B", "timbrewm": "#B279A2",
    "voicemark": "#D95F59", "wmcodec": "#F2A541",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bits_of(value: int, nbits: int) -> list[int]:
    return [(value >> i) & 1 for i in range(nbits)]


def savefig(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_focus(bits_by_model: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.9), sharex=True)
    for ax, model in zip(axes, ["audioseal", "voicemark"]):
        rows = bits_by_model[model]
        x = np.arange(len(rows))
        frac = np.array([r["coalition_one_fraction"] for r in rows])
        p1 = np.array([r["p_bit_1"] for r in rows])
        hard = np.array([r["native_decoded_bit"] for r in rows])
        bars = ax.bar(x, p1, width=.72, color=COLORS[model], alpha=.83,
                      edgecolor="white", linewidth=.4, label="Decoder P(bit=1)")
        ax.scatter(x, frac, s=30, marker="D", facecolor="white", edgecolor="#111111",
                   linewidth=.9, zorder=4, label="Coalition one-fraction")
        majority = (frac >= .6).astype(int)
        dev = np.where(hard != majority)[0]
        unanimous_bad = np.where(((frac == 0) & (hard == 1)) | ((frac == 1) & (hard == 0)))[0]
        for i in dev:
            bars[i].set_edgecolor("#7A3E00")
            bars[i].set_linewidth(2.0)
            ax.text(i, min(1.04, p1[i] + .055), "×", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#7A3E00")
        for i in unanimous_bad:
            ax.annotate("unanimous\nviolation", xy=(i, p1[i]), xytext=(i-2.6, .34),
                        arrowprops=dict(arrowstyle="->", color="#A02020", lw=1.2),
                        color="#A02020", fontsize=7.5, ha="center")
        ax.axhline(.5, color="#666666", lw=.8, ls="--")
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("P(bit=1)")
        ax.set_title(f"{LABELS[model]}: decoded ID = " +
                     ("45397" if model == "audioseal" else "4341"), loc="left",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(np.arange(16), [f"b{i}" for i in range(16)])
    axes[-1].set_xlabel("Payload bit (LSB-first)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .895),
               ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Equal waveform mean does not imply bitwise-majority decoding",
                 x=.08, y=.985, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=.80, hspace=.38)
    savefig(fig, "fig_focus_audioseal_voicemark")


def plot_summary(summary: dict[str, dict], bits_by_model: dict[str, list[dict]]) -> None:
    nca = [float(summary[m]["NCA"]) for m in MODELS]
    dev = [len(ast.literal_eval(summary[m]["decoded_vs_majority_mismatch_bits"])) for m in MODELS]
    uv = [len(ast.literal_eval(summary[m]["unanimous_bit_violation_bits"])) for m in MODELS]
    conf = [np.mean([r["confidence_native_decoded_bit"] for r in bits_by_model[m]]) for m in MODELS]
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    axes[0].bar(x, nca, color=[COLORS[m] for m in MODELS])
    axes[0].set_ylim(.6, .85); axes[0].set_ylabel("NCA")
    axes[0].set_title("Nearest-colluder\nagreement", fontsize=9.2, fontweight="bold")
    axes[1].bar(x, dev, color=[COLORS[m] for m in MODELS], label="Majority deviations")
    axes[1].bar(x, uv, color="#222222", label="Unanimous violations")
    axes[1].set_ylim(0, 5.8); axes[1].set_ylabel("Bits")
    axes[1].set_title("Decoded-bit\ndepartures", fontsize=9.2, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=6.6, loc="upper left")
    axes[2].bar(x, conf, color=[COLORS[m] for m in MODELS])
    axes[2].set_ylim(.5, 1.02); axes[2].set_ylabel("Mean native-bit confidence")
    axes[2].set_title("Mean decoded-bit\nconfidence", fontsize=9.2, fontweight="bold")
    for ax in axes:
        ax.set_xticks(x, [LABELS[m] for m in MODELS], rotation=38, ha="right", fontsize=7)
        ax.grid(axis="y", color="#E0E0E0", lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("All five systems trace to a non-colluder, through different bit mechanisms",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.subplots_adjust(wspace=.48, top=.76)
    savefig(fig, "fig_system_mechanism_summary")


def plot_agreement(summary: dict[str, dict]) -> None:
    matrix = []
    annotations = []
    for m in MODELS:
        nbits = int(summary[m]["payload_nbits"])
        decoded = bits_of(int(summary[m]["decoded_identity"]), nbits)
        payloads = ast.literal_eval(summary[m]["coalition_payloads"])
        row = [np.mean(np.array(decoded) == np.array(bits_of(p, nbits))) for p in payloads]
        matrix.append(row)
        best = max(row)
        annotations.append([f"{v:.2f}" + ("*" if abs(v-best) < 1e-12 else "") for v in row])
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    im = ax.imshow(arr, vmin=.45, vmax=1, cmap="Blues", aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=8, color="white" if arr[i, j] > .72 else "#222222")
    ax.set_xticks(range(5), [f"Colluder {i}" for i in range(1, 6)])
    ax.set_yticks(range(5), [LABELS[m] for m in MODELS])
    ax.set_title("Bit agreement between decoded identity and each colluder", fontsize=11,
                 fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    cbar.set_label("Bit agreement")
    ax.text(0, 5.25, "* nearest colluder; no decoded identity equals any colluder",
            fontsize=7.5, color="#555555", transform=ax.transData)
    savefig(fig, "fig_decoded_colluder_agreement")


def plot_fraction_response(bits_by_model: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    rng = np.random.default_rng(20260829)
    for m in MODELS:
        rows = bits_by_model[m]
        x = np.array([r["coalition_one_fraction"] for r in rows])
        y = np.array([r["p_bit_1"] for r in rows])
        jitter = rng.normal(0, .009, len(x))
        ax.scatter(x+jitter, y, s=24, alpha=.72, color=COLORS[m], label=LABELS[m],
                   edgecolor="white", linewidth=.3)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#333333", label="y = coalition fraction")
    ax.axhline(.5, color="#AAAAAA", lw=.7)
    ax.set_xlim(.15, 1.04); ax.set_ylim(-.04, 1.04)
    ax.set_xticks([.2, .4, .6, .8, 1.0])
    ax.set_xlabel("Fraction of five colluders carrying bit 1")
    ax.set_ylabel("Decoder P(bit=1)")
    ax.set_title("Decoder response to coalition bit composition (single case)",
                 fontsize=11, fontweight="bold")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="lower right")
    ax.grid(color="#E3E3E3", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig_coalition_fraction_response")


INK = colors.HexColor("#242A31")
MUTED = colors.HexColor("#626C77")
BLUE = colors.HexColor("#2878B5")
LIGHT = colors.HexColor("#F3F6F8")
RULE = colors.HexColor("#D7DDE3")


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                         topMargin=17*mm, bottomMargin=15*mm,
                         title="K=5 五系统合谋 payload 案例详细分析")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(RULE); canvas.line(15*mm, A4[1]-11*mm, A4[0]-15*mm, A4[1]-11*mm)
            canvas.setFont("STSong-Light", 7.2); canvas.setFillColor(MUTED)
            canvas.drawString(15*mm, A4[1]-8.7*mm, "K=5 五系统合谋 payload 案例分析")
        canvas.setFont("Helvetica", 7); canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0]-15*mm, 8*mm, str(doc.page))
        canvas.restoreState()


def styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="STSong-Light", fontSize=22,
                                leading=30, alignment=TA_LEFT, textColor=INK, spaceAfter=5*mm),
        "sub": ParagraphStyle("sub", fontName="STSong-Light", fontSize=10.5, leading=17,
                              textColor=MUTED, spaceAfter=6*mm),
        "h1": ParagraphStyle("h1", fontName="STSong-Light", fontSize=14, leading=20,
                             textColor=INK, spaceBefore=2*mm, spaceAfter=3*mm),
        "h2": ParagraphStyle("h2", fontName="STSong-Light", fontSize=11, leading=16,
                             textColor=BLUE, spaceBefore=2*mm, spaceAfter=1.5*mm),
        "body": ParagraphStyle("body", fontName="STSong-Light", fontSize=9.2, leading=14.7,
                               textColor=INK, spaceAfter=2.0*mm),
        "small": ParagraphStyle("small", fontName="STSong-Light", fontSize=7.5, leading=11,
                                textColor=MUTED, spaceAfter=1.5*mm),
        "bullet": ParagraphStyle("bullet", fontName="STSong-Light", fontSize=9, leading=14.2,
                                 leftIndent=5*mm, firstLineIndent=-3.5*mm, bulletIndent=1.5*mm,
                                 textColor=INK, spaceAfter=1.2*mm),
        "caption": ParagraphStyle("caption", fontName="STSong-Light", fontSize=7.5, leading=11,
                                  textColor=MUTED, spaceBefore=1.2*mm, spaceAfter=2.5*mm),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, st: dict) -> Paragraph:
    return P(f"<bullet>•</bullet>{text}", st["bullet"])


def table(rows: list[list[str]], widths: list[float], fontsize: float = 7.8) -> Table:
    cooked = []
    for ri, row in enumerate(rows):
        cooked.append([P(str(v), ParagraphStyle(f"c{ri}_{ci}", fontName="STSong-Light",
                                                fontSize=fontsize, leading=fontsize+2.2,
                                                alignment=TA_LEFT if ci == 0 else TA_CENTER,
                                                textColor=colors.white if ri == 0 else INK))
                       for ci, v in enumerate(row)])
    t = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
    cmds = [("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("BACKGROUND", (0,0), (-1,0), INK),
            ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LINEBELOW", (0,0), (-1,-1), .35, RULE)]
    for i in range(2, len(rows), 2): cmds.append(("BACKGROUND", (0,i), (-1,i), LIGHT))
    t.setStyle(TableStyle(cmds)); return t


def fit(path: Path, maxw: float, maxh: float) -> Image:
    from PIL import Image as PILImage
    with PILImage.open(path) as im: w, h = im.size
    scale = min(maxw/w, maxh/h)
    return Image(str(path), width=w*scale, height=h*scale)


def build_markdown(summary: dict[str, dict], bits_by_model: dict[str, list[dict]], audit: dict) -> None:
    lines = [
        "# K=5 五系统合谋 payload 案例：带图详细分析", "",
        f"- 样本：`{audit['spk']}`，`local_t={audit['local_t']}`，`trial_id={audit['trial_id']}`。",
        "- 五条水印音频做等权 waveform mean，权重均为 0.2。",
        "- 这是按完整性选择的首个对齐案例，不按结果挑选；用于机制展示，不替代总体统计。", "",
        "## 核心结论", "",
        "1. 五个系统都输出了不属于五个合谋者的身份，因此本例中 tracing failure 均为 1。",
        "2. 等权 waveform mean 不等于逐 bit 多数投票。AudioSeal/WavMark 偏离多数位 3 个，VoiceMark 4 个，WMCodec 5 个。",
        "3. TimbreWM 恰好输出 bitwise-majority mosaic，但该 mosaic 本身也不是任一合谋者，因此仍然逃逸。",
        "4. VoiceMark 和 WMCodec 在所有合谋者都为 1 的 b15 上解码为 0；这是 unanimous-bit violation，说明系统存在明显的跨 bit、chunk 或 digit 耦合。",
        "5. 本案例能够解释逃逸如何发生，不能据此声称某系统总体更脆弱；总体结论必须依赖 300-trial 统计。", "",
        "## 系统摘要", "",
        "| System | Decoded ID | Majority deviations | Unanimous violations | NCA | TF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        s = summary[m]
        lines.append(f"| {LABELS[m]} | {s['decoded_identity']} | {len(ast.literal_eval(s['decoded_vs_majority_mismatch_bits']))} | {len(ast.literal_eval(s['unanimous_bit_violation_bits']))} | {float(s['NCA']):.4f} | {s['tracing_failure']} |")
    lines += ["", "## 图表", "",
              "- `fig_focus_audioseal_voicemark.*`：AudioSeal 与 VoiceMark 的逐 bit 直观对照。",
              "- `fig_system_mechanism_summary.*`：五系统 NCA、偏离位和平均置信度。",
              "- `fig_decoded_colluder_agreement.*`：最终身份与五个合谋者的逐 bit agreement。",
              "- `fig_coalition_fraction_response.*`：coalition bit composition 与 decoder response。", "",
              "## 证据边界", "",
              "置信度在不同系统中的参数化并不完全相同；WavMark 使用有效滑窗 vote fraction，不能当作官方 neural posterior。单案例的系统差异只能作为机制说明，不能作为总体性能排序。", ""]
    (OUT / "K5_PAYLOAD_CASE_DETAILED_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def build_pdf(summary: dict[str, dict], bits_by_model: dict[str, list[dict]], audit: dict) -> None:
    st = styles(); output = OUT / "K5_Payload_Case_Detailed_Analysis.pdf"
    doc = ReportDoc(str(output)); story = []
    story += [Spacer(1, 8*mm), P("K=5 五系统合谋 payload", st["title"]),
              P("一个等权 waveform-mean 案例的逐 bit 置信度、mosaic identity 与 tracing failure 机制", st["sub"])]
    fact_rows = [["案例", "合谋规模", "混合权重", "结果"],
                 ["trial 0", "K=5", "0.2 × 5", "五系统均 TF=1"]]
    story += [table(fact_rows, [42*mm, 40*mm, 46*mm, 52*mm], 9), Spacer(1, 6*mm),
              P("一页结论", st["h1"]),
              bullet("五套 decoder 最终都输出了<b>不属于五个合谋者</b>的身份；但产生逃逸的 bit-level 路径并不相同。", st),
              bullet("AudioSeal 与 WavMark 各偏离 coalition bitwise majority 3 位；VoiceMark 偏离 4 位，WMCodec 偏离 5 位。", st),
              bullet("TimbreWM 完全等于 bitwise-majority mosaic，说明即使每一位都按多数选择，拼接后的身份仍可落到非合谋者。", st),
              bullet("VoiceMark 与 WMCodec 在 b15 上违反五人一致的 1，表现出跨 bit/chunk/digit 耦合；这不是简单独立 bit 投票可解释的。", st),
              bullet("这是按数据完整性选出的第一个跨五系统对齐案例，不是按攻击效果挑选。它用于解释机制，不承担总体统计结论。", st),
              Spacer(1, 3*mm), fit(OUT / "fig_system_mechanism_summary.png", doc.width, 66*mm),
              P("图 1　五系统机制总览。NCA 越高表示最终 payload 越接近某个合谋者；偏离位统计相对五人 bitwise majority；unanimous violation 表示 decoder 甚至违背了五人完全一致的 bit。置信度只在系统内部解释。", st["caption"]),
              PageBreak()]

    story += [P("1　案例配置与五个 payload", st["h1"]),
              P(f"内容键为 <b>{audit['spk']}</b>，local_t={audit['local_t']}。对每个系统取同一内容的五条水印音频，做 full-waveform 等权平均。16-bit 系统共享同一组身份；TimbreWM 使用其原生 10-bit payload。所有表均采用 LSB-first，b0 为最低位。", st["body"])]
    payload_rows = [["合谋者", "16-bit ID", "Hex / MSB→LSB", "TimbreWM ID", "10-bit MSB→LSB"]]
    p16 = [38853,41735,47835,55549,61557]; p10=[604,650,745,867,960]
    for i,(a,b) in enumerate(zip(p16,p10),1):
        payload_rows.append([f"C{i}", str(a), f"0x{a:04X} / {a:016b}", str(b), f"{b:010b}"])
    story += [table(payload_rows, [25*mm, 29*mm, 66*mm, 30*mm, 40*mm], 7.6), Spacer(1, 4*mm),
              P("一致位与冲突位", st["h2"]),
              bullet("16-bit 系统：b0 和 b15 在五人中均为 1；b1–b14 均存在冲突；没有 unanimous-0 位。", st),
              bullet("TimbreWM：仅 b9 为 unanimous-1；b0–b8 均冲突；没有 unanimous-0 位。", st),
              bullet("因此这个案例对 decoder 的冲突处理具有较高信息密度，同时仍保留可检查的 unanimous anchor。", st),
              fit(DATA / "fig_k5_payload_case_all_systems.png", doc.width, 128*mm),
              P("图 2　五系统逐 bit 总览。左侧为五个合谋者的 bit 矩阵；右侧彩色柱为 decoder P(bit=1)，黑色菱形为五人中 bit=1 的比例。绿色框表示 unanimous 位，棕色标记表示最终 hard bit 偏离多数。", st["caption"]),
              PageBreak()]

    story += [P("2　AudioSeal 与 VoiceMark：最直观的对照", st["h1"]),
              fit(OUT / "fig_focus_audioseal_voicemark.png", doc.width, 124*mm),
              P("图 3　AudioSeal 与 VoiceMark 的相同 16-bit coalition 输入。柱高是 P(bit=1)，白心菱形是 coalition one-fraction；棕色边框和 × 标记 majority deviation。", st["caption"]),
              P("AudioSeal", st["h2"]),
              bullet("b7、b8、b9 三个位偏离五人多数；但 b0、b15 两个 unanimous-1 位均被保留。", st),
              bullet("多数冲突位的概率仍处于 0.2–0.8 的中间范围，hard identity 是多个合谋者 bit 的组合：45397。", st),
              P("VoiceMark", st["h2"]),
              bullet("b5、b9、b13、b15 四个位偏离多数，其中 b15 是最强异常：五个合谋者都为 1，但 decoder 以接近 1 的 native confidence 输出 0。", st),
              bullet("b12–b15 的概率呈现近饱和状态，说明 chunk-level decision 可把多个 bits 联动推向某个离散 code state，而不是逐 bit 平滑平均。", st),
              P("准确结论：这个案例显示 VoiceMark 的输出更离散、耦合更强；但单案例不能证明 VoiceMark 的总体 tracing failure 更高。", st["body"]),
              PageBreak()]

    story += [P("3　最终身份与五个合谋者的关系", st["h1"]),
              fit(OUT / "fig_decoded_colluder_agreement.png", 145*mm, 78*mm),
              P("图 4　每个 decoded identity 与五个合谋者的 bit agreement。星号是最近合谋者，矩阵中没有 1.00，因此五个输出都不是任何一个合谋者。", st["caption"])]
    rows = [["系统", "Decoded ID", "Majority ID", "偏离多数位", "违反一致位", "NCA", "TF"]]
    for m in MODELS:
        s=summary[m]
        rows.append([LABELS[m], s["decoded_identity"], s["bitwise_majority_identity"],
                     s["decoded_vs_majority_mismatch_bits"], s["unanimous_bit_violation_bits"],
                     f"{float(s['NCA']):.4f}", s["tracing_failure"]])
    story += [table(rows, [29*mm,25*mm,26*mm,43*mm,33*mm,19*mm,14*mm], 6.8), Spacer(1, 4*mm),
              P("两种不同的逃逸机制", st["h2"]),
              bullet("<b>Mosaic escape：</b>TimbreWM 的 decoded payload 完全等于五人逐 bit 多数结果，但该组合身份不属于任何合谋者。", st),
              bullet("<b>Decoder-coupled escape：</b>VoiceMark/WMCodec 不仅形成 mosaic，还在 unanimous b15 上输出相反 bit，进一步远离五个源身份。", st),
              bullet("NCA 从 AudioSeal/WavMark 的 0.8125 降到 WMCodec 的 0.6875，表示后者离最近合谋者更远；这仍是单案例描述，不是系统总体排序。", st),
              PageBreak()]

    story += [P("4　coalition bit composition 与 decoder response", st["h1"]),
              fit(OUT / "fig_coalition_fraction_response.png", 150*mm, 96*mm),
              P("图 5　横轴是五人中 bit=1 的比例，纵轴是 decoder P(bit=1)。虚线 y=x 只是“独立线性平均”的视觉参照，不是理论期望。每个点对应本案例中的一个 bit。", st["caption"]),
              bullet("AudioSeal 的点大多位于中间概率区，和 coalition composition 的方向较一致，但仍存在 b8 等反多数决策。", st),
              bullet("WavMark、VoiceMark 与 WMCodec 出现大量接近 0 或 1 的响应；这说明 waveform averaging 后的 decoder confidence 不等于 coalition vote fraction。", st),
              bullet("WavMark 的数值是 44 个有效起始模式滑窗中的 vote fraction，不是官方 neural posterior；因此不能与其他系统做绝对 calibration 比较。", st),
              bullet("该散点图只说明本案例的非线性/离散响应，不能从 16 个或 10 个相关 bits 计算独立样本显著性。", st),
              PageBreak()]

    story += [P("5　逐系统解释与论文可用表述", st["h1"])]
    system_notes = {
        "audioseal": "保留两个 unanimous 位，只有三个 majority deviations；最终 ID 由不同合谋者 bits 拼接而成。",
        "wavmark": "NCA 与 AudioSeal 相同，但偏离位置不同；滑窗投票在多个冲突位上接近饱和。",
        "timbrewm": "严格等于 bitwise-majority identity，却仍然不是任何合谋者，是最干净的 mosaic-identity 示范。",
        "voicemark": "出现四个 majority deviations 和一次 unanimous violation；native chunk posterior 对若干 bits 给出近确定但非多数的输出。",
        "wmcodec": "majority deviations 最多且 NCA 最低，并同样违反 b15；体现 digit/chunk 联合决策可产生远离成员的 payload。",
    }
    for m in MODELS:
        story += [P(LABELS[m], st["h2"]), P(system_notes[m], st["body"])]
    story += [P("建议正文表述", st["h2"]),
              P("In an aligned K=5 case, equal waveform averaging yielded a non-colluder identity for all five systems. The decoded payloads were not generally equivalent to bitwise majority voting: AudioSeal and WavMark each departed from the coalition majority at three bits, while VoiceMark and WMCodec also violated a unanimous coalition bit. TimbreWM produced the exact bitwise-majority mosaic, which nevertheless corresponded to no colluder. This case illustrates two complementary escape mechanisms—mosaic recombination and decoder-coupled bit transitions—without implying a population-level ranking among systems.", st["body"]),
              P("审稿边界", st["h2"]),
              bullet("不要把该案例写成五系统总体比较；总体比例必须引用完整 trials。", st),
              bullet("不要把 decoder confidence 跨系统直接比较为校准优劣。", st),
              bullet("不要把 unanimous violation 直接归因于某一种网络结构；目前只能表述为与跨 bit/chunk coupling 一致。", st),
              bullet("不要把整数身份的数值距离当作 payload 距离；只使用 Hamming agreement/NCA。", st),
              PageBreak()]

    story += [P("附录　完整逐 bit 置信度", st["h1"]),
              P("P1 为 P(bit=1)；Hard 为 native decoded bit；C1…C5 是五个合谋者的 bit。所有 bit 顺序均为 LSB-first。", st["body"])]
    for mi,m in enumerate(MODELS):
        rows=[["Bit","C1","C2","C3","C4","C5","ones/5","P1","Hard","Conf"]]
        for r in bits_by_model[m]:
            rows.append([f"b{r['bit_index_lsb_first']}"]+[str(r[f"colluder_{i}_bit"]) for i in range(1,6)]+
                        [str(r["ones_among_5"]), f"{r['p_bit_1']:.3f}", str(r["native_decoded_bit"]),
                         f"{r['confidence_native_decoded_bit']:.3f}"])
        story += [P(LABELS[m], st["h2"]), table(rows, [17*mm]+[11*mm]*5+[18*mm,20*mm,17*mm,20*mm], 6.4), Spacer(1, 3*mm)]
        if mi in (1,3): story.append(PageBreak())
    story += [Spacer(1, 3*mm), P("数据与复现", st["h2"]),
              P(f"源数据：{DATA}。选择规则：{audit['selection']}。hard-bit 核验：{audit['hard_bit_validation']}。报告生成命令：python scripts/build_k5_payload_case_detailed_report.py", st["small"])]
    doc.build(story)


def main() -> None:
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.5, "axes.labelsize":8.5,
                         "xtick.labelsize":7.5, "ytick.labelsize":7.5, "legend.fontsize":7.5,
                         "figure.dpi":150, "savefig.dpi":300})
    summary_rows=read_csv("k5_payload_case_system_summary.csv")
    summary={r["model"]:r for r in summary_rows}
    bit_rows=read_csv("k5_payload_case_bits.csv")
    bits_by_model={m:[] for m in MODELS}
    for r in bit_rows:
        cooked=dict(r)
        for k in ["bit_index_lsb_first","colluder_1_bit","colluder_2_bit","colluder_3_bit",
                  "colluder_4_bit","colluder_5_bit","ones_among_5","native_decoded_bit"]:
            cooked[k]=int(cooked[k])
        for k in ["coalition_one_fraction","p_bit_1","confidence_native_decoded_bit"]:
            cooked[k]=float(cooked[k])
        bits_by_model[r["model"]].append(cooked)
    audit=json.loads((DATA/"analysis_audit.json").read_text(encoding="utf-8"))
    plot_focus(bits_by_model); plot_summary(summary,bits_by_model)
    plot_agreement(summary); plot_fraction_response(bits_by_model)
    build_markdown(summary,bits_by_model,audit); build_pdf(summary,bits_by_model,audit)
    print(OUT)


if __name__ == "__main__":
    main()

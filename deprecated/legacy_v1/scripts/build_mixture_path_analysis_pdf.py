#!/usr/bin/env python3
"""Build a Chinese PDF report for the adaptive K=5 one-bit mixture-path study."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/mixture_report_deps")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mixture_path_k5_adaptive_20260829"
ANALYSIS = DATA / "analysis"
OUTPUT = ANALYSIS / "Mixture_Path_Analysis_Report_K5.pdf"

BLUE = colors.HexColor("#2878B5")
RED = colors.HexColor("#D95F59")
INK = colors.HexColor("#222831")
MUTED = colors.HexColor("#5F6B78")
LIGHT_BLUE = colors.HexColor("#EAF3FA")
LIGHT_RED = colors.HexColor("#FBECEB")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
RULE = colors.HexColor("#D7DCE2")


def load_rows(name: str) -> list[dict]:
    with (ANALYSIS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                         topMargin=18*mm, bottomMargin=16*mm,
                         title="K=5 自适应 1-bit Mixture-Path 实验分析",
                         author="Experiment analysis pipeline")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(RULE); canvas.setLineWidth(.5)
            canvas.line(16*mm, A4[1]-12*mm, A4[0]-16*mm, A4[1]-12*mm)
            canvas.setFont("STSong-Light", 7.5); canvas.setFillColor(MUTED)
            canvas.drawString(16*mm, A4[1]-9.7*mm, "自适应 1-bit Mixture-Path 分析 · K=5")
        canvas.setFont("Helvetica", 7.2); canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0]-16*mm, 8.5*mm, f"{doc.page}")
        canvas.restoreState()


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def styled_table(data, widths, header=True, alignments=None, font_size=8.3,
                 first_col_blue=False) -> Table:
    cooked = []
    for ri, row in enumerate(data):
        cooked_row = []
        for ci, value in enumerate(row):
            align = (alignments[ci] if alignments else (TA_LEFT if ci == 0 else TA_CENTER))
            font = "STSong-Light"
            color = colors.white if header and ri == 0 else INK
            style = ParagraphStyle(f"cell_{ri}_{ci}", fontName=font, fontSize=font_size,
                                   leading=font_size+2, alignment=align, textColor=color)
            cooked_row.append(Paragraph(str(value), style))
        cooked.append(cooked_row)
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), .35, RULE),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), INK),
                     ("LINEBELOW", (0, 0), (-1, 0), .8, INK)]
        for row in range(1, len(data)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GRAY))
    if first_col_blue and len(data) > 1:
        commands.append(("TEXTCOLOR", (0, 1), (0, -1), BLUE))
    table.setStyle(TableStyle(commands))
    return table


def bullet(text: str, styles) -> Paragraph:
    return p(f"<bullet>•</bullet>{text}", styles["bullet"])


def make_styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="STSong-Light",
                                fontSize=24, leading=31, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=7*mm),
        "subtitle": ParagraphStyle("subtitle", fontName="STSong-Light", fontSize=11.5,
                                   leading=18, textColor=MUTED, spaceAfter=7*mm),
        "h1": ParagraphStyle("h1", fontName="STSong-Light", fontSize=15, leading=20,
                             textColor=INK, spaceBefore=3*mm, spaceAfter=3.2*mm,
                             borderPadding=(0, 0, 2, 0)),
        "h2": ParagraphStyle("h2", fontName="STSong-Light", fontSize=11.5, leading=16,
                             textColor=BLUE, spaceBefore=2.5*mm, spaceAfter=1.8*mm),
        "body": ParagraphStyle("body", fontName="STSong-Light", fontSize=9.4,
                               leading=15.3, textColor=INK, spaceAfter=2.3*mm,
                               alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName="STSong-Light", fontSize=7.8,
                                leading=11.5, textColor=MUTED, spaceAfter=1.5*mm),
        "bullet": ParagraphStyle("bullet", fontName="STSong-Light", fontSize=9.2,
                                 leading=14.5, leftIndent=5*mm, firstLineIndent=-3.5*mm,
                                 bulletIndent=1.5*mm, textColor=INK, spaceAfter=1.3*mm),
        "callout": ParagraphStyle("callout", fontName="STSong-Light", fontSize=10,
                                  leading=16, leftIndent=4*mm, rightIndent=4*mm,
                                  textColor=INK, spaceBefore=2*mm, spaceAfter=2*mm),
        "caption": ParagraphStyle("caption", fontName="STSong-Light", fontSize=7.8,
                                  leading=11.5, textColor=MUTED, alignment=TA_LEFT,
                                  spaceBefore=1.5*mm, spaceAfter=3*mm),
        "cover_big": ParagraphStyle("cover_big", fontName="Helvetica-Bold", fontSize=31,
                                    leading=34, textColor=BLUE, alignment=TA_CENTER),
        "cover_label": ParagraphStyle("cover_label", fontName="STSong-Light", fontSize=8.2,
                                      leading=11, textColor=MUTED, alignment=TA_CENTER),
    }


def fit_image(path: Path, max_w: float, max_h: float) -> Image:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w*scale, height=h*scale)


def build() -> None:
    comp_rows = load_rows("mixture_path_system_comparison.csv")
    comp = {r["metric"]: r for r in comp_rows}
    diag_rows = load_rows("mixture_path_transition_diagnostics.csv")
    diag = {r["model"]: r for r in diag_rows}
    audit = json.loads((DATA / "analysis_audit.json").read_text(encoding="utf-8"))
    analysis_audit = json.loads((ANALYSIS / "analysis_audit.json").read_text(encoding="utf-8"))
    styles = make_styles()
    doc = ReportDoc(str(OUTPUT))
    story = []

    # Cover and executive summary.
    story += [Spacer(1, 10*mm), p("K=5 自适应 1-bit", styles["cover_big"]),
              p("Mixture-Path 实验分析报告", styles["title"]),
              p("AudioSeal 与 VoiceMark 的置信度轨迹、payload 路径几何和身份转折行为", styles["subtitle"])]
    facts = [
        ["600", "12,171", "100", "10,000"],
        ["总 trials", "路径采样点", "speaker clusters", "bootstrap 次数"],
    ]
    fact_table = Table([[p(v, styles["cover_big"] if r == 0 else styles["cover_label"])
                         for v in row] for r, row in enumerate(facts)],
                       colWidths=[doc.width/4]*4)
    fact_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("BACKGROUND", (1, 0), (1, -1), LIGHT_GRAY),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_RED),
        ("BACKGROUND", (3, 0), (3, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), .4, RULE),
        ("INNERGRID", (0, 0), (-1, -1), .35, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [fact_table, Spacer(1, 7*mm), p("核心结论", styles["h1"])]
    story += [
        bullet("<b>AudioSeal 的路径更平滑。</b> Target probability R<super>2</super> 为 0.862，目标 bit 单调率为 97.3%。", styles),
        bullet("<b>VoiceMark 的端点更确定。</b> 目标概率从 0.017 上升至 0.986，但中间 identity path 更离散。", styles),
        bullet("VoiceMark 在 193/300 trials 中出现多次 identity transition；AudioSeal 仅为 1/300。", styles),
        bullet("因此结果支持“连续 bitwise path 与离散 chunk-level path 的结构差异”，不支持笼统的系统优劣结论。", styles),
    ]
    callout = Table([[p("最准确的论文表述：AudioSeal exhibits a smoother and predominantly monotonic one-bit interpolation path, whereas VoiceMark produces more decisive endpoint predictions but substantially less stable intermediate identity transitions.", styles["callout"])]], colWidths=[doc.width])
    callout.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                                 ("BOX", (0, 0), (-1, -1), .7, BLUE),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                 ("TOPPADDING", (0, 0), (-1, -1), 7),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story += [Spacer(1, 3*mm), callout, Spacer(1, 9*mm),
              p("生成日期：2026-08-29　·　随机种子：20260829　·　完整数据目录见报告末页", styles["small"]),
              PageBreak()]

    # Design and definitions.
    story += [p("1　实验设计与数据完整性", styles["h1"]),
              p("本实验比较同一内容下仅相差 1 bit 的两个水印副本，并沿线性混合路径观察 decoder 的连续置信度和离散身份输出：", styles["body"]),
              p("y(λ) = (1 − λ)x<sub>A</sub> + λx<sub>B</sub>", ParagraphStyle(
                  "eq", parent=styles["body"], fontName="Helvetica", fontSize=13,
                  leading=18, alignment=TA_CENTER, textColor=INK, spaceBefore=2*mm,
                  spaceAfter=3*mm))]
    design_data = [
        ["项目", "配置"],
        ["系统", "AudioSeal、VoiceMark"],
        ["联盟规模", "固定 K=5"],
        ["样本量", "每系统 300 trials；共 100 speakers，每 speaker 3 trials"],
        ["粗粒度采样", "λ = 0, 0.1, …, 1，共 11 点"],
        ["局部加密", "指定 bit 发生 hard transition 的每个粗区间，再以 0.01 粒度细分"],
        ["fallback", "若无指定 bit hard crossing，则细分目标 bit 概率变化最大的粗区间"],
        ["保存信息", "完整 bit probabilities/logits、hard payload、identity、confidence、margin 和混合音频"],
    ]
    story += [styled_table(design_data, [38*mm, 139*mm], font_size=8.5),
              Spacer(1, 4*mm), p("指标定义", styles["h2"])]
    story += [
        bullet("<b>Direction-normalized target probability：</b>无论原始变化是 0→1 还是 1→0，统一表示 decoder 分配给 flipped endpoint bit 的概率。", styles),
        bullet("<b>Primary path R<super>2</super>：</b>只使用均匀的 11 个粗粒度点计算，防止局部加密改变回归权重。", styles),
        bullet("<b>Monotonicity：</b>使用全部粗粒度和加密点，检查目标 bit 概率是否沿 endpoint 方向单调。", styles),
        bullet("<b>Identity transition：</b>相邻 λ 点之间 native decoded identity 发生变化的次数。", styles),
        bullet("<b>置信区间：</b>10,000 次 speaker-cluster bootstrap；以 speaker 而非单个 trial 重采样。", styles),
    ]
    story += [p("完整性检查", styles["h2"]),
              p(f"AudioSeal 共 6,000 个 path points；VoiceMark 共 6,171 个。两套系统的 300 个 trial ID 和 speaker/content key 完全对齐。所有点均具有 16 维 finite bit probability/logit、native hard bits 和有效音频路径。VoiceMark 多出的 171 个点来自 10 个包含多个目标-bit 转折区间的 trials。", styles["body"]),
              PageBreak()]

    # Main figure.
    story += [p("2　总体轨迹与路径稳定性", styles["h1"])]
    story.append(fit_image(ANALYSIS / "fig_mixture_path_overview.png", doc.width, 82*mm))
    story.append(p("图 1　(a) Direction-normalized target-bit probability。深色窄带为均值的 95% speaker-cluster bootstrap CI，浅色带为 trial-level 10–90% 范围；(b) 每个 trial 的 coarse-grid target probability R² 分布；(c) 三个路径稳定性指标及 95% CI。", styles["caption"]))
    story += [p("图中最重要的信息", styles["h2"]),
              bullet("AudioSeal 的均值轨迹变化更缓，trial 间异质性也较小；其概率不会在端点逼近 0/1。", styles),
              bullet("VoiceMark 的平均曲线看似平滑，但浅红色 10–90% 区间非常宽，说明单个 trial 往往不是平滑中间态，而是在不同 λ 位置突然跳变。", styles),
              bullet("AudioSeal 与 VoiceMark 的 probability R² 差值为 0.162 [0.146, 0.179]；目标 bit 单调率差值为 58.3 percentage points [52.3, 64.0]。", styles),
              bullet("这一区别不是失真指标，也不直接等价于攻击成功率；它描述的是 decoder 沿同一 waveform mixture path 的响应几何。", styles),
              PageBreak()]

    # Main table.
    story += [p("3　系统级配对结果", styles["h1"])]
    metrics = [
        ("target_bit_probability_r2", "Target probability R<super>2</super>", 3, ""),
        ("target_bit_logit_r2", "Target logit R<super>2</super>", 3, ""),
        ("target_bit_monotonic", "目标 bit 单调路径", 1, "%"),
        ("single_identity_transition", "仅一次 identity transition", 1, "%"),
        ("only_designated_endpoint_change", "端点只改变指定 bit", 1, "%"),
        ("target_probability_span", "端点目标概率跨度", 3, ""),
        ("identity_transition_count", "每路径 identity transitions", 3, ""),
        ("both_endpoints_exact", "两个端点 identity 均精确", 1, "%"),
    ]
    main_table = [["指标", "AudioSeal", "VoiceMark", "配对差值 AS−VM [95% CI]"]]
    for key, label, dec, suffix in metrics:
        r = comp[key]
        fmt = f"{{:.{dec}f}}"
        main_table.append([
            label,
            fmt.format(float(r["audioseal_mean"])) + suffix,
            fmt.format(float(r["voicemark_mean"])) + suffix,
            fmt.format(float(r["paired_difference_audioseal_minus_voicemark"])) +
            " [" + fmt.format(float(r["difference_ci95_low"])) + ", " +
            fmt.format(float(r["difference_ci95_high"])) + "]" + (" pp" if suffix == "%" else ""),
        ])
    story += [styled_table(main_table, [56*mm, 28*mm, 28*mm, 65*mm], font_size=7.7),
              Spacer(1, 4*mm)]
    story += [p("结果解释", styles["h2"]),
              bullet("AudioSeal 的 probability-space 和 logit-space R² 均更高；其中 probability-space 差异更大。", styles),
              bullet("VoiceMark 的 endpoint probability span 更大（0.969 vs. 0.544），说明端点决策更确定；这个结果与其路径不稳定性同时成立。", styles),
              bullet("AudioSeal 的 300 个 trials 均在两个端点只改变指定 bit；VoiceMark 为 275/300。", styles),
              bullet("VoiceMark 仅 269/300 trials 在两个端点都精确解码为指定 payload，因此其一部分 path irregularity 与 endpoint decoding error 有关，但不能解释全部 193 个多重 identity-transition trials。", styles),
              PageBreak()]

    # Transition diagnostics + representative figure.
    story += [p("4　转折诊断与代表案例", styles["h1"])]
    dtable = [
        ["诊断（每系统 300 trials）", "AudioSeal", "VoiceMark"],
        ["无指定 bit hard transition", diag["audioseal"]["zero_target_transitions"], diag["voicemark"]["zero_target_transitions"]],
        ["多次指定 bit transition", diag["audioseal"]["multiple_target_transitions"], diag["voicemark"]["multiple_target_transitions"]],
        ["多次 identity transition", diag["audioseal"]["multiple_identity_transitions"], diag["voicemark"]["multiple_identity_transitions"]],
        ["端点出现非指定 bit 变化", diag["audioseal"]["trials_with_unintended_endpoint_changes"], diag["voicemark"]["trials_with_unintended_endpoint_changes"]],
        ["两个端点均精确解码", diag["audioseal"]["both_endpoints_exact"], diag["voicemark"]["both_endpoints_exact"]],
        ["fallback 加密", diag["audioseal"]["fallback_trials"], diag["voicemark"]["fallback_trials"]],
    ]
    story += [styled_table(dtable, [105*mm, 36*mm, 36*mm], font_size=8.4), Spacer(1, 4*mm)]
    story.append(fit_image(ANALYSIS / "fig_mixture_path_representative.png", 108*mm, 102*mm))
    story.append(p("图 2　客观代表案例。案例选择规则为：在系统的 modal monotonicity 与 modal identity-transition pattern 中，选择 probability R² 最接近系统中位数的 trial。彩色曲线为 target-bit probability；灰色阶梯为 decoded identity 相对 base identity 的 Hamming distance；黄色区域标记 identity transition bracket。", styles["caption"]))
    story += [bullet("AudioSeal 代表案例呈现单一、局部化的 identity transition，指定 bit 的概率在转折附近连续变化。", styles),
              bullet("VoiceMark 代表案例的目标 bit 很快趋近 1，但 decoded identity 先跳到距 base 三个 bits 的状态，随后又回到仅相差一个 bit 的目标 identity。", styles),
              PageBreak()]

    # Interpretation, limitations, reproducibility.
    story += [p("5　论文解释、证据边界与复现信息", styles["h1"]),
              p("推荐的机制解释", styles["h2"]),
              p("AudioSeal 的 bitwise detector 在 waveform interpolation 下呈现较连续的单-bit 响应；VoiceMark 的 4×16-way chunk decision 更容易形成离散 identity state changes。该结果与 payload representation 影响 mixture-path geometry 的解释一致。", styles["body"]),
              p("但该实验不能单独证明 decoder architecture 是差异的唯一原因，因为系统还同时改变了训练数据、embedding objective、模型容量与置信度参数化。正文应使用“is consistent with”或“exhibits”，不应使用“causes”。", styles["body"]),
              p("必须保留的限制", styles["h2"])]
    story += [
        bullet("两套系统的 confidence parameterization 不同：AudioSeal 使用 bit logits，VoiceMark 的 bit probability 来自 16-way chunk distribution 的精确边缘化；原始数值未做跨系统 calibration。", styles),
        bullet("R² 描述线性规则性，不是 tracing failure、攻击成功率或音质指标。", styles),
        bullet("分析只覆盖 AudioSeal、VoiceMark、K=5 和 one-bit perturbation，不能直接推广到全部五个系统或其他 K。", styles),
        bullet("Primary R² 只使用均匀粗网格；adaptive-grid R² 仅作为诊断，避免在 transition 附近过采样造成加权偏差。", styles),
        bullet("所有 negative、multi-transition 和 fallback trials 均保留；没有按结果筛选案例或删除异常系统。", styles),
    ]
    story += [p("统计审查", styles["h2"]),
              p("所有系统差值按完全对齐的 trial/content key 计算，再以 speaker 为 cluster 进行 10,000 次 bootstrap。每个 path 是分析单位；λ 点没有被当作独立样本。已检查 11 类常见统计谬误，主要风险是跨系统 confidence calibration、探索性分析和过度推广，而不是样本伪重复。", styles["body"]),
              p("关键文件", styles["h2"])]
    files = [
        ("逐点数据", DATA / "mixture_path_audioseal_points.csv"),
        ("逐点数据", DATA / "mixture_path_voicemark_points.csv"),
        ("逐 trial 汇总", DATA / "mixture_path_trial_summary.csv"),
        ("系统比较", ANALYSIS / "mixture_path_system_comparison.csv"),
        ("粗网格轨迹", ANALYSIS / "mixture_path_coarse_trajectory.csv"),
        ("转折诊断", ANALYSIS / "mixture_path_transition_diagnostics.csv"),
        ("完整文字分析", ANALYSIS / "RESULTS_ANALYSIS.md"),
    ]
    file_table = [["类型", "路径"]] + [[kind, str(path)] for kind, path in files]
    story += [styled_table(file_table, [30*mm, 147*mm], font_size=6.6), Spacer(1, 3*mm),
              p(f"随机种子：{analysis_audit['random_seed']}；bootstrap：{analysis_audit['bootstrap_replicates']}；原始实验审计总点数：{audit['total_point_rows']}。", styles["small"])]

    doc.build(story)
    print(f"created {OUTPUT}")


if __name__ == "__main__":
    build()

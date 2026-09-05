#!/usr/bin/env python3
"""Aggregate all NATIVE (full-codebook) attack, tamper and MRC results and
write a Markdown report.

Sources (all evaluated against the native / full codebook, 65536):
  * Attacks: attack_evasion_*.csv, rows with N_registry == 65536.
  * Tamper : tamper_<m>_K<K>.csv (matched) and tamper_arbitrary_<m>_K<K>.csv.
  * MRC    : margin_reachability_v2_native_<m>_K<K>.csv (softmin-v2), targets
             chosen inside N=1024 but ranked against the native codebook.

Usage: python scripts/summarize_native_results.py [--md OUTPATH]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
from datetime import date

EVAL = "results/evaluation"
MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
Ks = [2, 3, 5, 8]


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def pct(num, den):
    return (100.0 * num / den) if den else float("nan")


def best_csv(base):
    """Prefer full .csv over .partial; skip .smoke."""
    full = os.path.join(EVAL, base + ".csv")
    part = os.path.join(EVAL, base + ".partial.csv")
    if os.path.exists(full):
        return full
    if os.path.exists(part):
        return part
    return None


def avg(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


# ------------------------------------------------------------------ attacks
def load_attack_rows(path, want_registry="65536"):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if r.get("N_registry") == want_registry]


def summarize_attack(rows):
    n = len(rows)
    if not n:
        return None
    g = lambda col, val="1": sum(1 for r in rows if r.get(col) == val)
    col = lambda c: [fnum(r[c]) for r in rows if fnum(r.get(c)) is not None]
    return dict(
        n=n,
        escape=pct(g("identity_escape"), n),
        src_top1=pct(g("source_top1"), n),
        forge=pct(g("attacker_target_hit"), n),
        pesq=avg(col("PESQ")), stoi=avg(col("STOI")),
        sisdr=avg(col("SI_SDR")), snr=avg(col("SNR")),
    )


def collect_attacks():
    """Returns list of (attack, cell_label, summary|None|'missing')."""
    out = []
    simple = [
        ("overwrite_same", "attack_evasion_overwrite_same_{m}"),
        ("encodec", "attack_evasion_encodec_{m}"),
        ("vocoder", "attack_evasion_vocoder_{m}"),
        ("fgsm", "attack_evasion_fgsm_{m}"),
    ]
    for label, tmpl in simple:
        for m in MODELS:
            path = best_csv(tmpl.format(m=m))
            if not path:
                out.append((label, m, "missing"))
                continue
            s = summarize_attack(load_attack_rows(path))
            out.append((label, m, s))  # s may be None -> no native rows
    # overwrite_cross: explicit src__to_dst names
    cross = sorted(
        os.path.basename(p)[len("attack_evasion_overwrite_cross_"):-4]
        for p in glob.glob(os.path.join(EVAL, "attack_evasion_overwrite_cross_*.csv"))
        if ".partial" not in p and ".smoke" not in p
    )
    for pair in cross:
        path = best_csv(f"attack_evasion_overwrite_cross_{pair}")
        s = summarize_attack(load_attack_rows(path)) if path else "missing"
        out.append(("overwrite_cross", pair.replace("__to_", "→"), s))
    return out


# ------------------------------------------------------------------- tamper
def summarize_tamper(rows, method):
    sub = [r for r in rows if r.get("method") == method]
    n = len(sub)
    if not n:
        return None
    return n, pct(sum(1 for r in sub if r.get("target_top1") == "1"), n)


def collect_tamper(prefix):
    out = []
    for m in MODELS:
        for K in Ks:
            path = best_csv(f"{prefix}_{m}_K{K}")
            if not path:
                out.append((m, K, None))
                continue
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            out.append((m, K, dict(
                mean=summarize_tamper(rows, "mean"),
                tct=summarize_tamper(rows, "tct"),
            )))
    return out


# ---------------------------------------------------------------------- MRC
def summarize_mrc(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    n = len(rows)
    trials = defaultdict(list)
    for r in rows:
        trials[r["gi"]].append(r)
    col = lambda c: [fnum(r[c]) for r in rows if fnum(r.get(c)) is not None]
    return dict(
        n=n, n_tr=len(trials),
        cand=pct(sum(1 for r in rows if r.get("target_top1") == "1"), n),
        any10=pct(sum(1 for rs in trials.values()
                      if any(x.get("target_top1") == "1" for x in rs)), len(trials)),
        margin=avg(col("target_margin")),
        solver=pct(sum(1 for r in rows if r.get("solver_success") in ("True", "1")), n),
        keff=avg(col("effective_K")),
        pesq=avg(col("PESQ")), stoi=avg(col("STOI")),
    )


def collect_mrc():
    out = []
    for m in ["audioseal", "wavmark", "voicemark", "wmcodec"]:
        for K in Ks:
            path = best_csv(f"margin_reachability_v2_native_{m}_K{K}")
            out.append((m, K, summarize_mrc(path) if path else None))
    return out


# ------------------------------------------------------------------- render
def f(x, nd=1):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def build_markdown(attacks, tamper_matched, tamper_arb, mrc):
    L = []
    L.append("# Native 结果汇总：攻击 / 篡改 / MRC")
    L.append("")
    L.append(f"> 生成日期：{date.today().isoformat()}　·　"
             "全部在 native 全库（65536）下评估　·　"
             "由 `scripts/summarize_native_results.py` 自动生成")
    L.append("")
    L.append("每格约 300 trials。native 指身份归属在完整码本（65536 条身份）下排名，"
             "而非受限的 N=1024 子库。")
    L.append("")
    L.append("**已知数据缺口**")
    L.append("")
    L.append("- **timbrewm 的 native（65536）单拷贝攻击与 MRC 都未运行**，只有 N=1024 档；"
             "篡改（tamper）的 timbrewm native 存在。")
    L.append("- **fgsm 对 wavmark 不支持**（NotImplementedError），故缺该格。")
    L.append("")

    # ---- attacks
    L.append("## 一、Native 攻击结果")
    L.append("")
    L.append("`escape%` 原身份逃逸成功率；`srcTop1%` 攻击后源身份仍排第一；"
             "`forge%` 伪造目标命中率；后四列为音质（PESQ / STOI / SI-SDR dB / SNR dB）。")
    L.append("")
    L.append("| 攻击 | 模型 | n | escape% | srcTop1% | forge% | PESQ | STOI | SI-SDR | SNR |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for label, cell, s in attacks:
        if s == "missing":
            L.append(f"| {label} | {cell} | — | | | | | | | *(文件缺失)* |")
        elif s is None:
            L.append(f"| {label} | {cell} | — | | | | | | | *(无 native 行)* |")
        else:
            L.append(f"| {label} | {cell} | {s['n']} | {f(s['escape'])} | {f(s['src_top1'])} "
                     f"| {f(s['forge'])} | {f(s['pesq'],2)} | {f(s['stoi'],3)} "
                     f"| {f(s['sisdr'],2)} | {f(s['snr'],2)} |")
    L.append("")
    L.append("要点：overwrite / vocoder 对多数系统逃逸率≈100%；encodec 对 audioseal/voicemark "
             "逃逸弱（原身份多半保住）；`audioseal→wavmark` 这条 cross 逃逸为 0、源身份 100% 保住，属异常格。")
    L.append("")

    # ---- tamper
    L.append("## 二、Native 篡改结果")
    L.append("")
    L.append("`target_top1%` = 目标身份被篡改到 native 排名第一的比例。每格 n=3000"
             "（300 trials × 10 候选）。`mean` = 均匀权重基线，`tct` = 定向凸篡改（Targeted Convex Tampering）。")
    L.append("")
    for title, data in [("### matched（目标靠近联盟凸包）", tamper_matched),
                        ("### arbitrary（任意目标）", tamper_arb)]:
        L.append(title)
        L.append("")
        L.append("| 模型 | K | mean_top1% | tct_top1% |")
        L.append("|---|--:|--:|--:|")
        for m, K, s in data:
            if s is None:
                L.append(f"| {m} | {K} | | *(缺失)* |")
                continue
            mp = f(s["mean"][1]) if s["mean"] else "n/a"
            tp = f(s["tct"][1]) if s["tct"] else "n/a"
            L.append(f"| {m} | {K} | {mp} | {tp} |")
        L.append("")
    L.append("要点：matched 下 tct 随 K 上升才有效（timbrewm K=8 达 62.9%，audioseal/wavmark ≈11%），"
             "voicemark/wmcodec 基本不可篡改；arbitrary 目标几乎全部失败（仅 timbrewm 高 K 有个位数）。")
    L.append("")

    # ---- mrc
    L.append("## 三、Native MRC 结果（margin_reachability_v2 / softmin-v2）")
    L.append("")
    L.append("目标在 N=1024 活跃库内挑选，最终排名在全库 65536（native）下验证。"
             "`cand_top1%` 单候选命中率（n=3000/格）；`any@10%` 每 trial 的 10 个候选里至少一个升到第一（n=300/格）；"
             "`avg_margin` 平均判决余量（越接近 0 越易达成）。")
    L.append("")
    L.append("| 模型 | K | cand_top1% | any@10% | avg_margin | solver% | eff_K | PESQ | STOI |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m, K, s in mrc:
        if s is None:
            L.append(f"| {m} | {K} | | | | | | | *(缺 native)* |")
            continue
        L.append(f"| {m} | {K} | {f(s['cand'])} | {f(s['any10'])} | {f(s['margin'],4)} "
                 f"| {f(s['solver'])} | {f(s['keff'],2)} | {f(s['pesq'],2)} | {f(s['stoi'],3)} |")
    L.append("")
    L.append("- **timbrewm 的 native（65536）未运行**，只有 N=1024 档。")
    L.append("")
    L.append("要点：MRC 在 native 下的强弱由 `avg_margin` 决定——audioseal/wavmark 余量仅 −1.5~−3.8"
             "（离决策面近），随 K 上升 any@10 冲到 60~71%；voicemark/wmcodec 余量高达 −30~−39"
             "（隔得极远），基本不可达（any@10 ≤3%）。音质几乎无损（PESQ 4.0~4.6，STOI ≥0.97），"
             "因为 softmin-v2 只在 payload 空间做凸组合、不加噪。此结论与篡改表中 voicemark/wmcodec 抗篡改一致。")
    L.append("")
    return "\n".join(L)


def build_text(attacks, tamper_matched, tamper_arb, mrc):
    """Compact fixed-width echo for the terminal."""
    out = ["NATIVE ATTACKS (N_registry=65536)"]
    for label, cell, s in attacks:
        if isinstance(s, dict):
            out.append(f"  {label:16s} {cell:22s} esc={f(s['escape'])}% "
                       f"src={f(s['src_top1'])}% forge={f(s['forge'])}% PESQ={f(s['pesq'],2)}")
        else:
            out.append(f"  {label:16s} {cell:22s} ({s})")
    out.append("NATIVE TAMPER matched/arbitrary: see markdown")
    out.append("NATIVE MRC (softmin-v2)")
    for m, K, s in mrc:
        if s:
            out.append(f"  {m:10s} K{K}  cand={f(s['cand'])}%  any@10={f(s['any10'])}%  "
                       f"margin={f(s['margin'],3)}")
        else:
            out.append(f"  {m:10s} K{K}  (missing native)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="docs/native_results_summary.md",
                    help="Markdown output path")
    args = ap.parse_args()

    attacks = collect_attacks()
    tamper_matched = collect_tamper("tamper")
    tamper_arb = collect_tamper("tamper_arbitrary")
    mrc = collect_mrc()

    md = build_markdown(attacks, tamper_matched, tamper_arb, mrc)
    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    with open(args.md, "w") as fh:
        fh.write(md + "\n")

    print(build_text(attacks, tamper_matched, tamper_arb, mrc))
    print(f"\nMarkdown report written to: {args.md}")


if __name__ == "__main__":
    main()

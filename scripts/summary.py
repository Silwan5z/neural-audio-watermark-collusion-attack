"""v19 数据汇总：生成论文主表（攻击 + 篡改）。

从 evaluation/*.csv 聚合：
- 攻击主表：ASR / R3 / R5 / ACC_near / PESQ，按 (model, K, method)
- 篡改主表：target_top1 单候选 + N候选≥1，按 (model, K, method)

用法：python scripts/summary.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
MODELS = ["audioseal", "timbrewm", "wavmark", "voicemark", "wmcodec"]
KS = [2, 3, 5, 8]


def agg_attack(model, k):
    f = RESULTS / f"attack_{model}_K{k}.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    out = {}
    for m in ["mean", "blind_gram_cb", "extreme_pair"]:
        v = [r for r in rows if r["method"] == m]
        if not v:
            continue
        asr = np.mean([int(r["ASR"]) for r in v])
        r3 = np.mean([int(r["R3_escape"]) for r in v])
        r5 = np.mean([int(r["R5_escape"]) for r in v])
        accs = [float(r["ACC_near_norm"]) for r in v if r["ACC_near_norm"] != ""]
        acc = np.mean(accs) if accs else float("nan")
        pesqs = [float(r["PESQ"]) for r in v]
        out[m] = {"ASR": asr, "R3": r3, "R5": r5, "ACC_near": acc,
                  "PESQ": np.mean(pesqs)}
    return out


def agg_baseline(model, k):
    f = RESULTS / f"baselines_{model}_K{k}.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    out = {}
    for m in ["median", "minimum", "maximum", "rand_minmax", "copy_paste"]:
        v = [r for r in rows if r["method"] == m]
        if not v:
            continue
        out[m] = {
            "ASR": np.mean([int(r["ASR"]) for r in v]),
            "ACC_near": np.mean([float(r["ACC_near_norm"]) for r in v if r["ACC_near_norm"] != ""]),
            "PESQ": np.mean([float(r["PESQ"]) for r in v]),
        }
    return out


def agg_tamper(model, k):
    f = RESULTS / f"tamper_{model}_K{k}.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    out = {}
    for m in ["mean", "framing_cb"]:
        v = [r for r in rows if r["method"] == m]
        if not v:
            continue
        out[m] = {"single": np.mean([int(r["target_top1"]) for r in v])}
    # N候选≥1（framing_cb 每 trial）
    by_trial = {}
    for r in rows:
        if r["method"] == "framing_cb":
            by_trial.setdefault((r["spk"], r["local_t"]), []).append(int(r["target_top1"]))
    out["framing_cb"]["anyN"] = np.mean([1 if max(v) == 1 else 0 for v in by_trial.values()])
    return out


def main():
    print("=" * 110)
    print("攻击主表（ASR / R3 / R5 / ACC_near / PESQ）")
    print("=" * 110)
    print(f"{'模型':<10} {'K':>2} | {'mean':>30} {'blind_gram_cb':>30} {'extreme_pair':>30}")
    for m in MODELS:
        for k in KS:
            a = agg_attack(m, k)
            if a is None:
                print(f"{m:<10} {k:>2} | 缺失")
                continue
            def fmt(d):
                if not d:
                    return "-"
                return f"A{float(d['ASR']):.2f} R5{float(d['R5']):.2f} Ac{float(d['ACC_near']):.2f} P{float(d['PESQ']):.1f}"
            print(f"{m:<10} {k:>2} | {fmt(a.get('mean')):>30} {fmt(a.get('blind_gram_cb')):>30} {fmt(a.get('extreme_pair')):>30}")

    print()
    print("=" * 110)
    print("经典 baseline（ASR / ACC_near）")
    print("=" * 110)
    print(f"{'模型':<10} {'K':>2} | {'median':>22} {'minimum':>22} {'maximum':>22} {'rand_minmax':>22} {'copy_paste':>22}")
    for m in MODELS:
        for k in KS:
            b = agg_baseline(m, k)
            if b is None:
                print(f"{m:<10} {k:>2} | 缺失")
                continue
            def fmt(d):
                return f"A{float(d['ASR']):.2f} Ac{float(d['ACC_near']):.2f}"
            print(f"{m:<10} {k:>2} | {fmt(b.get('median')):>22} {fmt(b.get('minimum')):>22} {fmt(b.get('maximum')):>22} {fmt(b.get('rand_minmax')):>22} {fmt(b.get('copy_paste')):>22}")

    print()
    print("=" * 110)
    print("篡改主表（target_top1 单候选 / N候选≥1）")
    print("=" * 110)
    print(f"{'模型':<10} {'K':>2} | {'mean 单':>10} {'framing 单':>12} {'framing N≥1':>14}")
    for m in MODELS:
        for k in [2, 3, 5, 8]:
            t = agg_tamper(m, k)
            if t is None:
                print(f"{m:<10} {k:>2} | 缺失")
                continue
            mean_s = t.get("mean", {}).get("single", float("nan"))
            fc_s = t.get("framing_cb", {}).get("single", float("nan"))
            fc_n = t.get("framing_cb", {}).get("anyN", float("nan"))
            print(f"{m:<10} {k:>2} | {mean_s*100:>9.1f}% {fc_s*100:>11.1f}% {fc_n*100:>13.1f}%")


if __name__ == "__main__":
    main()

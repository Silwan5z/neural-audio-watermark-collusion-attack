"""统计检验：按说话人分层的 McNemar + paired bootstrap。

关键（STATUS 步骤3）：先重采样说话人集合，再在被抽中的说话人内部重采样 trial。
这样把"说话人"当作第一层随机效应，避免把同一说话人的多个 trial 当独立样本。

用法：python scripts/stats.py --model voicemark --K 2
输出：mean vs blind_gram_cb / extreme_pair / 经典baseline 的 McNemar p 值 + bootstrap CI。
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
N_BOOT = 2000
SEED = 0


def load_csv(model, k, prefix):
    f = DATA / prefix / f"{prefix}_{model}_K{k}.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    # 按 (spk, local_t) 对齐，每行 = 一个 trial × 方法
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], {})[(r["spk"], r["local_t"])] = r
    return by_method


def mcnemar(a, b):
    """a, b: 两个方法的二值结果序列（同 trial 对齐）。"""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    # exact binomial (mid-p)，小样本精确
    from scipy.stats import binomtest
    p = binomtest(n01, n=n).pvalue
    return p, n01, n10


def speaker_stratified_bootstrap(speakers, get_pair, n_boot=N_BOOT):
    """按说话人分层 bootstrap：重采样说话人 → 层内重采样 trial。
    get_pair(spk, trial) -> (x, y) 两个方法的指标。返回 (diff 均值, 95% CI)。"""
    rng = np.random.default_rng(SEED)
    spk_list = list(speakers.keys())
    diffs = []
    for _ in range(n_boot):
        # 重采样说话人（有放回）
        spk_boot = rng.choice(spk_list, size=len(spk_list), replace=True)
        xs, ys = [], []
        for spk in spk_boot:
            trials = speakers[spk]
            # 层内重采样 trial（有放回，数量 = 原 trial 数）
            t_boot = rng.choice(trials, size=len(trials), replace=True)
            for t in t_boot:
                x, y = get_pair(spk, t)
                xs.append(x)
                ys.append(y)
        diffs.append(np.mean(ys) - np.mean(xs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--agg", choices=["ASR", "ACC_near_norm"], default="ASR")
    args = ap.parse_args()

    atk = load_csv(args.model, args.K, "attack")
    if atk is None:
        print(f"attack_{args.model}_K{args.K}.csv 缺失")
        return
    # 合并所有数据源：attack(3方法) + blind_dist_cb + blind_minimax_cb + baselines(5方法)
    sources = dict(atk)
    for prefix in ["blind_dist_cb", "blind_minimax_cb", "baselines"]:
        m = load_csv(args.model, args.K, prefix)
        if m:
            sources.update(m)

    def metric(r):
        v = r.get(args.agg, "")
        return float(v) if v != "" else None

    # 组织：method -> {(spk, trial): metric}
    def method_metric(by_method, method):
        return {k: metric(v) for k, v in by_method.get(method, {}).items() if metric(v) is not None}

    # 说话人分层结构
    all_keys = set()
    for m in atk.values():
        all_keys |= set(m.keys())
    speakers = {}
    for spk, t in all_keys:
        speakers.setdefault(spk, []).append(t)

    # 对比：mean vs 各方法
    print(f"=== {args.model} K={args.K} ({args.agg}) 按说话人分层统计 ===")
    print(f"{'对比':<28} {'diff(b-a)':>10} {'95%CI':>18} {'McNemar_p':>10}")
    mean_m = method_metric(sources, "mean")
    methods = ["blind_gram_cb", "extreme_pair", "blind_dist_cb", "blind_minimax_cb",
               "median", "minimum", "maximum", "rand_minmax", "copy_paste"]
    for method in methods:
        other_m = method_metric(sources, method)
        if not other_m:
            continue
        # 对齐到共同 trial
        common = set(mean_m) & set(other_m)
        if not common:
            continue
        # 说话人分层 bootstrap
        spk_map = {}
        for spk, t in common:
            spk_map.setdefault(spk, []).append(t)
        def get_pair(spk, t, mm=mean_m, om=other_m):
            return mm[(spk, t)], om[(spk, t)]
        diff, lo, hi = speaker_stratified_bootstrap(spk_map, get_pair)
        # McNemar（ASR 时才有意义）
        p = ""
        if args.agg == "ASR":
            a_vals = [mean_m[k] for k in common]
            b_vals = [other_m[k] for k in common]
            p, _, _ = mcnemar([int(v) for v in a_vals], [int(v) for v in b_vals])
        print(f"{'mean vs '+method:<28} {diff:>+10.4f} [{lo:>7.4f},{hi:>7.4f}] {p if p!='' else '':>10}")


if __name__ == "__main__":
    main()

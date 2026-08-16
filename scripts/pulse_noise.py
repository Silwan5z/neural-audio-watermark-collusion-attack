"""v19 Kiyavash & Moulin 脉冲噪声对照（多说话人 + 全空间注册表）。

理论：Kiyavash & Moulin 证明"线性平均 + 脉冲噪声"是攻击最优（加性模型）。
验证：mean 凸混合基础上加两点分布脉冲噪声，攻击 ASR / 篡改命中率是否变化。

用法：python scripts/pulse_noise.py --model timbrewm --K 5 --n_trials 50
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
    int_to_bits, full_registry_size,
)
from watermarks import detect  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
EPS = [0.0, 0.001, 0.01, 0.05]
R0S = [0.05, 0.1, 0.3]


def add_impulsive(y, eps, r0, rng):
    n = len(y)
    k = max(1, int(n * eps))
    idx = rng.choice(n, size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k)
    y2 = y.copy()
    y2[idx] += signs * r0
    return y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=50)
    args = ap.parse_args()

    model = args.model
    K = args.K
    d = NBITS[model]
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)
    out_csv = RESULTS / f"pulse_noise_{model}_K{K}.csv"

    rows = []
    t_start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]
        coll_set = set(coll_ints)

        # mean 混合（基线）
        a_mean = np.ones(K) / K
        y_mean = sum(a_mean[i] * wavs[i] for i in range(K)).astype(np.float32)

        for eps in EPS:
            for r0 in R0S:
                y = y_mean if eps == 0.0 else add_impulsive(y_mean, eps, r0, rng)
                scores, _, _ = detect(model, y, registry_bits)
                top1_idx = int(np.argsort(scores)[::-1][0])
                top1_int = int((registry_bits[top1_idx] @ (2 ** np.arange(d))).sum())
                asr = int(top1_int not in coll_set)
                rows.append({
                    "model": model, "K": K, "spk": spk, "local_t": local_t,
                    "eps": eps, "r0": r0, "ASR": asr,
                })
        if (gi + 1) % 20 == 0:
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)} ({time.time()-t_start:.0f}s)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== {model} K={K} 脉冲噪声攻击 ASR（n={len(trial_idx)}）===")
    for eps in EPS:
        for r0 in R0S:
            v = [r["ASR"] for r in rows if r["eps"] == eps and r["r0"] == r0]
            tag = " <== 基线" if eps == 0.0 else ""
            print(f"  eps={eps:>5} r0={r0:>4}: ASR={np.mean(v)*100:.1f}%{tag}")


if __name__ == "__main__":
    main()

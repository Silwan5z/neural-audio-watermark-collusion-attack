"""v19 盲方法补充：blind_dist_cb（用两两波形距离矩阵代替 median 残差 Gram）。

不重跑/不覆盖已有的 attack_*.csv（mean/blind_gram_cb/extreme_pair），只单独产出
blind_dist_cb_{model}_K{K}.csv，供后续与已有数据对齐比较（同一套 coalition_seed）。

推导：假设各成员码字能量近似相等（神经水印嵌入强度基本恒定），有
  ||x_i - x_j||^2 = ||Q_i - Q_j||^2 = const - 2<Q_i, Q_j>
两两波形距离矩阵 D 是码字 Gram 矩阵 G 的精确线性变换（非估计，不依赖 median 猜测
公共内容 S），因此 min_a a^T G a 等价于 max_a a^T D a（常数项在 sum(a)=1 约束下
不影响 argmin/argmax）。

max_a a^T D a 是凸函数最大化（凹最小化），最优解在可行域顶点，用多起点（含
extreme_pair 解作 warm start）避免 SLSQP 卡在差的局部解/鞍点。

验证记录（blind_dist_cb_probe.py，探针实验，未正式入库）：
- timbrewm K=5 n=60: blind_dist_cb ASR=0.967/R5esc=0.867/ACC=0.800，
  extreme_pair ASR=0.967/R5esc=0.850/ACC=0.768 —— 基本打平，ACC 略逊
- timbrewm K=8 n=60: blind_dist_cb ASR=1.000/R5esc=0.967/ACC=0.790，
  extreme_pair ASR=0.950/R5esc=0.800/ACC=0.800 —— blind_dist_cb 全面更优
- voicemark K=8 n=60: 四方法接近饱和，blind_dist_cb 并列最优
结论：K 越大 blind_dist_cb 相对 extreme_pair 的优势越明显（K=2时数学上完全退化为
同一个解），K=5/8 上不差于 extreme_pair，K=8 上明显更优。

用法：python scripts/blind_distance.py --model timbrewm --K 5 --n_trials 150
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, CAP, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
)
from watermarks import detect, pesq_wb, stoi, si_sdr  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"


def extreme_pair(wavs, K):
    best_e = -1
    bp = None
    for i in range(K):
        for j in range(i + 1, K):
            e = np.mean((wavs[i] - wavs[j]) ** 2)
            if e > best_e:
                best_e, bp = e, (i, j)
    a = np.zeros(K)
    a[bp[0]] = 0.5
    a[bp[1]] = 0.5
    return a


def dist_matrix(wavs):
    """两两波形距离矩阵 D_ij = ||x_i - x_j||^2（精确，非估计）。"""
    K = len(wavs)
    n = min(len(w) for w in wavs)
    X = np.stack([w[:n] for w in wavs])
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            D[i, j] = np.mean((X[i] - X[j]) ** 2)
    return D


def blind_dist_cb_weights(wavs, cap=0.5, n_random_starts=5):
    """max_a a^T D a，多起点避免局部解（含 extreme_pair warm start）。"""
    D = dist_matrix(wavs)
    K = D.shape[0]

    def obj(a):
        a = np.asarray(a, float)
        return -float(a @ D @ a)

    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    bounds = [(0, cap)] * K

    starts = [np.full(K, 1.0 / K), extreme_pair(wavs, K)]
    rng = np.random.default_rng(0)
    for _ in range(n_random_starts):
        idx = rng.choice(K, size=min(K, int(np.ceil(1 / cap))), replace=False)
        a0 = np.zeros(K)
        a0[idx] = 1.0 / len(idx)
        starts.append(a0)

    best_a, best_val = None, np.inf
    for a0 in starts:
        res = minimize(obj, a0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-14})
        a = np.clip(res.x, 0, cap)
        s = a.sum()
        a = a / s if s > 1e-8 else np.ones(K) / K
        val = obj(a)
        if val < best_val:
            best_val, best_a = val, a
    return best_a


def _int_to_bits_row(v, d):
    return np.array([(v >> i) & 1 for i in range(d)], dtype=np.int8)


def _rows_to_ints(row_idx, registry_bits):
    d = registry_bits.shape[1]
    weights = 2 ** np.arange(d)
    return (registry_bits[row_idx] @ weights).tolist()


def metrics_of(model, y, coll_ints, registry_bits, d):
    scores, _, hard = detect(model, y.astype(np.float32), registry_bits)
    rank = np.argsort(scores)[::-1]
    coll_set = set(coll_ints)

    top1_ints = _rows_to_ints(rank[:1], registry_bits)
    top3_ints = _rows_to_ints(rank[:3], registry_bits)
    top5_ints = _rows_to_ints(rank[:5], registry_bits)

    asr = int(len(set(top1_ints) & coll_set) == 0)
    r3_escape = int(len(set(top3_ints) & coll_set) == 0)
    r5_escape = int(len(set(top5_ints) & coll_set) == 0)

    if hard is None:
        acc_near = None
    else:
        coll_bits = np.array([_int_to_bits_row(ci, d) for ci in coll_ints])
        same = (coll_bits == hard[None, :]).sum(axis=1)
        acc_near = int(same.max())

    return asr, r3_escape, r5_escape, acc_near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=150)
    args = ap.parse_args()

    model, K = args.model, args.K
    d = NBITS[model]

    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    out_csv = RESULTS / f"blind_dist_cb_{model}_K{K}.csv"
    rows = []
    t_start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]
        wm_ref = wavs[0]  # 与 attack.py 口径一致

        a = blind_dist_cb_weights(wavs, CAP)
        y = sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
        asr, r3e, r5e, acc = metrics_of(model, y, coll_ints, registry_bits, d)
        pesq = pesq_wb(wm_ref, y)
        st = stoi(wm_ref, y)
        sdr = si_sdr(wm_ref, y)
        rows.append({
            "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
            "method": "blind_dist_cb",
            "ASR": asr, "R3_escape": r3e, "R5_escape": r5e,
            "ACC_near": "" if acc is None else acc,
            "ACC_near_norm": "" if acc is None else f"{acc/d:.4f}",
            "PESQ": f"{pesq:.4f}", "STOI": f"{st:.4f}", "SI_SDR": f"{sdr:.2f}",
        })
        if (gi + 1) % 30 == 0:
            elapsed = time.time() - t_start
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)}  ({elapsed:.0f}s)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    asrs = [r["ASR"] for r in rows]
    r5s = [r["R5_escape"] for r in rows]
    accs = [float(r["ACC_near_norm"]) for r in rows if r["ACC_near_norm"] != ""]
    print(f"\n=== {model} K={K} blind_dist_cb (n={len(trial_idx)}) ===")
    print(f"  ASR={np.mean(asrs):.3f}  R5_escape={np.mean(r5s):.3f}"
          f"  ACC_near_norm={np.mean(accs) if accs else float('nan'):.3f}")
    print(f"写入 {out_csv}")


if __name__ == "__main__":
    main()

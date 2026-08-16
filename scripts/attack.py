"""攻击主脚本：全空间注册表 + 多说话人（38人）+ 150 trial 均匀分配。

方法：mean / blind_gram_cb / extreme_pair（核心三方法，全部盲）
指标：
  - ASR：top-1 逃逸率（P[top1 not in coalition]）
  - R@3 / R@5：top-K 候选名单逃逸率（P[topK ∩ coalition = empty]），取证兜底能力
  - ACC_near：与最近 coalition 成员的逐位重合率（0..d），越低攻击越强，主指标
  - AggResid：||C_coalition @ a||^2 / K（聚合残留证据，归一化到 [0,1] 附近），机制解释辅助指标

用法：python scripts/attack.py --model audioseal --K 5
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
    NBITS, CAP, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
)
from watermarks import detect, pesq_wb, stoi, si_sdr  # noqa: E402
from convex import blind_cb_weights  # noqa: E402

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


def metrics_of(model, y, coll_ints, registry_bits, a, d):
    """返回 (asr, r3_escape, r5_escape, acc_near, agg_resid)。"""
    scores, _, hard = detect(model, y.astype(np.float32), registry_bits)
    rank = np.argsort(scores)[::-1]
    coll_set = set(coll_ints)

    # rank 是 registry 行索引，需转成 codeword int 再与 coalition 比较
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

    # 聚合残留证据：||C_coalition @ a||^2 / K，用 pm 编码 (-1,+1)
    coll_bits_pm = np.array([_int_to_bits_row(ci, d) for ci in coll_ints]).astype(float) * 2 - 1
    resid = coll_bits_pm.T @ a  # [d]
    agg_resid = float(np.sum(resid ** 2) / d)

    return asr, r3_escape, r5_escape, acc_near, agg_resid


def _int_to_bits_row(v, d):
    return np.array([(v >> i) & 1 for i in range(d)], dtype=np.int8)


def _rows_to_ints(row_idx, registry_bits):
    d = registry_bits.shape[1]
    weights = 2 ** np.arange(d)
    return (registry_bits[row_idx] @ weights).tolist()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=150)
    args = ap.parse_args()

    model = args.model
    K = args.K
    d = NBITS[model]

    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    out_csv = RESULTS / f"attack_{model}_K{K}.csv"
    rows = []
    t_start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]

        a_mean = np.ones(K) / K
        a_gram = blind_cb_weights(wavs, CAP)  # 盲 CB：只用波形残差，不用 payload
        a_pair = extreme_pair(wavs, K)

        # 固定取 coalition 里的第一个成员做音质参照（不随机，同一 trial 内 mean/cb/extreme_pair
        # 三个方法共用同一个参照成员，保证组内可比）
        wm_ref = wavs[0]

        for mname, a in [("mean", a_mean), ("blind_gram_cb", a_gram), ("extreme_pair", a_pair)]:
            y = sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
            asr, r3e, r5e, acc, agg = metrics_of(model, y, coll_ints, registry_bits, a, d)
            # PESQ/STOI/SI-SDR 参照随机挑中的那个合谋者的水印音频（不是 clean）
            pesq = pesq_wb(wm_ref, y)
            st = stoi(wm_ref, y)
            sdr = si_sdr(wm_ref, y)
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi, "method": mname,
                "ASR": asr, "R3_escape": r3e, "R5_escape": r5e,
                "ACC_near": "" if acc is None else acc,
                "ACC_near_norm": "" if acc is None else f"{acc/d:.4f}",
                "AggResid": f"{agg:.6f}",
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

    print(f"\n=== {model} K={K} 汇总（全空间注册表, 38说话人, n={len(trial_idx)}）===")
    for m in ["mean", "blind_gram_cb", "extreme_pair"]:
        asrs = [r["ASR"] for r in rows if r["method"] == m]
        r5s = [r["R5_escape"] for r in rows if r["method"] == m]
        accs = [r["ACC_near_norm"] for r in rows if r["method"] == m and r["ACC_near_norm"] != ""]
        accs = [float(x) for x in accs]
        print(f"  {m:12s}: ASR={np.mean(asrs):.3f}  R5_escape={np.mean(r5s):.3f}"
              f"  ACC_near_norm={np.mean(accs) if accs else float('nan'):.3f}")


if __name__ == "__main__":
    main()

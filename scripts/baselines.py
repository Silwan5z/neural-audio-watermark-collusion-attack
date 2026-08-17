"""经典合谋 baseline：median / copy_paste / rand_minmax / minimum / maximum。

来源核实（见调研报告，只用 attack_v3/paper/paper_text/notes/ 那份文档列出的文献）：
- Zhao et al. 2005 (IEEE TIP) 定义了 average/minimum/maximum/median/minmax/modified-negative/
  randomized-negative 共7种攻击，本质都是逐样本点对K个副本取顺序统计量的变体。
- minimum/maximum 定义明确：逐样本点取K个副本的最小值/最大值。
- modified-negative/randomized-negative 原文数学公式未能核实（WebFetch读不到原文PDF，只能
  搜到摘要片段），按用户决定不实现，避免公式对不上被审稿人戳破。
- rand_minmax（旧称"minmax"）：逐样本点在 min 和 max 之间随机二选一，对应 Zhao 2005 的
  randomized-negative 概念性描述，但不是核实过的原文公式，改名 rand_minmax 以示区分，
  不再声称是文献里的"minmax"这个具体方法。
- copy_paste：不是任何文献里的标准方法，是本项目自行设计的对照（波形分段各自复制拼接，
  衡量"完整保留片段"这种激进操作是否比混合类方法更强/更弱）。

方法定义：
- median：逐样本点取K个副本中位数
- minimum：逐样本点取K个副本最小值
- maximum：逐样本点取K个副本最大值
- rand_minmax：逐样本点在min/max之间随机二选一
- copy_paste：波形分段（20ms/段），每段整段复制自随机选中的某个成员

与 attack.py 使用同一套 coalition_seed / speaker_trial_index / 全空间注册表 /
按需嵌入缓存，音质参照 coalition 第一个成员的水印音频（与 attack.py 口径一致）。

用法：python scripts/baselines.py --model audioseal --K 5 --n_trials 150
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
)
from watermarks import detect, detect_many, pesq_wb, stoi, si_sdr  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
BLOCK_LEN = 320  # 20ms @ 16kHz


def median_mix(wavs):
    return np.median(np.stack(wavs, axis=0), axis=0)


def minimum_mix(wavs):
    return np.min(np.stack(wavs, axis=0), axis=0)


def maximum_mix(wavs):
    return np.max(np.stack(wavs, axis=0), axis=0)


def rand_minmax_mix(wavs, rng):
    stacked = np.stack(wavs, axis=0)  # [K, N]
    mn = stacked.min(axis=0)
    mx = stacked.max(axis=0)
    choice = rng.integers(0, 2, size=mn.shape[0]).astype(bool)
    return np.where(choice, mx, mn)


def copy_paste_mix(wavs, rng, block_len=BLOCK_LEN):
    n = len(wavs[0])
    K = len(wavs)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, block_len):
        end = min(start + block_len, n)
        src = rng.integers(0, K)
        out[start:end] = wavs[src][start:end]
    return out


def _int_to_bits_row(v, d):
    return np.array([(v >> i) & 1 for i in range(d)], dtype=np.int8)


def _rows_to_ints(row_idx, registry_bits):
    d = registry_bits.shape[1]
    weights = 2 ** np.arange(d)
    return (registry_bits[row_idx] @ weights).tolist()


def metrics_of(model, y, coll_ints, registry_bits, d, decoded=None):
    scores, _, hard = decoded if decoded is not None else detect(model, y.astype(np.float32), registry_bits)
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
    ap.add_argument("--n_trials", type=int, default=300)
    args = ap.parse_args()

    model = args.model
    K = args.K
    d = NBITS[model]

    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    out_csv = RESULTS / f"baselines_{model}_K{K}.csv"
    rows = []
    t_start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]

        # 混合方法内部随机性用独立种子，与 coalition 抽取种子区分但仍可复现
        mix_rng = np.random.default_rng(coalition_seed(spk, K, local_t) + 10_000_000)
        wm_ref = wavs[0]  # 与 attack.py 口径一致：固定取第一个成员做音质参照

        methods = {
            "median": median_mix(wavs),
            "minimum": minimum_mix(wavs),
            "maximum": maximum_mix(wavs),
            "rand_minmax": rand_minmax_mix(wavs, mix_rng),
            "copy_paste": copy_paste_mix(wavs, mix_rng),
        }

        method_items = list(methods.items())
        decoded_outputs = detect_many(model, [y.astype(np.float32) for _, y in method_items], registry_bits)
        for (mname, y), decoded in zip(method_items, decoded_outputs):
            y = y.astype(np.float32)
            asr, r3e, r5e, acc = metrics_of(model, y, coll_ints, registry_bits, d, decoded)
            pesq = pesq_wb(wm_ref, y)
            st = stoi(wm_ref, y)
            sdr = si_sdr(wm_ref, y)
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi, "method": mname,
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

    print(f"\n=== {model} K={K} 经典baseline汇总（全空间注册表, 38说话人, n={len(trial_idx)}）===")
    for m in ["median", "minimum", "maximum", "rand_minmax", "copy_paste"]:
        asrs = [r["ASR"] for r in rows if r["method"] == m]
        accs = [float(r["ACC_near_norm"]) for r in rows if r["method"] == m and r["ACC_near_norm"] != ""]
        pesqs = [float(r["PESQ"]) for r in rows if r["method"] == m]
        print(f"  {m:12s}: ASR={np.mean(asrs):.3f}  ACC_near_norm={np.mean(accs) if accs else float('nan'):.3f}"
              f"  PESQ={np.mean(pesqs):.3f}")


if __name__ == "__main__":
    main()

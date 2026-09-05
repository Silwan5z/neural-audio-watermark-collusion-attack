"""RP（Random Pair）：均匀随机选一对成员做 0.5/0.5 混合（selection 效果的对照组）。

与 FWP 共享同一套 coalition_seed/speaker_trial_index/全空间注册表/按需嵌入缓存，
唯一区别是配对规则：RP 均匀随机选一对，FWP 选波形距离最远的一对。两者用同一个
per-trial RNG（coalition_seed 派生），互相独立于 attack.py 里的 mean/fwp 抽样，
保证与 attack.py 可比但不共享随机状态。

用法：python scripts/rp.py --model audioseal --K 5 --n_trials 300
输出：results/evaluation/rp_{model}_K{K}.csv
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
from watermarks import detect, pesq_wb, stoi, si_sdr  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"


def random_pair(rng, K):
    i, j = rng.choice(K, size=2, replace=False)
    a = np.zeros(K)
    a[i] = 0.5
    a[j] = 0.5
    return a


def metrics_of(model, y, coll_ints, registry_bits, a, d):
    """返回 (asr, r3_escape, r5_escape, acc_near, agg_resid)。"""
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

    coll_bits_pm = np.array([_int_to_bits_row(ci, d) for ci in coll_ints]).astype(float) * 2 - 1
    resid = coll_bits_pm.T @ a
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
    ap.add_argument("--n_trials", type=int, default=300)
    args = ap.parse_args()

    model = args.model
    K = args.K
    d = NBITS[model]

    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    out_csv = RESULTS / f"rp_{model}_K{K}.csv"
    rows = []
    t_start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]
        wm_ref = wavs[0]

        a = random_pair(rng, K)
        y = sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
        asr, r3e, r5e, acc, agg = metrics_of(model, y, coll_ints, registry_bits, a, d)
        pesq = pesq_wb(wm_ref, y)
        st = stoi(wm_ref, y)
        sdr = si_sdr(wm_ref, y)
        rows.append({
            "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi, "method": "rp",
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

    asrs = [r["ASR"] for r in rows]
    r5s = [r["R5_escape"] for r in rows]
    accs = [float(r["ACC_near_norm"]) for r in rows if r["ACC_near_norm"] != ""]
    print(f"\n=== {model} K={K} RP（n={len(trial_idx)}）===")
    print(f"  ASR={np.mean(asrs):.3f}  R5_escape={np.mean(r5s):.3f}"
          f"  ACC_near_norm={np.mean(accs) if accs else float('nan'):.3f}")


if __name__ == "__main__":
    main()

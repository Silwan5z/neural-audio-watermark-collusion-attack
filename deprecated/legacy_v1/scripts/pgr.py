"""PGR（Payload Geometry Reference）：payload-aware minimax 参照方法（min_α max_i [G_c α]_i）。

与 blind 攻击家族（Mean/FWP/DM/BDB）不同：PGR 直接使用 coalition 的真实码字 Gram
G_c（±1 编码），回答"如果攻击者拿到 payload，能比盲方法多赚多少"这一诊断性问题。
G_c 从未被本仓库任何盲方法读取，只在此脚本和 evidence-chain 诊断中作为特权信息使用。

评估口径与 attack.py 一致：ASR / R3_escape / R5_escape / ACC_near / AggResid / PESQ/STOI/SI-SDR，
因为 minimax 目标是"最小化最强成员证据"（evasion 性质），不栽赃到 target。

用法：python scripts/pgr.py --model timbrewm --K 5 --n_trials 300
输出：results/evaluation/pgr_{model}_K{K}.csv
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
    speaker_trial_index, coalition_seed, sample_coalition, int_to_bits,
)
from watermarks import detect, pesq_wb, stoi, si_sdr  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"


def pgr_weights(G, cap=0.5):
    """min_α max_i [G_c α]_i（LSE 光滑近似）。G_c 为 ±1 编码的真实码字 Gram。"""
    K = G.shape[0]
    Gs = (G + G.T) / 2

    def obj(a):
        a = np.asarray(a, float)
        evidence = Gs @ a
        m = evidence.max()
        return float(m + np.log(np.sum(np.exp(evidence - m))))

    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    res = minimize(obj, np.full(K, 1.0 / K), method="SLSQP",
                   bounds=[(0, cap)] * K, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-14})
    a = np.clip(res.x, 0, cap)
    s = a.sum()
    return a / s if s > 1e-8 else np.ones(K) / K


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

    model, K = args.model, args.K
    d = NBITS[model]
    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    out_csv = RESULTS / f"pgr_{model}_K{K}.csv"
    rows = []
    t_start = time.time()
    for gi, (spk, local_t) in enumerate(trial_idx):
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]

        # payload-aware：知道 coalition 码字，算 ±1 Gram
        C = np.array([int_to_bits(ci, d) for ci in coll_ints])  # [K,d] {0,1}
        Cpm = C.astype(float) * 2 - 1  # ±1
        G = Cpm @ Cpm.T  # [K,K] Gram

        a = pgr_weights(G, CAP)
        y = sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
        wm_ref = wavs[0]

        scores, _, hard = detect(model, y, registry_bits)
        rank = np.argsort(scores)[::-1]
        coll_set = set(coll_ints)
        top1 = _rows_to_ints(rank[:1], registry_bits)
        top3 = _rows_to_ints(rank[:3], registry_bits)
        top5 = _rows_to_ints(rank[:5], registry_bits)
        asr = int(len(set(top1) & coll_set) == 0)
        r3 = int(len(set(top3) & coll_set) == 0)
        r5 = int(len(set(top5) & coll_set) == 0)
        if hard is None:
            acc = None
        else:
            coll_bits = np.array([_int_to_bits_row(ci, d) for ci in coll_ints])
            acc = int((coll_bits == hard[None, :]).sum(axis=1).max())
        resid = Cpm.T @ a
        agg = float(np.sum(resid ** 2) / d)

        rows.append({
            "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
            "method": "pgr",
            "ASR": asr, "R3_escape": r3, "R5_escape": r5,
            "ACC_near": "" if acc is None else acc,
            "ACC_near_norm": "" if acc is None else f"{acc/d:.4f}",
            "AggResid": f"{agg:.6f}",
            "PESQ": f"{pesq_wb(wm_ref, y):.4f}",
            "STOI": f"{stoi(wm_ref, y):.4f}",
            "SI_SDR": f"{si_sdr(wm_ref, y):.2f}",
        })
        if (gi + 1) % 50 == 0:
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)} ({time.time()-t_start:.0f}s)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    asrs = [r["ASR"] for r in rows]
    accs = [float(r["ACC_near_norm"]) for r in rows if r["ACC_near_norm"] != ""]
    print(f"\n=== {model} K={K} PGR（payload-aware, n={len(trial_idx)}）===")
    print(f"  ASR={np.mean(asrs):.3f}  ACC_near_norm={np.mean(accs) if accs else float('nan'):.3f}")


if __name__ == "__main__":
    main()

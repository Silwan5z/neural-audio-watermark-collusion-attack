"""TCT（Targeted Convex Tampering）主脚本：全空间注册表 + 多说话人 + opportunistic 目标。

口径（用户确认）：
- 篡改 = payload-aware（知道 target payload）
- 目标选择 = opportunistic：从注册表里选几何最近、最容易篡改的 target
- 方法：mean（baseline） vs tct（`argmin‖Cᵀa−c_t‖²`）

指标：target_top1（top-1 是否命中指定 target），单候选 + N 候选 ≥1 两种口径。

用法：python scripts/framing.py --model timbrewm --K 5 --n_trials 300
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, CAP, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, sample_coalition,
    int_to_bits, full_registry_size,
)
from watermarks import detect, detect_wavmark_many, get_wavmark  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"
N_CAND = 10  # opportunistic 候选数


def write_rows_atomic(path, rows):
    """Publish a complete CSV atomically so a restart never reads a partial row."""
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_completed_trials(partial_csv):
    """Load only full 10-target / two-method trial records from a checkpoint."""
    if not partial_csv.exists():
        return [], set()
    with partial_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_trial = {}
    for row in rows:
        by_trial.setdefault(int(row["gi"]), []).append(row)
    expected = 2 * N_CAND
    completed = {gi for gi, trial_rows in by_trial.items() if len(trial_rows) == expected}
    rows = [row for row in rows if int(row["gi"]) in completed]
    return rows, completed


def tct(C, c_t, cap=0.5):
    """payload-aware 篡改：a* = argmin ||Cᵀa − c_t||²。"""
    K = C.shape[0]
    def obj(a):
        return float(np.sum((C.T @ a - c_t) ** 2))
    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    res = minimize(obj, np.full(K, 1 / K), method="SLSQP", bounds=[(0, cap)] * K,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-14})
    a = np.clip(res.x, 0, cap)
    s = a.sum()
    return a / s if s > 1e-8 else np.ones(K) / K


def convex_dist(C, c_t):
    """码字 c_t 到 coalition 凸包的距离（越小越容易篡改）。"""
    K = C.shape[0]
    def obj(a):
        return float(np.sum((C.T @ a - c_t) ** 2))
    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    res = minimize(obj, np.full(K, 1 / K), method="SLSQP", bounds=[(0, 1)] * K,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
    return float(np.sqrt(obj(res.x)))


def convex_dist_batch_exact(C, targets):
    """Exact distances from many targets to conv(C), for K <= 8.

    The nearest point lies in the relative interior of some face.  Enumerating
    all nonempty faces and solving their equality-constrained projections in
    batch is equivalent to the SLSQP formulation above, but avoids one Python
    optimizer call per candidate target.
    """
    C = np.asarray(C, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    K, _ = C.shape
    n = len(targets)
    best_sq = np.full(n, np.inf, dtype=np.float64)
    for mask in range(1, 1 << K):
        ids = [i for i in range(K) if mask & (1 << i)]
        P = C[ids]  # [face_size, d]
        s = len(ids)
        # min_a ||P.T a - x||^2 subject to 1.T a = 1, for every x at once.
        kkt = np.empty((s + 1, s + 1), dtype=np.float64)
        kkt[:s, :s] = P @ P.T
        kkt[:s, s] = 1.0
        kkt[s, :s] = 1.0
        kkt[s, s] = 0.0
        rhs = np.vstack((P @ targets.T, np.ones((1, n))))
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        a = sol[:s]
        valid = np.all(a >= -1e-9, axis=0)
        if not np.any(valid):
            continue
        proj = a.T @ P
        dist_sq = np.sum((proj - targets) ** 2, axis=1)
        best_sq[valid] = np.minimum(best_sq[valid], dist_sq[valid])
    return np.sqrt(best_sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=300)
    args = ap.parse_args()

    model = args.model
    K = args.K
    d = NBITS[model]
    reg_size = full_registry_size(model)
    registry_bits = full_registry_bits(model)

    trial_idx = speaker_trial_index(n_total=args.n_trials)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / f"tamper_{model}_K{K}.csv"
    partial_csv = RESULTS / f"tamper_{model}_K{K}.partial.csv"
    rows, completed = load_completed_trials(partial_csv)
    if completed:
        print(f"  resuming {model} K={K}: {len(completed)}/{len(trial_idx)} trials from {partial_csv.name}",
              flush=True)
    t_start = time.time()

    for gi, (spk, local_t) in enumerate(trial_idx):
        if gi in completed:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        coll_ints = sample_coalition(rng, model, K)
        wavs = [get_or_embed(model, spk, ci) for ci in coll_ints]
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]
        C = np.array([int_to_bits(ci, d) for ci in coll_ints])  # [K,d] {0,1}

        # opportunistic：从注册表抽样子集找几何最近的 N_CAND 个 target（非 coalition 成员）
        # 全空间（65536）逐个算凸包距离太慢，随机抽 subset 个候选再选最近 N_CAND
        coll_set = set(coll_ints)
        subset_size = min(reg_size, 2000)
        cand_ids = [ci for ci in range(reg_size) if ci not in coll_set]
        rng_sub = np.random.default_rng(coalition_seed(spk, K, local_t) + 999)
        sample_ids = rng_sub.choice(cand_ids, size=min(subset_size, len(cand_ids)), replace=False)
        cand_dist = convex_dist_batch_exact(C, registry_bits[sample_ids])
        cands = sample_ids[np.argsort(cand_dist)[:N_CAND]].tolist()

        # Mean output is independent of the target.  Decode it once, then
        # compare that same top-1 identity against every opportunistic target.
        # This is exactly equivalent to the previous inner-loop calculation.
        mean_a = np.ones(K) / K
        mean_y = sum(mean_a[i] * wavs[i] for i in range(K)).astype(np.float32)

        # Each target still needs an independent TCT optimization.  WavMark
        # then decodes the resulting waveforms in a shared sliding-window batch.
        tct_outputs = []
        for q_t in cands:
            c_t = int_to_bits(q_t, d)
            a = tct(C, c_t, CAP)
            tct_outputs.append(sum(a[i] * wavs[i] for i in range(K)).astype(np.float32))

        if model == "wavmark":
            decoded = detect_wavmark_many(
                get_wavmark(), [mean_y, *tct_outputs], registry_bits)
        else:
            decoded = [detect(model, mean_y, registry_bits)]
            decoded.extend(detect(model, y, registry_bits) for y in tct_outputs)

        mean_scores, _, _ = decoded[0]
        mean_top1_idx = int(np.argsort(mean_scores)[::-1][0])
        mean_top1_int = int((registry_bits[mean_top1_idx] @ (2 ** np.arange(d))).sum())
        for q_t, (scores, _, _) in zip(cands, decoded[1:]):
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
                "target": q_t, "method": "mean", "target_top1": int(mean_top1_int == q_t),
            })
            # top1 转 codeword int 与 q_t 比较
            top1_idx = int(np.argsort(scores)[::-1][0])
            top1_int = int((registry_bits[top1_idx] @ (2 ** np.arange(d))).sum())
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t, "gi": gi,
                "target": q_t, "method": "tct", "target_top1": int(top1_int == q_t),
            })
        if (gi + 1) % 10 == 0:
            write_rows_atomic(partial_csv, rows)
            print(f"  {model} K={K}: {gi+1}/{len(trial_idx)} checkpointed ({time.time()-t_start:.0f}s)",
                  flush=True)

    write_rows_atomic(partial_csv, rows)
    write_rows_atomic(out_csv, rows)

    # 汇总：单候选平均命中率 + 每 trial N候选≥1
    print(f"\n=== {model} K={K} 篡改汇总（opportunistic, n={len(trial_idx)}）===")
    for m in ["mean", "tct"]:
        v = [r["target_top1"] for r in rows if r["method"] == m]
        print(f"  {m:12s}: 单候选命中率={np.mean(v)*100:.1f}%")
    # 每 trial ≥1 命中
    by_trial = {}
    for r in rows:
        if r["method"] == "tct":
            by_trial.setdefault((r["spk"], r["local_t"]), []).append(r["target_top1"])
    any_hit = [1 if max(v) == 1 else 0 for v in by_trial.values()]
    print(f"  tct  N候选≥1命中率={np.mean(any_hit)*100:.1f}%")


if __name__ == "__main__":
    main()

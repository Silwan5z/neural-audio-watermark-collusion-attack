"""攻击主脚本：全空间注册表 + 多说话人（38人）+ 300 trial 均匀分配。

方法：Mean / FWP（farthest waveform pair，核心两方法，全部盲）。
注：盲估计Gram的 blind_gram_cb 方法不在论文正文方法家族表中（详见 README），
本脚本不再产出该方法的数据。
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
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import (  # noqa: E402
    NBITS, clean_path_v19, get_or_embed, full_registry_bits,
    speaker_trial_index, coalition_seed, source_record,
)
from watermarks import detect, detect_many, pesq_wb, stoi, si_sdr  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "evaluation"

FIELDS = [
    "model", "K", "spk", "local_t", "clip_index", "source_path",
    "source_exact_count", "source_payload_attempts", "gi", "method",
    "ASR", "R3_escape", "R5_escape", "ACC_near", "ACC_near_norm",
    "AggResid", "PESQ", "STOI", "SI_SDR",
]


def atomic_csv(path: Path, rows: list[dict]) -> None:
    """Write a restart-safe checkpoint without exposing a partial CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: str, K: int,
                    expected_methods: set[str]) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row["model"] != model or int(row["K"]) != K:
            raise ValueError(f"checkpoint model/K mismatch: {path}")
        grouped.setdefault(int(row["gi"]), []).append(row)
    complete = {
        gi for gi, rr in grouped.items()
        if len(rr) == len(expected_methods)
        and {r["method"] for r in rr} == expected_methods
    }
    kept = [r for r in rows if int(r["gi"]) in complete]
    kept.sort(key=lambda r: (int(r["gi"]), r["method"]))
    return kept, complete


def fwp(wavs, K):
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


def metrics_of(model, y, coll_ints, registry_bits, a, d, decoded=None):
    """返回 (asr, r3_escape, r5_escape, acc_near, agg_resid)。"""
    scores, _, hard = decoded if decoded is not None else detect(model, y.astype(np.float32), registry_bits)
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


def source_correct_coalition(model, spk, local_t, K, rng, registry_bits,
                             max_attempts=1000):
    """Draw K distinct payloads whose individual copies decode exactly."""
    selected, wavs, attempts = [], [], 0
    limit = 2 ** NBITS[model]
    while len(selected) < K:
        needed = K - len(selected)
        candidates = []
        while len(candidates) < needed:
            payload = int(rng.integers(0, limit))
            if payload not in selected and payload not in candidates:
                candidates.append(payload)
        candidate_wavs = [get_or_embed(model, spk, p, local_t) for p in candidates]
        decoded = detect_many(model, candidate_wavs, registry_bits)
        attempts += len(candidates)
        for payload, wav, (_, _, hard) in zip(candidates, candidate_wavs, decoded):
            if hard is None:
                continue
            got = int(sum(int(v) << i for i, v in enumerate(hard)))
            if got == payload:
                selected.append(payload); wavs.append(wav)
        if attempts >= max_attempts and len(selected) < K:
            raise RuntimeError(
                f"{model} {spk} local_t={local_t}: only {len(selected)}/{K} "
                f"source-correct payloads after {attempts} attempts")
    return selected, wavs, attempts



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--output-dir", type=Path, default=RESULTS)
    ap.add_argument("--mean-only", action="store_true",
                    help="evaluate only uniform averaging")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="zero-based trial shard (default: 0)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of disjoint trial shards (default: 1)")
    args = ap.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("require 0 <= shard-id < num-shards")

    model = args.model
    K = args.K
    d = NBITS[model]

    registry_bits = full_registry_bits(model)
    trial_idx = speaker_trial_index(n_total=args.n_trials)

    stem = f"attack_{model}_K{K}"
    shard_suffix = ("" if args.num_shards == 1
                    else f".shard{args.shard_id}of{args.num_shards}")
    out_csv = args.output_dir / f"{stem}{shard_suffix}.csv"
    partial_csv = args.output_dir / f"{stem}{shard_suffix}.partial.csv"
    methods_expected = {"mean"} if args.mean_only else {"mean", "fwp"}
    rows, complete = load_checkpoint(partial_csv, model, K, methods_expected)
    # When a previously single-worker run exists, preserve it as the immutable
    # base and distribute only its missing global trial indices across shards.
    base_complete: set[int] = set()
    if args.num_shards > 1:
        base_partial = args.output_dir / f"{stem}.partial.csv"
        _, base_complete = load_checkpoint(base_partial, model, K, methods_expected)
    assigned = [
        (gi, item) for gi, item in enumerate(trial_idx)
        if gi not in base_complete and gi % args.num_shards == args.shard_id
    ]
    assigned_ids = {gi for gi, _ in assigned}
    if not complete.issubset(assigned_ids):
        raise ValueError(f"shard checkpoint contains unassigned trials: {partial_csv}")
    if complete:
        print(f"resume {model} K={K} shard={args.shard_id}/{args.num_shards}: "
              f"{len(complete)}/{len(assigned)} trials", flush=True)
    t_start = time.time()
    for gi, (spk, local_t) in assigned:
        if gi in complete:
            continue
        rng = np.random.default_rng(coalition_seed(spk, K, local_t))
        source = source_record(spk, local_t)
        coll_ints, wavs, source_attempts = source_correct_coalition(
            model, spk, local_t, K, rng, registry_bits)
        n = min(len(w) for w in wavs)
        wavs = [w[:n] for w in wavs]

        a_mean = np.ones(K) / K
        a_pair = None if args.mean_only else fwp(wavs, K)

        # 固定取 coalition 里的第一个成员做音质参照（不随机，同一 trial 内 mean/fwp
        # 两个方法共用同一个参照成员，保证组内可比）
        wm_ref = wavs[0]

        methods = [("mean", a_mean)]
        if a_pair is not None:
            methods.append(("fwp", a_pair))
        outputs = [sum(a[i] * wavs[i] for i in range(K)).astype(np.float32)
                   for _, a in methods]
        decoded_outputs = detect_many(model, outputs, registry_bits)
        for (mname, a), y, decoded in zip(methods, outputs, decoded_outputs):
            asr, r3e, r5e, acc, agg = metrics_of(model, y, coll_ints, registry_bits, a, d, decoded)
            # PESQ/STOI/SI-SDR 参照随机挑中的那个合谋者的水印音频（不是 clean）
            pesq = pesq_wb(wm_ref, y)
            st = stoi(wm_ref, y)
            sdr = si_sdr(wm_ref, y)
            rows.append({
                "model": model, "K": K, "spk": spk, "local_t": local_t,
                "clip_index": source["clip_index"],
                "source_path": str(clean_path_v19(spk, local_t)),
                "source_exact_count": K, "source_payload_attempts": source_attempts,
                "gi": gi, "method": mname,
                "ASR": asr, "R3_escape": r3e, "R5_escape": r5e,
                "ACC_near": "" if acc is None else acc,
                "ACC_near_norm": "" if acc is None else f"{acc/d:.4f}",
                "AggResid": f"{agg:.6f}",
                "PESQ": f"{pesq:.4f}", "STOI": f"{st:.4f}", "SI_SDR": f"{sdr:.2f}",
            })
        complete.add(gi)
        if len(complete) % 10 == 0:
            rows.sort(key=lambda r: (int(r["gi"]), r["method"]))
            atomic_csv(partial_csv, rows)
            elapsed = time.time() - t_start
            print(f"  {model} K={K} shard={args.shard_id}/{args.num_shards}: "
                  f"{len(complete)}/{len(assigned)}  ({elapsed:.0f}s)", flush=True)

    rows.sort(key=lambda r: (int(r["gi"]), r["method"]))
    atomic_csv(partial_csv, rows)
    atomic_csv(out_csv, rows)

    print(f"\n=== {model} K={K} shard={args.shard_id}/{args.num_shards} "
          f"汇总（n={len(assigned)}）===")
    for m in [x[0] for x in methods]:
        asrs = [int(r["ASR"]) for r in rows if r["method"] == m]
        r5s = [int(r["R5_escape"]) for r in rows if r["method"] == m]
        accs = [r["ACC_near_norm"] for r in rows if r["method"] == m and r["ACC_near_norm"] != ""]
        accs = [float(x) for x in accs]
        print(f"  {m:12s}: ASR={np.mean(asrs):.3f}  R5_escape={np.mean(r5s):.3f}"
              f"  ACC_near_norm={np.mean(accs) if accs else float('nan'):.3f}")


if __name__ == "__main__":
    main()

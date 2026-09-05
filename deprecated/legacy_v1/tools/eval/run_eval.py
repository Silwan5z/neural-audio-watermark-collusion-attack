"""
音频评估主程序
用法:
    python run_eval.py --config config.yaml
    python run_eval.py --ref_dir /path/to/ref --deg_dir /path/to/deg --output results.csv
"""

import os
import sys
import argparse
import glob
import csv
import json
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from eval_metrics import evaluate_pair

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 支持的音频格式
# ─────────────────────────────────────────────
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


# ─────────────────────────────────────────────
# 文件配对逻辑
# ─────────────────────────────────────────────

def find_audio_files(directory: str) -> List[Path]:
    """递归查找目录下所有音频文件，按文件名排序"""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(files)


def pair_files(
    ref_dir: str,
    deg_dir: str,
    match_by: str = "filename",
) -> List[Tuple[Path, Path]]:
    """
    将参考目录和降质目录中的文件配对

    match_by:
        'filename' - 按文件名（不含扩展名）匹配（默认）
        'order'    - 按排序顺序一一对应
    """
    ref_files = find_audio_files(ref_dir)
    deg_files = find_audio_files(deg_dir)

    if not ref_files:
        raise FileNotFoundError(f"参考目录中未找到音频文件: {ref_dir}")
    if not deg_files:
        raise FileNotFoundError(f"降质目录中未找到音频文件: {deg_dir}")

    pairs = []

    if match_by == "filename":
        # 构建 stem → path 映射
        deg_map: Dict[str, Path] = {}
        for f in deg_files:
            deg_map[f.stem] = f

        for ref_f in ref_files:
            if ref_f.stem in deg_map:
                pairs.append((ref_f, deg_map[ref_f.stem]))
            else:
                print(f"  [警告] 未找到对应的降质文件: {ref_f.name}，已跳过")

        if not pairs:
            raise ValueError(
                "按文件名匹配失败，未找到任何配对。\n"
                "请检查两个目录中的文件名是否一致，或改用 match_by: order"
            )

    elif match_by == "order":
        n = min(len(ref_files), len(deg_files))
        if len(ref_files) != len(deg_files):
            print(f"  [警告] 文件数量不一致 (ref={len(ref_files)}, deg={len(deg_files)})，"
                  f"仅处理前 {n} 对")
        pairs = list(zip(ref_files[:n], deg_files[:n]))

    else:
        raise ValueError(f"不支持的 match_by 值: {match_by}")

    return pairs


# ─────────────────────────────────────────────
# 结果统计
# ─────────────────────────────────────────────

def summarize(records: List[Dict]) -> Dict[str, float]:
    """计算各指标的均值、标准差、最大值、最小值"""
    if not records:
        return {}

    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if c not in ("ref_file", "deg_file")]

    summary = {}
    for col in metric_cols:
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        summary[f"{col}_mean"] = float(valid.mean())
        summary[f"{col}_std"]  = float(valid.std())
        summary[f"{col}_min"]  = float(valid.min())
        summary[f"{col}_max"]  = float(valid.max())
    return summary


# ─────────────────────────────────────────────
# 主评估流程
# ─────────────────────────────────────────────

def run_evaluation(cfg: Dict) -> None:
    """
    根据配置字典执行完整评估流程

    cfg 字段说明:
        ref_dir          : 参考音频目录（必填）
        deg_dir          : 待评估音频目录（必填）
        output_csv       : 结果 CSV 路径（默认 results.csv）
        output_json      : 汇总统计 JSON 路径（默认 summary.json）
        metrics          : 要计算的指标列表（默认全部）
        match_by         : 文件配对方式 filename/order（默认 filename）
        pesq_sr          : PESQ 采样率 8000/16000（默认 16000）
        stoi_extended    : 是否使用 ESTOI（默认 False）
        visqol_speech_mode: ViSQOL speech 模式（默认 True）
    """
    ref_dir   = cfg["ref_dir"]
    deg_dir   = cfg["deg_dir"]
    out_csv   = cfg.get("output_csv",  "results.csv")
    out_json  = cfg.get("output_json", "summary.json")
    metrics   = tuple(cfg.get("metrics", ["pesq", "sisnr", "stoi", "visqol"]))
    match_by  = cfg.get("match_by", "filename")

    pesq_sr            = int(cfg.get("pesq_sr", 16000))
    stoi_extended      = bool(cfg.get("stoi_extended", False))
    visqol_speech_mode = bool(cfg.get("visqol_speech_mode", True))

    print(f"\n{'='*60}")
    print(f"  音频评估任务")
    print(f"{'='*60}")
    print(f"  参考目录  : {ref_dir}")
    print(f"  评估目录  : {deg_dir}")
    print(f"  评估指标  : {', '.join(metrics).upper()}")
    print(f"  配对方式  : {match_by}")
    print(f"  输出 CSV  : {out_csv}")
    print(f"  输出 JSON : {out_json}")
    print(f"{'='*60}\n")

    # 文件配对
    pairs = pair_files(ref_dir, deg_dir, match_by=match_by)
    print(f"共找到 {len(pairs)} 对音频文件\n")

    records = []
    failed  = []

    for ref_path, deg_path in tqdm(pairs, desc="评估进度", unit="pair"):
        row: Dict = {
            "ref_file": str(ref_path),
            "deg_file": str(deg_path),
        }
        try:
            scores = evaluate_pair(
                ref_path=str(ref_path),
                deg_path=str(deg_path),
                metrics=metrics,
                pesq_sr=pesq_sr,
                stoi_extended=stoi_extended,
                visqol_speech_mode=visqol_speech_mode,
            )
            row.update(scores)
        except Exception as e:
            print(f"\n  [错误] {ref_path.name}: {e}")
            failed.append(str(ref_path))
            for m in metrics:
                row[m] = float("nan")

        records.append(row)

    # ── 保存逐文件结果 ──
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\n✅ 逐文件结果已保存至: {out_csv}")

    # ── 打印汇总统计 ──
    summary = summarize(records)
    print(f"\n{'─'*50}")
    print("  评估汇总统计")
    print(f"{'─'*50}")
    metric_names = list(metrics)
    for m in metric_names:
        mean_key = f"{m}_mean"
        if mean_key in summary:
            print(f"  {m.upper():10s}  "
                  f"mean={summary[f'{m}_mean']:7.4f}  "
                  f"std={summary[f'{m}_std']:6.4f}  "
                  f"min={summary[f'{m}_min']:7.4f}  "
                  f"max={summary[f'{m}_max']:7.4f}")
    print(f"{'─'*50}")

    # ── 保存汇总 JSON ──
    summary["total_pairs"]  = len(pairs)
    summary["failed_pairs"] = len(failed)
    summary["failed_files"] = failed

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总统计已保存至: {out_json}")

    if failed:
        print(f"\n⚠️  {len(failed)} 个文件处理失败，详见 {out_json}")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="音频质量评估工具 (PESQ / SI-SNR / STOI / ViSQOL)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="YAML 配置文件路径（优先级高于命令行参数）"
    )
    parser.add_argument(
        "--ref_dir", type=str, default=None,
        help="参考音频目录（干净语音）"
    )
    parser.add_argument(
        "--deg_dir", type=str, default=None,
        help="待评估音频目录"
    )
    parser.add_argument(
        "--output_csv", type=str, default="results.csv",
        help="逐文件结果输出路径（默认: results.csv）"
    )
    parser.add_argument(
        "--output_json", type=str, default="summary.json",
        help="汇总统计输出路径（默认: summary.json）"
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["pesq", "sisnr", "stoi", "visqol"],
        choices=["pesq", "sisnr", "stoi", "visqol"],
        help="要计算的指标（默认: 全部）"
    )
    parser.add_argument(
        "--match_by", type=str, default="filename",
        choices=["filename", "order"],
        help="文件配对方式（默认: filename）"
    )
    parser.add_argument(
        "--pesq_sr", type=int, default=16000,
        choices=[8000, 16000],
        help="PESQ 采样率（默认: 16000）"
    )
    parser.add_argument(
        "--stoi_extended", action="store_true",
        help="使用 Extended STOI (ESTOI)"
    )
    parser.add_argument(
        "--visqol_audio_mode", action="store_true",
        help="ViSQOL 使用 audio 模式（48kHz），默认为 speech 模式（16kHz）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 优先读取 YAML 配置文件
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        if args.ref_dir is None or args.deg_dir is None:
            print("错误: 请通过 --config 指定配置文件，或同时提供 --ref_dir 和 --deg_dir")
            sys.exit(1)
        cfg = {
            "ref_dir":            args.ref_dir,
            "deg_dir":            args.deg_dir,
            "output_csv":         args.output_csv,
            "output_json":        args.output_json,
            "metrics":            args.metrics,
            "match_by":           args.match_by,
            "pesq_sr":            args.pesq_sr,
            "stoi_extended":      args.stoi_extended,
            "visqol_speech_mode": not args.visqol_audio_mode,
        }

    run_evaluation(cfg)


if __name__ == "__main__":
    main()

"""Scheduler for the margin_reachability_v2 production batch.

Launches one margin_reachability_v2.py subprocess per (model, K) job, each
pinned to a single GPU via WATERMARK_DEVICE, and keeps up to N_GPUS jobs
running at once, assigning the next queued job to whichever GPU frees up.
Jobs are ordered slowest-first (K=8, 16-bit models, wavmark) so the biggest
jobs get the longest run of wall-clock time rather than waiting behind a
pile of quick ones.

Each underlying job is independently checkpointed/resumable (partial CSV),
so killing and re-running this scheduler is safe.

Usage:
    python scripts/run_margin_reachability_batch.py
    python scripts/run_margin_reachability_batch.py --gpus 0,1,2,3,4,5,6 --n_trials 300
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["audioseal", "wavmark", "voicemark", "wmcodec", "timbrewm"]
KS = [2, 3, 5, 8]
LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "margin_reachability_v2"


def job_weight(model, K):
    """Rough relative cost, for scheduling order only (not a timing guarantee)."""
    w = K
    if model == "wavmark":
        w *= 3.0  # sliding-window decode is the slowest backend
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--poll_interval", type=float, default=15.0)
    args = ap.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    jobs = sorted(((model, K) for model in MODELS for K in KS),
                  key=lambda mk: job_weight(*mk), reverse=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    queue = list(jobs)
    running = {}  # gpu_id -> (proc, model, K, logfile)
    free_gpus = list(gpu_ids)
    done, failed = [], []

    print(f"scheduling {len(jobs)} jobs across {len(gpu_ids)} GPUs: {gpu_ids}", flush=True)

    while queue or running:
        while queue and free_gpus:
            model, K = queue.pop(0)
            gpu = free_gpus.pop(0)
            log_path = LOG_DIR / f"{model}_K{K}.log"
            cmd = [sys.executable, str(Path(__file__).resolve().parent / "margin_reachability_v2.py"),
                   "--model", model, "--K", str(K), "--n_trials", str(args.n_trials)]
            env = {**os.environ, "WATERMARK_DEVICE": f"cuda:{gpu}"}
            logf = log_path.open("w")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (proc, model, K, logf)
            print(f"  [gpu {gpu}] launched {model} K={K} (pid={proc.pid}) -> {log_path}", flush=True)

        time.sleep(args.poll_interval)

        for gpu, (proc, model, K, logf) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            logf.close()
            del running[gpu]
            free_gpus.append(gpu)
            if ret == 0:
                done.append((model, K))
                print(f"  [gpu {gpu}] finished {model} K={K}", flush=True)
            else:
                failed.append((model, K))
                print(f"  [gpu {gpu}] FAILED {model} K={K} (exit={ret}), see {LOG_DIR / f'{model}_K{K}.log'}",
                      flush=True)

    print(f"\nbatch complete: {len(done)} ok, {len(failed)} failed")
    if failed:
        print("failed jobs:", failed)


if __name__ == "__main__":
    main()

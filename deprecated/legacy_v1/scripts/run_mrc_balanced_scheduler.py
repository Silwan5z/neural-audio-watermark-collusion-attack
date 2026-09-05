#!/usr/bin/env python3
"""Persistent 7-GPU scheduler for balanced MRC/Softmin-v2 ablations.

Every job advances exactly one (variant, model, K) cell by ten completed
trials.  After any job exits, the scheduler rescans all partial CSVs and gives
the free GPU to a cell with the shortest complete prefix.  Thus fast variants
automatically donate their GPUs to slow variants, and an interruption always
leaves a maximally large rectangular/common-prefix result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"
LOG_ROOT = ROOT / "results" / "logs" / "mrc_balanced_scheduler"
STATUS = LOG_ROOT / "status.json"
RUNNER = ROOT / "scripts" / "run_mrc_ablation.py"
PYTHON = Path("/private/users/lym/venv/bin/python")
VARIANTS = (
    "beta5", "beta10", "no_keff_floor", "uniform_floor",
    "entropy005", "entropy005_keff_floor", "hull_targets", "uniform_weights",
)
MODELS = ("wavmark", "voicemark", "wmcodec", "timbrewm", "audioseal")
KS = (2, 3, 5, 8)
MODEL_COST = {"wavmark": 5, "voicemark": 4, "wmcodec": 3,
              "timbrewm": 2, "audioseal": 1}
CHUNK = 10
MAX_TRIALS = 300


@dataclass(frozen=True)
class Cell:
    variant: str
    model: str
    k: int

    @property
    def stem(self) -> str:
        return f"mrc_ablation_{self.variant}_N1024_{self.model}_K{self.k}"


@dataclass
class Running:
    cell: Cell
    start: int
    end: int
    process: subprocess.Popen
    log_handle: object
    started_at: float


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def complete_prefix(cell: Cell) -> int:
    # A runner may atomically replace or promote the partial file between the
    # existence check and open(). Retry the scan so this harmless checkpoint
    # race cannot terminate the persistent scheduler.
    partial = RESULTS / f"{cell.stem}.partial.csv"
    final = RESULTS / f"{cell.stem}.csv"
    counts: Counter[int] = Counter()
    for attempt in range(5):
        path = partial if partial.exists() else final
        if not path.exists():
            return 0
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    counts[int(row["gi"])] += 1
            break
        except FileNotFoundError:
            counts.clear()
            if attempt == 4:
                raise
            time.sleep(0.05)
    prefix = 0
    while prefix < MAX_TRIALS and counts[prefix] == 10:
        prefix += 1
    return prefix


def all_cells() -> list[Cell]:
    return [Cell(variant, model, k) for variant in VARIANTS
            for model in MODELS for k in KS]


def snapshot(cells: list[Cell]) -> dict[Cell, int]:
    return {cell: complete_prefix(cell) for cell in cells}


def choose_cell(progress: dict[Cell, int], occupied: set[Cell],
                failures: dict[Cell, int], retry_after: dict[Cell, float]) -> Cell | None:
    current = time.time()
    eligible = [cell for cell, done in progress.items()
                if done < MAX_TRIALS and cell not in occupied
                and current >= retry_after.get(cell, 0.0)]
    if not eligible:
        return None
    # Shortest prefix first gives balance.  For ties, longest jobs first avoids
    # leaving WavMark as a serial tail.  Failure count is only a final tie-break.
    return min(eligible, key=lambda cell: (
        progress[cell], -MODEL_COST[cell.model], failures.get(cell, 0),
        cell.k, cell.variant, cell.model))


def launch(gpu: int, cell: Cell, start: int, end: int) -> Running:
    log_dir = LOG_ROOT / f"gpu{gpu}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell.stem}.log"
    handle = log_path.open("a", buffering=1)
    command = [
        str(PYTHON), "-u", str(RUNNER), "--variant", cell.variant,
        "--model", cell.model, "--K", str(cell.k),
        "--n_trials", str(MAX_TRIALS), "--trial_start", "0",
        "--trial_end", str(end),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "WATERMARK_DEVICE": "cuda:0",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "MRC_TARGET_WORKERS": "8",
        "WAVMARK_WINDOW_BATCH_SIZE": "600",
    })
    handle.write(
        f"[{now()}] LAUNCH gpu={gpu} range=[{start},{end}) "
        f"cmd={' '.join(command)}\n")
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=handle,
        stderr=subprocess.STDOUT, start_new_session=True)
    return Running(cell, start, end, process, handle, time.time())


def status_payload(progress: dict[Cell, int], running: dict[int, Running],
                   failures: dict[Cell, int]) -> dict:
    by_k = {str(k): min(progress[cell] for cell in progress if cell.k == k)
            for k in KS}
    by_variant = {
        variant: min(progress[cell] for cell in progress
                     if cell.variant == variant)
        for variant in VARIANTS
    }
    return {
        "updated_at": now(), "pid": os.getpid(),
        "global_balanced_prefix": min(progress.values()),
        "minimum_prefix_by_k": by_k,
        "minimum_prefix_by_variant": by_variant,
        "completed_cells": sum(value == MAX_TRIALS for value in progress.values()),
        "total_cells": len(progress),
        "running": {
            str(gpu): {"pid": task.process.pid, "variant": task.cell.variant,
                       "model": task.cell.model, "K": task.cell.k,
                       "start": task.start, "end": task.end,
                       "started_at": datetime.fromtimestamp(
                           task.started_at).astimezone().isoformat(timespec="seconds")}
            for gpu, task in running.items()
        },
        "failures": {cell.stem: count for cell, count in failures.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    cells = all_cells()
    progress = snapshot(cells)
    if args.dry_run:
        occupied: set[Cell] = set()
        for gpu in gpus:
            cell = choose_cell(progress, occupied, {}, {})
            if cell is None:
                break
            occupied.add(cell)
            print(gpu, cell, f"[{progress[cell]},{min(progress[cell]+CHUNK, MAX_TRIALS)})")
        print(json.dumps(status_payload(progress, {}, {}), indent=2))
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    running: dict[int, Running] = {}
    failures: dict[Cell, int] = {}
    retry_after: dict[Cell, float] = {}
    stopping = False

    def request_stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[{now()}] received signal {signum}; stopping children", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(f"[{now()}] scheduler start pid={os.getpid()} gpus={gpus}", flush=True)

    while not stopping:
        progress_changed = False
        for gpu, task in list(running.items()):
            code = task.process.poll()
            if code is None:
                continue
            task.log_handle.write(
                f"[{now()}] EXIT gpu={gpu} code={code} elapsed="
                f"{time.time()-task.started_at:.1f}s\n")
            task.log_handle.close()
            del running[gpu]
            progress_changed = True
            if code != 0:
                failures[task.cell] = failures.get(task.cell, 0) + 1
                # Retry later without blocking other balanced cells.  Backoff
                # is capped at ten minutes and remains visible in status.json.
                retry_after[task.cell] = time.time() + min(
                    600, 60 * failures[task.cell])

        # Runner outputs are atomically replaced only at checkpoints/end, so a
        # full 140-file rescan is needed only after a child exits, not every
        # five-second health poll.
        if progress_changed:
            progress = snapshot(cells)
        occupied = {task.cell for task in running.values()}
        for gpu in gpus:
            if gpu in running:
                continue
            cell = choose_cell(progress, occupied, failures, retry_after)
            if cell is None:
                continue
            start = progress[cell]
            end = min(start + CHUNK, MAX_TRIALS)
            running[gpu] = launch(gpu, cell, start, end)
            occupied.add(cell)

        atomic_json(STATUS, status_payload(progress, running, failures))
        if all(value == MAX_TRIALS for value in progress.values()) and not running:
            print(f"[{now()}] ALL_CELLS_DONE", flush=True)
            break
        time.sleep(args.poll_seconds)

    if stopping:
        for task in running.values():
            if task.process.poll() is None:
                os.killpg(task.process.pid, signal.SIGTERM)
        for task in running.values():
            try:
                task.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task.process.pid, signal.SIGKILL)
            task.log_handle.close()
    progress = snapshot(cells)
    atomic_json(STATUS, status_payload(progress, {}, failures))


if __name__ == "__main__":
    main()

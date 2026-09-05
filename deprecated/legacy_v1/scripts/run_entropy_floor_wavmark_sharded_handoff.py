#!/usr/bin/env python3
"""Hand off the final WavMark tail from four canonical cells to seven GPUs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "evaluation"
LOG_ROOT = ROOT / "results" / "logs" / "entropy005_keff_floor_wavmark_shards"
RUNNER = ROOT / "scripts" / "run_mrc_ablation.py"
PYTHON = Path("/private/users/lym/venv/bin/python")
VARIANT = "entropy005_keff_floor"
KS = (2, 3, 5, 8)
NON_WAVMARK = ("audioseal", "timbrewm", "voicemark", "wmcodec")
FIELDS = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical(k: int, partial: bool = True) -> Path:
    tail = ".partial.csv" if partial else ".csv"
    return RESULTS / f"mrc_ablation_{VARIANT}_N1024_wavmark_K{k}{tail}"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def atomic_write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def prefix_for(k: int) -> tuple[int, list[str], list[dict[str, str]]]:
    path = canonical(k, True)
    if not path.exists():
        path = canonical(k, False)
    fields, rows = read_rows(path)
    counts = Counter(int(row["gi"]) for row in rows)
    prefix = 0
    while prefix < 300 and counts[prefix] == 10:
        prefix += 1
    if any(gi >= prefix for gi, count in counts.items() if count == 10):
        raise ValueError(f"non-prefix complete trial in {path}")
    rows = [row for row in rows if int(row["gi"]) < prefix]
    if len(rows) != prefix * 10:
        raise ValueError(f"invalid canonical prefix in {path}: prefix={prefix}, rows={len(rows)}")
    return prefix, fields, rows


def nonwav_final_count() -> int:
    return sum((RESULTS / f"mrc_ablation_{VARIANT}_N1024_{model}_K{k}.csv").exists()
               for model in NON_WAVMARK for k in KS)


def stop_canonical_scheduler() -> None:
    status_path = ROOT / "results" / "logs" / "mrc_balanced_scheduler" / "status.json"
    status = json.loads(status_path.read_text())
    pid = int(status["pid"])
    command = Path(f"/proc/{pid}/cmdline")
    if not command.exists() or "run_mrc_balanced_scheduler.py" not in command.read_bytes().replace(b"\0", b" ").decode():
        raise RuntimeError(f"refusing to signal unverified scheduler pid={pid}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(90):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(1)
    else:
        raise TimeoutError(f"scheduler pid={pid} did not stop cleanly")
    subprocess.run(["tmux", "kill-session", "-t", "mrc_balanced_scheduler"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # The scheduler terminates and waits for every child process group.
    processes = subprocess.run(["ps", "-eo", "args="], text=True, capture_output=True, check=True).stdout
    active = [line for line in processes.splitlines()
              if "run_mrc_ablation.py" in line and f"--variant {VARIANT}" in line]
    if active:
        raise RuntimeError(f"MRC children remain after clean stop: {active[:3]}")


def require_gpus_free() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    active = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError(
            "refusing to compete for GPUs after scheduler handoff; active compute processes: "
            + "; ".join(active[:20]))


def run_block(gpu: int, k: int, start: int, end: int) -> None:
    suffix = f"wavshard_K{k}_{start:03d}_{end:03d}"
    log = LOG_ROOT / f"gpu{gpu}_K{k}_{start:03d}_{end:03d}.log"
    command = [str(PYTHON), "-u", str(RUNNER), "--variant", VARIANT,
               "--model", "wavmark", "--K", str(k), "--n_trials", "300",
               "--trial_start", str(start), "--trial_end", str(end),
               "--output-suffix", suffix]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "WATERMARK_DEVICE": "cuda:0",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "MRC_TARGET_WORKERS": "8", "WAVMARK_WINDOW_BATCH_SIZE": "600"})
    for attempt in range(1, 4):
        with log.open("a") as handle:
            handle.write(f"[{now()}] launch attempt={attempt} {' '.join(command)}\n")
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        time.sleep(10 * attempt)
    raise RuntimeError(f"block failed after 3 attempts gpu={gpu} K={k} [{start},{end}); log={log}")


def merge(k: int, prefix: int, fields: list[str], base_rows: list[dict[str, str]],
          blocks: list[tuple[int, int, int]]) -> None:
    rows = list(base_rows)
    for block_k, start, end in blocks:
        if block_k != k:
            continue
        suffix = f"wavshard_K{k}_{start:03d}_{end:03d}"
        path = RESULTS / f"mrc_ablation_{VARIANT}_N1024_wavmark_K{k}_{suffix}.partial.csv"
        shard_fields, shard_rows = read_rows(path)
        if shard_fields != fields:
            raise ValueError(f"schema mismatch: {path}")
        counts = Counter(int(row["gi"]) for row in shard_rows)
        if set(counts) != set(range(start, end)) or any(value != 10 for value in counts.values()):
            raise ValueError(f"invalid shard coverage: {path}")
        rows.extend(shard_rows)
    keys = [(int(row["gi"]), int(row["target"])) for row in rows]
    if len(rows) != 3000 or len(set(keys)) != 3000:
        raise ValueError(f"merged WavMark K={k} is not unique 300x10")
    counts = Counter(gi for gi, _ in keys)
    if set(counts) != set(range(300)) or any(value != 10 for value in counts.values()):
        raise ValueError(f"merged WavMark K={k} trial layout invalid")
    rows.sort(key=lambda row: (int(row["gi"]), int(row["target_rank"])))
    atomic_write(canonical(k, True), fields, rows)
    atomic_write(canonical(k, False), fields, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        while nonwav_final_count() < 16:
            print(f"[{now()}] waiting nonwav finals={nonwav_final_count()}/16", flush=True)
            time.sleep(30)
        print(f"[{now()}] nonwav complete; stopping canonical scheduler", flush=True)
        stop_canonical_scheduler()
        require_gpus_free()

    states = {k: prefix_for(k) for k in KS}
    blocks = [(k, start, min(start + 10, 300)) for k in KS
              for start in range(states[k][0], 300, 10)]
    # Large-K blocks are marginally more CPU intensive; start them first.
    blocks.sort(key=lambda item: (-item[0], item[1]))
    print("prefixes", {k: states[k][0] for k in KS}, "blocks", len(blocks), flush=True)
    if args.dry_run:
        loads = {gpu: 0 for gpu in gpus}
        for block in blocks:
            gpu = min(loads, key=loads.get); loads[gpu] += block[2] - block[1]
            print("gpu", gpu, "block", block)
        print("loads", loads)
        return

    tasks: queue.Queue[tuple[int, int, int]] = queue.Queue()
    for block in blocks:
        tasks.put(block)
    errors: list[str] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                k, start, end = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                run_block(gpu, k, start, end)
            except Exception as exc:  # preserve other GPUs; fail merge later
                with lock:
                    errors.append(str(exc))
            finally:
                tasks.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    for k in KS:
        prefix, fields, base_rows = states[k]
        merge(k, prefix, fields, base_rows, blocks)
    (LOG_ROOT / "ALL_WAVMARK_SHARDS_DONE").write_text(now() + "\n")
    print(f"[{now()}] SHARDED_WAVMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?GPU id required}
SHARD_ID=${2:?shard id required}
NUM_SHARDS=${3:-7}
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PY=/private/users/lym/venv/bin/python
OUT="$ROOT/results/mrc_pm_native_shared4_fulltop10_300clips_20260904"
SHARED="$ROOT/data/shared4_coalitions_300clips_20260904"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export WATERMARK_DEVICE=cuda:0
export PYTHONPATH="/tmp/k5case_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WAVMARK_WINDOW_BATCH_SIZE=256

cd "$ROOT"
echo "wavmark_worker_start gpu=$GPU_ID shard=$SHARD_ID/$NUM_SHARDS utc=$(date -u +%FT%TZ)"
for kval in 5 8; do
  for method in mrc pm; do
    "$PY" -u scripts/run_mrc_pm_native_300clips_shard.py \
      --method "$method" --model wavmark --K "$kval" \
      --n-trials 300 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
      --target-workers 16 --output-dir "$OUT" \
      --shared-coalition-dir "$SHARED"
  done
done
echo "wavmark_worker_complete gpu=$GPU_ID shard=$SHARD_ID/$NUM_SHARDS utc=$(date -u +%FT%TZ)"

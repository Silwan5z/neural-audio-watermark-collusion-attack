#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?GPU id required}
SHARD_ID=${2:?shard id required}
NUM_SHARDS=${3:-7}
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PY=/private/users/lym/venv/bin/python
OUT="$ROOT/results/mrc_pm_native_fulltop10_300clips_20260904"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export WATERMARK_DEVICE=cuda:0
export PYTHONPATH="/tmp/k5case_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WAVMARK_WINDOW_BATCH_SIZE=256

cd "$ROOT"
echo "worker_start gpu=$GPU_ID shard=$SHARD_ID/$NUM_SHARDS utc=$(date -u +%FT%TZ)"

# Run paired cells consecutively so PM reuses the source-correct waveform cache
# populated by MRC.  Separate Python processes release each model before the
# next cell and keep GPU memory bounded.
for model in audioseal timbrewm voicemark wmcodec wavmark; do
  for kval in 5 8; do
    for method in mrc pm; do
      echo "cell_start gpu=$GPU_ID shard=$SHARD_ID method=$method model=$model K=$kval utc=$(date -u +%FT%TZ)"
      "$PY" -u scripts/run_mrc_pm_native_300clips_shard.py \
        --method "$method" --model "$model" --K "$kval" \
        --n-trials 300 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
        --target-workers 16 --output-dir "$OUT"
      echo "cell_complete gpu=$GPU_ID shard=$SHARD_ID method=$method model=$model K=$kval utc=$(date -u +%FT%TZ)"
    done
  done
done

echo "worker_complete gpu=$GPU_ID shard=$SHARD_ID/$NUM_SHARDS utc=$(date -u +%FT%TZ)"

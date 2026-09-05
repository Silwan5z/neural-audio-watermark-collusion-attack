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
echo "worker_start gpu=$GPU_ID shard=$SHARD_ID/$NUM_SHARDS utc=$(date -u +%FT%TZ)"

# Build one source-correct 16-bit coalition shared by AudioSeal, WavMark,
# VoiceMark, and WMCodec. TimbreWM remains independent because it is 10-bit.
for kval in 5 8; do
  for model in voicemark wmcodec; do
    "$PY" -u scripts/prepare_shared4_coalitions.py \
      --stage validate --model "$model" --pool full --K "$kval" \
      --n-trials 300 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
      --candidate-count 64 --output-dir "$SHARED"
  done
  "$PY" -u scripts/prepare_shared4_coalitions.py \
    --stage preliminary --K "$kval" --n-trials 300 \
    --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
    --defer-insufficient --output-dir "$SHARED"
  # Only unusually hard trials expand to a large pool. This keeps the common
  # case fast while preserving the exact shared-source condition.
  for model in voicemark wmcodec; do
    "$PY" -u scripts/prepare_shared4_coalitions.py \
      --stage validate --model "$model" --pool full --K "$kval" \
      --n-trials 300 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
      --candidate-count 1024 --only-expansion-requests --output-dir "$SHARED"
  done
  "$PY" -u scripts/prepare_shared4_coalitions.py \
    --stage preliminary --K "$kval" --n-trials 300 \
    --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
    --output-dir "$SHARED"
  for model in audioseal; do
    "$PY" -u scripts/prepare_shared4_coalitions.py \
      --stage validate --model "$model" --pool preliminary --K "$kval" \
      --n-trials 300 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
      --output-dir "$SHARED"
  done
  "$PY" -u scripts/prepare_shared4_coalitions.py \
    --stage finalize --K "$kval" --n-trials 300 \
    --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
    --defer-wavmark-validation --output-dir "$SHARED"
done

# AudioSeal goes first and writes the shared MRC/PM target cache. The other
# three 16-bit systems reuse exactly those targets and weights. TimbreWM uses
# its own source-correct coalition and therefore its own 10-bit target search.
for model in audioseal timbrewm voicemark wmcodec; do
  for kval in 5 8; do
    for method in mrc pm; do
      echo "cell_start gpu=$GPU_ID shard=$SHARD_ID method=$method model=$model K=$kval utc=$(date -u +%FT%TZ)"
      "$PY" -u scripts/run_mrc_pm_native_300clips_shard.py \
        --method "$method" --model "$model" --K "$kval" \
        --n-trials 300 --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
        --target-workers 16 --output-dir "$OUT" \
        --shared-coalition-dir "$SHARED"
      echo "cell_complete gpu=$GPU_ID shard=$SHARD_ID method=$method model=$model K=$kval utc=$(date -u +%FT%TZ)"
    done
  done
done

echo "worker_nonwav_complete gpu=$GPU_ID shard=$SHARD_ID/$NUM_SHARDS utc=$(date -u +%FT%TZ)"

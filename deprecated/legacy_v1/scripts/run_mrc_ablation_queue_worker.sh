#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU VARIANT WAIT_RUN_ID" >&2
  exit 2
fi
GPU="$1"
VARIANT="$2"
WAIT_RUN_ID="$3"
ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
WAIT_FILE="$ROOT/results/logs/favorable_n1024/$WAIT_RUN_ID/status/gpu${GPU}.done"
LOG_DIR="$ROOT/results/logs/mrc_ablation_7gpu"
mkdir -p "$LOG_DIR"

while [[ ! -f "$WAIT_FILE" ]]; do sleep 15; done

source /private/users/lym/venv/bin/activate
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export WATERMARK_DEVICE="cuda:0"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MRC_TARGET_WORKERS=8
# 600 windows uses about 35 GB on the 46 GB cards and reduces the number of
# WavMark forward batches per trial from five to three.
export WAVMARK_WINDOW_BATCH_SIZE=600

LOG="$LOG_DIR/gpu${GPU}_${VARIANT}.log"
echo "[$(date '+%F %T')] START gpu=$GPU variant=$VARIANT" | tee -a "$LOG"
# Advance every K in ten-trial rounds.  Rotate the K and model order between
# rounds so an interruption leaves a nearly rectangular, directly comparable
# prefix instead of one fully completed K and three empty K values.
KS=(2 3 5 8)
MODELS=(audioseal timbrewm voicemark wmcodec wavmark)
for TRIAL_END in $(seq 10 10 300); do
  ROUND=$((TRIAL_END / 10 - 1))
  echo "[$(date '+%F %T')] ROUND variant=$VARIANT trial_end=$TRIAL_END" | tee -a "$LOG"
  for K_OFFSET in 0 1 2 3; do
    K_INDEX=$(((ROUND + K_OFFSET) % 4))
    K="${KS[$K_INDEX]}"
    for MODEL_OFFSET in 0 1 2 3 4; do
      MODEL_INDEX=$(((ROUND + MODEL_OFFSET) % 5))
      MODEL="${MODELS[$MODEL_INDEX]}"
      echo "[$(date '+%F %T')] CELL variant=$VARIANT model=$MODEL K=$K trial_end=$TRIAL_END" | tee -a "$LOG"
      python -u scripts/run_mrc_ablation.py \
        --variant "$VARIANT" --model "$MODEL" --K "$K" --n_trials 300 \
        --trial_end "$TRIAL_END" 2>&1 | tee -a "$LOG"
    done
  done
done
echo "[$(date '+%F %T')] ALL_CELLS_DONE gpu=$GPU variant=$VARIANT" | tee -a "$LOG"

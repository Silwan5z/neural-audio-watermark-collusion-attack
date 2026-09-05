#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL PHYSICAL_GPU K [K ...]" >&2
  exit 2
fi

MODEL="$1"
PHYSICAL_GPU="$2"
shift 2

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
VENV="/private/users/lym/venv"
LOG_DIR="$ROOT/results/logs/favorable_n1024"
mkdir -p "$LOG_DIR"

source "$VENV/bin/activate"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export WATERMARK_DEVICE="cuda:0"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

for K in "$@"; do
  LOG="$LOG_DIR/${MODEL}_K${K}_gpu${PHYSICAL_GPU}.log"
  echo "[$(date '+%F %T')] START model=$MODEL K=$K N=1024 gpu=$PHYSICAL_GPU" | tee -a "$LOG"
  python -u scripts/framing.py \
    --model "$MODEL" \
    --K "$K" \
    --n_trials 300 \
    --target_policy opportunistic \
    --registry_size 1024 2>&1 | tee -a "$LOG"
  echo "[$(date '+%F %T')] DONE model=$MODEL K=$K N=1024 gpu=$PHYSICAL_GPU" | tee -a "$LOG"
done

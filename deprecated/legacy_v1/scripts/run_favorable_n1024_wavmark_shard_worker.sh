#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU RUN_ID K:START:END:TAG [...]" >&2
  exit 2
fi

GPU="$1"
RUN_ID="$2"
shift 2
ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
LOG_DIR="$ROOT/results/logs/favorable_n1024/$RUN_ID"
STATUS_DIR="$LOG_DIR/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

source /private/users/lym/venv/bin/activate
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export WATERMARK_DEVICE="cuda:0"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

for SPEC in "$@"; do
  IFS=: read -r K START END TAG <<< "$SPEC"
  LOG="$LOG_DIR/gpu${GPU}_${TAG}.log"
  echo "[$(date '+%F %T')] START gpu=$GPU K=$K range=[$START,$END) tag=$TAG" | tee -a "$LOG"
  python -u scripts/framing.py \
    --model wavmark --K "$K" --n_trials 300 \
    --target_policy opportunistic --registry_size 1024 \
    --trial_start "$START" --trial_end "$END" --output_tag "$TAG" 2>&1 | tee -a "$LOG"
  echo "[$(date '+%F %T')] DONE gpu=$GPU K=$K range=[$START,$END) tag=$TAG" | tee -a "$LOG"
done

printf 'done %s\n' "$(date '+%F %T')" > "$STATUS_DIR/gpu${GPU}.done"

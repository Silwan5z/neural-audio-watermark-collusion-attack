#!/usr/bin/env bash
set -euo pipefail
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PY=/private/users/lym/venv/bin/python
SESSION=k8_constructed_native
OUT="$ROOT/data/k8_constructed_payload_case_native_20260829"
LOG="$ROOT/results/logs/k8_constructed_payload_case_native_20260829"
mkdir -p "$OUT" "$LOG"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session exists: $SESSION" >&2; exit 2; fi
models=(audioseal wavmark timbrewm voicemark wmcodec)
for gpu in 0 1 2 3 4; do
  model=${models[$gpu]}
  cmd="cd '$ROOT' && export PYTHONPATH='/tmp/k5case_deps' && export WATERMARK_DEVICE='cuda:$gpu' && export WAVMARK_WINDOW_BATCH_SIZE=600 && '$PY' scripts/run_k8_constructed_payload_case_native.py --model '$model' --output-dir '$OUT' > '$LOG/$model.log' 2>&1"
  if [[ $gpu -eq 0 ]]; then tmux new-session -d -s "$SESSION" -n "$model" "$cmd"
  else tmux new-window -t "$SESSION" -n "$model" "$cmd"; fi
done
echo "launched session=$SESSION output=$OUT logs=$LOG"

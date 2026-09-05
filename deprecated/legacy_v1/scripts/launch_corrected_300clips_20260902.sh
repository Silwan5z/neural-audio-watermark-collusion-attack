#!/usr/bin/env bash
set -euo pipefail

ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
RUN_DIR="$ROOT/runs/corrected_300clips_20260902"
mkdir -p "$RUN_DIR/logs"

for gpu in 0 1 2 3 4 5 6; do
  session="wm300clips_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "gpu=$gpu already running session=$session"
    continue
  fi
  tmux new-session -d -s "$session" \
    "echo '--- relaunch '$(date -u +%FT%TZ)' ---' >>'$RUN_DIR/logs/gpu${gpu}.log'; \
     '$ROOT/scripts/run_corrected_300clips_worker.sh' '$gpu' >>'$RUN_DIR/logs/gpu${gpu}.log' 2>&1"
  echo "gpu=$gpu session=$session log=$RUN_DIR/logs/gpu${gpu}.log"
done

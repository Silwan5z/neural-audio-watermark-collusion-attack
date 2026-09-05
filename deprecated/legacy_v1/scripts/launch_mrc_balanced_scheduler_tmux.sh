#!/usr/bin/env bash
set -euo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
SESSION="mrc_balanced_scheduler"
LOG="$ROOT/results/logs/mrc_balanced_scheduler/scheduler.log"
mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n scheduler \
  "cd '$ROOT' && exec /private/users/lym/venv/bin/python -u scripts/run_mrc_balanced_scheduler.py --gpus 0,1,2,3,4,5,6 2>&1 | tee -a '$LOG'"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
echo "launched session=$SESSION"

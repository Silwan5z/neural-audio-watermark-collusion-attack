#!/usr/bin/env bash
set -euo pipefail
SESSION=k5_mechanism_suite_20260828
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION"
    exit 0
fi
tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "bash scripts/wait_then_run_k5_mechanism_suite.sh"
echo "launched session=$SESSION"

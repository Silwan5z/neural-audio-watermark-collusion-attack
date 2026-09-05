#!/usr/bin/env bash
set -euo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
SESSION="mrc_ablation_7gpu"
WAIT_RUN_ID="20260823_7gpu_resume"
WORKER="$ROOT/scripts/run_mrc_ablation_queue_worker.sh"
VARIANTS=(beta5 beta10 no_keff_floor uniform_floor entropy005 hull_targets uniform_weights)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
for GPU in 0 1 2 3 4 5 6; do
  VARIANT="${VARIANTS[$GPU]}"
  if [[ "$GPU" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${GPU}_${VARIANT}" \
      "bash '$WORKER' '$GPU' '$VARIANT' '$WAIT_RUN_ID'"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${GPU}_${VARIANT}" \
      "bash '$WORKER' '$GPU' '$VARIANT' '$WAIT_RUN_ID'"
  fi
done
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
echo "launched session=$SESSION; workers wait for their TCT GPU handoff"
tmux list-windows -t "$SESSION"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
SESSION="tct_wav_fav1024_7gpu"
RUN_ID="20260823_7gpu_resume"
WORKER="$ROOT/scripts/run_favorable_n1024_wavmark_shard_worker.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n gpu0 \
  "bash '$WORKER' 0 '$RUN_ID' '2:250:290:resume7_g0_k2_250_290'"
tmux new-window -d -t "$SESSION" -n gpu1 \
  "bash '$WORKER' 1 '$RUN_ID' '2:290:300:resume7_g1_k2_290_300' '3:240:270:resume7_g1_k3_240_270'"
tmux new-window -d -t "$SESSION" -n gpu2 \
  "bash '$WORKER' 2 '$RUN_ID' '3:270:300:resume7_g2_k3_270_300' '5:220:230:resume7_g2_k5_220_230'"
tmux new-window -d -t "$SESSION" -n gpu3 \
  "bash '$WORKER' 3 '$RUN_ID' '5:230:270:resume7_g3_k5_230_270'"
tmux new-window -d -t "$SESSION" -n gpu4 \
  "bash '$WORKER' 4 '$RUN_ID' '5:270:300:resume7_g4_k5_270_300' '8:220:230:resume7_g4_k8_220_230'"
tmux new-window -d -t "$SESSION" -n gpu5 \
  "bash '$WORKER' 5 '$RUN_ID' '8:230:265:resume7_g5_k8_230_265'"
tmux new-window -d -t "$SESSION" -n gpu6 \
  "bash '$WORKER' 6 '$RUN_ID' '8:265:300:resume7_g6_k8_265_300'"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

echo "launched session=$SESSION run_id=$RUN_ID"
tmux list-windows -t "$SESSION"

#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/private/users/lym/venv/bin/python
LOG_DIR="$ROOT/results/logs/p0_p1"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/scripts/wavmark_arbitrary_tct.py" \
  --K 8 --n_trials 300 --trial_start 0 --trial_end 215 \
  >"$LOG_DIR/wavmark_detail_K8_shard_a.log" 2>&1 &
pid_a=$!

CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$ROOT/scripts/wavmark_arbitrary_tct.py" \
  --K 8 --n_trials 300 --trial_start 215 --trial_end 300 --output_tag shard_b \
  >"$LOG_DIR/wavmark_detail_K8_shard_b.log" 2>&1 &
pid_b=$!

status_a=0
status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?
if (( status_a != 0 || status_b != 0 )); then
  echo "K=8 shard failure: shard_a=$status_a shard_b=$status_b" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" scripts/merge_wavmark_arbitrary_shards.py --K 8 --n_trials 300 \
  results/evaluation/tamper_arbitrary_detail_wavmark_K8.csv \
  results/evaluation/tamper_arbitrary_detail_wavmark_K8.shard_b.csv
"$PYTHON" scripts/aggregate_wavmark_arbitrary.py

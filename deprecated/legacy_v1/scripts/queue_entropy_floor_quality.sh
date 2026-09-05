#!/usr/bin/env bash
set -euo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
VARIANT="entropy005_keff_floor"
LOG_ROOT="$ROOT/results/logs/${VARIANT}_quality"
MODELS=(audioseal wavmark timbrewm voicemark wmcodec)
KS=(2 3 5 8)
mkdir -p "$LOG_ROOT"

while true; do
  complete=0
  for model in "${MODELS[@]}"; do
    for k in "${KS[@]}"; do
      file="$ROOT/results/evaluation/mrc_ablation_${VARIANT}_N1024_${model}_K${k}.csv"
      if [[ -f "$file" ]]; then
        complete=$((complete + 1))
      fi
    done
  done
  printf '%s waiting_for_gpu_results complete=%d/20\n' "$(date --iso-8601=seconds)" "$complete"
  [[ "$complete" -eq 20 ]] && break
  sleep 30
done

/private/users/lym/venv/bin/python -u \
  "$ROOT/scripts/repair_mrc_solver_flags.py" --variant "$VARIANT" \
  > "$LOG_ROOT/solver_revalidation.log" 2>&1
rg -q 'SOLVER_REVALIDATION_COMPLETE' "$LOG_ROOT/solver_revalidation.log"

commands="$LOG_ROOT/commands.txt"
: > "$commands"
for model in "${MODELS[@]}"; do
  for k in "${KS[@]}"; do
    printf '%q ' /private/users/lym/venv/bin/python -u \
      "$ROOT/scripts/backfill_mrc_variant_quality.py" \
      --variant "$VARIANT" --model "$model" --K "$k" >> "$commands"
    printf '> %q 2>&1\n' "$LOG_ROOT/${model}_K${k}.log" >> "$commands"
  done
done

xargs -P 20 -I CMD bash -lc CMD < "$commands"
for model in "${MODELS[@]}"; do
  for k in "${KS[@]}"; do
    rg -q 'QUALITY_COMPLETE|quality already complete' "$LOG_ROOT/${model}_K${k}.log"
  done
done
date --iso-8601=seconds > "$LOG_ROOT/ALL_QUALITY_DONE"
printf '%s all_quality_done\n' "$(date --iso-8601=seconds)"

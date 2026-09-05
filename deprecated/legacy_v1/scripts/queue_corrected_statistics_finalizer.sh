#!/usr/bin/env bash
set -euo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
QUALITY_DONE="$ROOT/results/logs/entropy005_keff_floor_quality/ALL_QUALITY_DONE"
OUT="$ROOT/paper/identity_attribution_icassp_v8_source/supplementary_analysis_20260826_corrected"
mkdir -p "$OUT"

while [[ ! -f "$QUALITY_DONE" ]]; do
  printf '%s waiting_for_quality\n' "$(date --iso-8601=seconds)"
  sleep 30
done

CUDA_VISIBLE_DEVICES='' /private/users/lym/venv/bin/python -u \
  "$ROOT/scripts/finalize_corrected_statistics_20260826.py"
CUDA_VISIBLE_DEVICES='' /private/users/lym/venv/bin/python -u \
  "$ROOT/scripts/verify_corrected_statistics_20260826.py"
date --iso-8601=seconds > "$OUT/FINALIZATION_DONE"
printf '%s finalization_done\n' "$(date --iso-8601=seconds)"

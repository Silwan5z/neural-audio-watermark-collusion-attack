#!/usr/bin/env bash
set -euo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
RUN_ID="20260823_7gpu_resume"
STATUS_DIR="$ROOT/results/logs/favorable_n1024/$RUN_ID/status"
LOG="$ROOT/results/logs/favorable_n1024/$RUN_ID/finalize.log"
cd "$ROOT"

echo "[$(date '+%F %T')] waiting for seven successful workers" | tee -a "$LOG"
while true; do
  done_count=0
  for gpu in 0 1 2 3 4 5 6; do
    [[ -f "$STATUS_DIR/gpu${gpu}.done" ]] && done_count=$((done_count + 1))
  done
  echo "[$(date '+%F %T')] workers_done=$done_count/7" | tee -a "$LOG"
  [[ "$done_count" -eq 7 ]] && break
  sleep 30
done

source /private/users/lym/venv/bin/activate
python scripts/merge_favorable_n1024_wavmark_shards.py --K 2 \
  --tags resume7_g0_k2_250_290 resume7_g1_k2_290_300 | tee -a "$LOG"
python scripts/merge_favorable_n1024_wavmark_shards.py --K 3 \
  --tags resume7_g1_k3_240_270 resume7_g2_k3_270_300 | tee -a "$LOG"
python scripts/merge_favorable_n1024_wavmark_shards.py --K 5 \
  --tags resume7_g2_k5_220_230 resume7_g3_k5_230_270 resume7_g4_k5_270_300 | tee -a "$LOG"
python scripts/merge_favorable_n1024_wavmark_shards.py --K 8 \
  --tags resume7_g4_k8_220_230 resume7_g5_k8_230_265 resume7_g6_k8_265_300 | tee -a "$LOG"
python scripts/publish_favorable_n1024_wavmark.py | tee -a "$LOG"
python scripts/generate_comprehensive_comparison_report.py | tee -a "$LOG"
echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$LOG"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WAIT_PID=${1:?usage: wait_then_run_single_tamper_queue.sh PID}
LOG="$ROOT/logs/single_tamper_queue_coordinator.log"
mkdir -p "$ROOT/logs"

echo "$(date '+%F %T') waiting for WavMark native-v2 queue pid=${WAIT_PID}" | tee -a "$LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 2
done
echo "$(date '+%F %T') WavMark native-v2 queue ended; starting single-tamper queue" | tee -a "$LOG"
exec env GPUS=0,1,2,3,4,5,6 SHARD_SIZE=25 \
    bash "$ROOT/scripts/run_remaining_single_tamper_queue.sh"

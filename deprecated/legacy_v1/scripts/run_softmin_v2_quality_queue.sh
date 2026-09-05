#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/softmin_v2_quality_next_index"
LOCK_FILE="$ROOT/results/softmin_v2_quality.lock"
FAILED_FILE="$ROOT/results/softmin_v2_quality_failures.log"
LOG_DIR="$ROOT/logs/softmin_v2_quality"
N_WORKERS=${N_WORKERS:-8}
mkdir -p "$LOG_DIR"
cd "$ROOT"

TASKS=()
for model in audioseal wavmark voicemark wmcodec timbrewm; do
    for K in 2 3 5 8; do
        TASKS+=("$model|$K")
    done
done

printf '0\n' > "$STATE_FILE"
: > "$LOCK_FILE"
: > "$FAILED_FILE"

next_index() {
    local index
    exec 9>"$LOCK_FILE"
    flock -x 9
    index=$(<"$STATE_FILE")
    if (( index >= ${#TASKS[@]} )); then
        flock -u 9
        return 1
    fi
    printf '%s\n' "$((index + 1))" > "$STATE_FILE"
    flock -u 9
    printf '%s\n' "$index"
}

worker() {
    local worker_id=$1
    local index model K log status
    while true; do
        index=$(next_index) || break
        IFS='|' read -r model K <<< "${TASKS[$index]}"
        log="$LOG_DIR/worker${worker_id}_${model}_K${K}.log"
        echo "$(date '+%F %T') worker=${worker_id} START ${model} K=${K}" | tee -a "$log"
        if env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
               NUMEXPR_NUM_THREADS=1 \
           nice -n 10 "$PYTHON" -u scripts/backfill_softmin_v2_quality.py \
               --model "$model" --K "$K" >> "$log" 2>&1; then
            echo "$(date '+%F %T') worker=${worker_id} DONE ${model} K=${K}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') worker=${worker_id} FAILED(${status}) ${model} K=${K}" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for worker_id in $(seq 1 "$N_WORKERS"); do
    worker "$worker_id" &
done
wait

failures=$(wc -l < "$FAILED_FILE")
echo "all softmin-v2 quality workers finished; failures=${failures}"
(( failures == 0 ))

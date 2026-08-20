#!/usr/bin/env bash
# Seven-GPU resumable dispatcher for the arbitrary-target tampering control.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/arbitrary_tamper_next_index"
LOCK_FILE="$ROOT/results/arbitrary_tamper_queue.lock"
mkdir -p "$ROOT/results/evaluation" "$ROOT/results/logs"
cd "$ROOT"

TASKS=()
for model in audioseal wavmark voicemark wmcodec timbrewm; do
    for k in 2 3 5 8; do
        TASKS+=("$model|$k")
    done
done
[[ -f "$STATE_FILE" ]] || printf '0\n' > "$STATE_FILE"
: > "$LOCK_FILE"

next_task_index() {
    local idx
    exec 9>"$LOCK_FILE"
    flock -x 9
    idx=$(<"$STATE_FILE")
    if (( idx >= ${#TASKS[@]} )); then
        flock -u 9
        return 1
    fi
    printf '%s\n' "$((idx + 1))" > "$STATE_FILE"
    flock -u 9
    printf '%s\n' "$idx"
}

task_complete() {
    local model=$1 k=$2 output="results/evaluation/tamper_arbitrary_${1}_K${2}.csv"
    [[ -s "$output" ]] || return 1
    local rows=$(( $(wc -l < "$output") - 1 ))
    (( rows >= 6000 ))
}

run_worker() {
    local gpu=$1 idx model k log status
    while idx=$(next_task_index); do
        IFS='|' read -r model k <<< "${TASKS[$idx]}"
        log="results/logs/arbitrary_tamper_${model}_K${k}.log"
        if task_complete "$model" "$k"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${model} K=${k}" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} start ${model} K=${k}" | tee -a "$log"
        if WATERMARK_DEVICE="cuda:${gpu}" PYTHONPATH=src "$PYTHON" scripts/framing.py \
            --model "$model" --K "$k" --n_trials 300 --target_policy arbitrary >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} done ${model} K=${k}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${model} K=${k}" | tee -a "$log"
        fi
    done
}

for gpu in 0 1 2 3 4 5 6; do run_worker "$gpu" & done
wait

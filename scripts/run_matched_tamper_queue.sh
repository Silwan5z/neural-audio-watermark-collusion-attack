#!/usr/bin/env bash
# Seven-GPU resumable dispatcher for arbitrary-target TCT at matched N=1024.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/matched_tamper_next_index"
LOCK_FILE="$ROOT/results/matched_tamper_queue.lock"
FAILED_FILE="$ROOT/results/matched_tamper_failures.log"
mkdir -p "$ROOT/results/evaluation" "$ROOT/results/logs"
cd "$ROOT"

TASKS=()
for model in audioseal wavmark timbrewm voicemark wmcodec; do
    for k in 2 3 5 8; do
        TASKS+=("$model|$k")
    done
done
[[ -f "$STATE_FILE" ]] || printf '0\n' > "$STATE_FILE"
: > "$LOCK_FILE"
touch "$FAILED_FILE"

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

gpu_busy() {
    nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'
}

task_complete() {
    local output="results/evaluation/tamper_arbitrary_N1024_${1}_K${2}.csv"
    [[ -s "$output" ]] || return 1
    (( $(wc -l < "$output") - 1 >= 6000 ))
}

run_worker() {
    local gpu=$1 idx model k log status
    while true; do
        while gpu_busy "$gpu"; do sleep 10; done
        idx=$(next_task_index) || break
        IFS='|' read -r model k <<< "${TASKS[$idx]}"
        log="results/logs/matched_tamper_${model}_K${k}.log"
        if task_complete "$model" "$k"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${model} K=${k}" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} start ${model} K=${k}" | tee -a "$log"
        if WATERMARK_DEVICE="cuda:${gpu}" PYTHONPATH=src "$PYTHON" scripts/framing.py \
            --model "$model" --K "$k" --n_trials 300 --target_policy arbitrary \
            --registry_size 1024 >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} done ${model} K=${k}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${model} K=${k}" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in 0 1 2 3 4 5 6; do run_worker "$gpu" & done
wait

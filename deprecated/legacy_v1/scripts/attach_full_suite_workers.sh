#!/usr/bin/env bash
# Safely attach replacement workers to a legacy full-suite run.  It uses the
# original queue's state file and flock, and skips tasks already completed or
# currently owned by an existing process.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/full_suite_next_index"
LOCK_FILE="$ROOT/results/full_suite_queue.lock"
FAILED_FILE="$ROOT/results/full_suite_failures.log"
GPU_IDS=${GPU_IDS:-"0 2"}
cd "$ROOT"

declare -a TASKS=()
add_task() { TASKS+=("$1|$2|$3|$4|$5"); }
for model in audioseal wavmark voicemark wmcodec timbrewm; do
    for k in 2 3 5 8; do
        case "${model}:${k}" in
            audioseal:2|audioseal:3|wavmark:2|wavmark:3|voicemark:2|wmcodec:2|timbrewm:2) ;;
            *) add_task attack "$model" "$k" 300 attack ;;
        esac
        add_task attack_ecc "$model" "$k" 300 attack_ecc
        add_task rp "$model" "$k" 300 rp
        add_task eep "$model" "$k" 300 eep
        add_task baselines "$model" "$k" 300 baselines
        add_task blind_distance "$model" "$k" 300 dm
        add_task blind_minimax "$model" "$k" 300 bdb
        add_task pgr "$model" "$k" 300 pgr
        add_task framing "$model" "$k" 300 tamper
        add_task pulse_noise "$model" "$k" 50 pulse_noise
    done
done

next_task_index() {
    local idx
    exec 9>"$LOCK_FILE"
    flock -x 9
    idx=$(<"$STATE_FILE")
    printf '%s\n' "$((idx + 1))" > "$STATE_FILE"
    flock -u 9
    printf '%s\n' "$idx"
}

task_active() {
    local script=$1 model=$2 k=$3
    pgrep -f "scripts/${script}\.py --model ${model} --K ${k}" >/dev/null
}

run_worker() {
    local gpu=$1 idx spec script model k trials prefix output log status
    while true; do
        idx=$(next_task_index)
        (( idx < ${#TASKS[@]} )) || break
        IFS='|' read -r script model k trials prefix <<< "${TASKS[$idx]}"
        output="results/evaluation/${prefix}_${model}_K${k}.csv"
        log="results/logs/attach_${script}_${model}_K${k}.log"
        if [[ -s "$output" ]]; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${script} ${model} K=${k}" >> "$log"
            continue
        fi
        if task_active "$script" "$model" "$k"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-active ${script} ${model} K=${k}" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} start ${script} ${model} K=${k} n=${trials}" | tee -a "$log"
        if WATERMARK_DEVICE="cuda:${gpu}" PYTHONPATH=src "$PYTHON" "scripts/${script}.py" \
            --model "$model" --K "$k" --n_trials "$trials" >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} done ${script} ${model} K=${k}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${script} ${model} K=${k}" | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in $GPU_IDS; do run_worker "$gpu" & done
wait

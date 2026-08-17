#!/usr/bin/env bash
# Run the complete experiment matrix with seven persistent tmux workers.
# The seven main attack jobs already launched in watermark_collusion_300 are
# deliberately excluded here, so they are never duplicated.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
WORKERS=6  # GPU 0--5；GPU 6 留给其他工作

cd "$ROOT"
mkdir -p results/evaluation results/logs

# Wait for the currently running first wave to finish before taking the GPUs.
while tmux has-session -t watermark_collusion_300 2>/dev/null; do
    echo "$(date '+%F %T') waiting for first-wave attack jobs" >&2
    sleep 30
done

declare -a TASKS=()
models=(audioseal wavmark voicemark wmcodec timbrewm)
ks=(2 3 5 8)

add_task() {
    local script=$1 model=$2 k=$3 trials=$4 prefix=$5
    local output="results/evaluation/${prefix}_${model}_K${k}.csv"
    # A CSV is written only when a script completes, so it is safe to resume.
    [[ -s "$output" ]] && return
    TASKS+=("$script|$model|$k|$trials|$prefix")
}

for model in "${models[@]}"; do
    for k in "${ks[@]}"; do
        # Seven attack configurations are handled by the first tmux session.
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

STATE_FILE=results/full_suite_next_index
LOCK_FILE=results/full_suite_queue.lock
FAILED_FILE=results/full_suite_failures.log
printf '0\n' > "$STATE_FILE"
: > "$LOCK_FILE"
: > "$FAILED_FILE"

next_task_index() {
    local idx
    exec 9>"$LOCK_FILE"
    flock -x 9
    idx=$(<"$STATE_FILE")
    printf '%s\n' "$((idx + 1))" > "$STATE_FILE"
    flock -u 9
    printf '%s\n' "$idx"
}

run_worker() {
    local gpu=$1 idx script model k trials prefix log
    while true; do
        idx=$(next_task_index)
        (( idx < ${#TASKS[@]} )) || break
        IFS='|' read -r script model k trials prefix <<< "${TASKS[$idx]}"
        log="results/logs/full_${script}_${model}_K${k}.log"
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

for gpu in $(seq 0 $((WORKERS - 1))); do
    run_worker "$gpu" &
done
wait

"$PYTHON" scripts/minimax_framing.py > results/logs/minimax_framing.log 2>&1
echo "$(date '+%F %T') all full-suite experiment jobs completed"

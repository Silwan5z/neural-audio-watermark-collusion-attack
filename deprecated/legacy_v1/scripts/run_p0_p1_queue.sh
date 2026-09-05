#!/usr/bin/env bash
# Persistent seven-GPU queue for the registry, arbitrary-TCT-detail, temporal,
# and independent-codec controls requested after the main experiment suite.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/p0_p1_next_index"
LOCK_FILE="$ROOT/results/p0_p1_queue.lock"
FAILED_FILE="$ROOT/results/p0_p1_failures.log"
mkdir -p "$ROOT/results/evaluation" "$ROOT/results/logs"
cd "$ROOT"

# Registry jobs already completed or launched in dedicated tmux sessions are
# omitted; their cards join this queue as soon as those processes release them.
TASKS=()
for k in 5 8; do TASKS+=("registry|timbrewm|$k|300|registry_control_timbrewm_K${k}.csv|600"); done
for k in 2 3 5 8; do TASKS+=("registry|wavmark|$k|300|registry_control_wavmark_K${k}.csv|3000"); done
for k in 2 3 5 8; do TASKS+=("wavmark_detail|wavmark|$k|300|tamper_arbitrary_detail_wavmark_K${k}.csv|3000"); done
for model in audioseal wavmark timbrewm voicemark wmcodec; do
    TASKS+=("temporal|$model|5|100|temporal_sensitivity_${model}_K5.csv|1400")
done
for model in audioseal wavmark timbrewm voicemark wmcodec; do
    TASKS+=("codec|$model|5|300|codec_sensitivity_${model}_K5.csv|1800")
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
    local output=$1 expected=$2 rows
    [[ -s "results/evaluation/$output" ]] || return 1
    rows=$(( $(wc -l < "results/evaluation/$output") - 1 ))
    (( rows >= expected ))
}

run_task() {
    local kind=$1 model=$2 k=$3 trials=$4
    case "$kind" in
        registry)
            WATERMARK_DEVICE="$5" PYTHONPATH=src "$PYTHON" scripts/registry_size_control.py \
                --model "$model" --K "$k" --n_trials "$trials"
            ;;
        wavmark_detail)
            WATERMARK_DEVICE="$5" PYTHONPATH=src "$PYTHON" scripts/wavmark_arbitrary_tct.py \
                --K "$k" --n_trials "$trials"
            ;;
        temporal)
            WATERMARK_DEVICE="$5" PYTHONPATH=src "$PYTHON" scripts/temporal_sensitivity.py \
                --model "$model" --n_trials "$trials"
            ;;
        codec)
            WATERMARK_DEVICE="$5" PYTHONPATH=src "$PYTHON" scripts/codec_sensitivity.py \
                --model "$model" --n_trials "$trials"
            ;;
        *) return 2 ;;
    esac
}

run_worker() {
    local gpu=$1 idx spec kind model k trials output expected log status
    while true; do
        while gpu_busy "$gpu"; do sleep 10; done
        idx=$(next_task_index) || break
        IFS='|' read -r kind model k trials output expected <<< "${TASKS[$idx]}"
        log="results/logs/p0_p1_${kind}_${model}_K${k}.log"
        if task_complete "$output" "$expected"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${kind} ${model} K=${k}" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} start ${kind} ${model} K=${k} n=${trials}" | tee -a "$log"
        if run_task "$kind" "$model" "$k" "$trials" "cuda:${gpu}" >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} done ${kind} ${model} K=${k}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${kind} ${model} K=${k}" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in 0 1 2 3 4 5 6; do run_worker "$gpu" & done
wait

#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/v2_native_queue_next_index"
LOCK_FILE="$ROOT/results/v2_native_queue.lock"
FAILED_FILE="$ROOT/results/v2_native_queue_failures.log"
LOG_DIR="$ROOT/logs/v2_native_queue"
mkdir -p "$LOG_DIR" "$ROOT/results/evaluation"
cd "$ROOT"

TASKS=()
# Keep WavMark last because it is much slower than the other detectors.
for model in audioseal voicemark wmcodec wavmark; do
    for K in 2 3 5 8; do
        TASKS+=("$model|$K")
    done
done

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    mapfile -t GPU_LIST < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -F', ' '$2 < 1000 {print $1}'
    )
fi
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "No free GPU found" >&2
    exit 1
fi
echo "Using GPUs: ${GPU_LIST[*]}"

if [[ "${RESET_QUEUE:-0}" == 1 || ! -f "$STATE_FILE" ]]; then
    printf '0\n' > "$STATE_FILE"
fi
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

complete() {
    local path=$1
    [[ -s "$path" ]] || return 1
    (( $(wc -l < "$path") - 1 == 3000 ))
}

worker() {
    local gpu=$1 index model K output log status
    while true; do
        index=$(next_index) || break
        IFS='|' read -r model K <<< "${TASKS[$index]}"
        output="results/evaluation/margin_reachability_v2_native_${model}_K${K}.csv"
        log="$LOG_DIR/gpu${gpu}_${model}_K${K}.log"
        if complete "$output"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${model} K=${K}" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} START ${model} K=${K}" | tee -a "$log"
        if env WATERMARK_DEVICE="cuda:${gpu}" \
               PYTHONPATH="$ROOT/src:$ROOT/scripts" \
               OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
               NUMEXPR_NUM_THREADS=2 \
           nice -n 10 "$PYTHON" -u scripts/evaluate_margin_reachability_v2_native.py \
               --model "$model" --K "$K" >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} DONE ${model} K=${K}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${model} K=${K}" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done
wait

echo "all v2 native workers finished; failures=$(wc -l < "$FAILED_FILE")"
[[ $(wc -l < "$FAILED_FILE") -eq 0 ]]

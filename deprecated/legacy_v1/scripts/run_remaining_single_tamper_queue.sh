#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/single_tamper_shard_next_index"
LOCK_FILE="$ROOT/results/single_tamper_shard.lock"
FAILED_FILE="$ROOT/results/single_tamper_shard_failures.log"
LOG_DIR="$ROOT/logs/single_tamper_shards"
SHARD_SIZE=${SHARD_SIZE:-25}
mkdir -p "$LOG_DIR" "$ROOT/results/evaluation"
cd "$ROOT"

if (( SHARD_SIZE < 1 )); then
    echo "SHARD_SIZE must be positive" >&2
    exit 2
fi

TASKS=()
add_tasks() {
    local attack=$1 model=$2 start=0 end
    while (( start < 300 )); do
        end=$((start + SHARD_SIZE))
        (( end > 300 )) && end=300
        TASKS+=("$attack|$model|$start|$end")
        start=$end
    done
}

# Put the known slow families first so their tails do not serialize the queue.
add_tasks overwrite wavmark
for model in audioseal voicemark wmcodec timbrewm; do
    add_tasks ifgsm "$model"
done
for model in audioseal voicemark wmcodec timbrewm; do
    add_tasks overwrite "$model"
done

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    mapfile -t GPU_LIST < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
            awk -F', ' '$2 < 1000 {print $1}'
    )
fi
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "No free GPU found" >&2
    exit 1
fi

printf '0\n' > "$STATE_FILE"
: > "$LOCK_FILE"
: > "$FAILED_FILE"
echo "Using GPUs: ${GPU_LIST[*]}; tasks=${#TASKS[@]}; shard_size=${SHARD_SIZE}"

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
    local gpu=$1
    local index attack model start end tag output rows_per_trial expected log status
    while true; do
        index=$(next_index) || break
        IFS='|' read -r attack model start end <<< "${TASKS[$index]}"
        tag="shard_${start}_${end}"
        output="results/evaluation/single_tamper_${attack}_${model}.${tag}.csv"
        rows_per_trial=20
        [[ "$model" == timbrewm ]] && rows_per_trial=10
        expected=$(((end - start) * rows_per_trial))
        log="$LOG_DIR/gpu${gpu}_${attack}_${model}_${start}_${end}.log"
        if [[ -s "$output" ]] && (( $(wc -l < "$output") - 1 == expected )); then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${attack}/${model} [${start},${end})" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} START ${attack}/${model} [${start},${end})" | tee -a "$log"
        if env WATERMARK_DEVICE="cuda:${gpu}" WAVMARK_WINDOW_BATCH_SIZE=400 \
               PYTHONPATH="$ROOT/src:$ROOT/scripts" \
               OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
               NUMEXPR_NUM_THREADS=2 \
           nice -n 10 "$PYTHON" -u scripts/run_single_tamper.py \
               --attack "$attack" --model "$model" --n_trials 300 \
               --registry_sizes 1024,native --trial_start "$start" --trial_end "$end" \
               --output_tag "$tag" --steps 10 >> "$log" 2>&1; then
            if [[ ! -s "$output" ]] || (( $(wc -l < "$output") - 1 != expected )); then
                echo "$(date '+%F %T') gpu=${gpu} INVALID ${attack}/${model} [${start},${end})" \
                    | tee -a "$log" "$FAILED_FILE"
            else
                echo "$(date '+%F %T') gpu=${gpu} DONE ${attack}/${model} [${start},${end})" | tee -a "$log"
            fi
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${attack}/${model} [${start},${end})" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done
wait

failures=$(wc -l < "$FAILED_FILE")
echo "all single-tamper shard workers finished; failures=${failures}"
if (( failures != 0 )); then
    exit 1
fi
"$PYTHON" scripts/merge_single_tamper_shards.py
bash scripts/run_softmin_v2_quality_queue.sh
"$PYTHON" scripts/validate_final_attack_tamper_outputs.py
"$PYTHON" scripts/publish_results_to_data.py

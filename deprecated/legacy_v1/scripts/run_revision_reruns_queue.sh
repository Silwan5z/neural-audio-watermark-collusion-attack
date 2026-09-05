#!/usr/bin/env bash
# Seven persistent workers for the corrected nested-registry rerun and the
# d_hull/R_bit rank-fusion target experiment.  One worker is placed on each GPU
# so it time-shares with, but never stops, the pre-existing training process.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/revision_reruns_next_index"
LOCK_FILE="$ROOT/results/revision_reruns_queue.lock"
FAILED_FILE="$ROOT/results/revision_reruns_failures.log"
LOG_DIR="$ROOT/results/logs/revision_reruns"
mkdir -p "$ROOT/results/evaluation" "$LOG_DIR"
cd "$ROOT"

TASKS=()
add_task() { TASKS+=("$1|$2|$3|$4|$5|$6|$7"); }

# WavMark is the slow path: four 75-trial shards per K.
for k in 2 3 5 8; do
    for shard in 0 1 2 3; do
        start=$((shard * 75)); end=$((start + 75)); tag="shard${shard}of4"
        add_task registry wavmark "$k" "$start" "$end" "$tag" 10
    done
done

# Other 16-bit systems: two 150-trial shards per cell.  TimbreWM has only one
# native N=1024 registry and therefore has no nested size sweep to correct.
for model in audioseal voicemark wmcodec; do
    for k in 2 3 5 8; do
        add_task registry "$model" "$k" 0 150 shard0of2 10
        add_task registry "$model" "$k" 150 300 shard1of2 10
    done
done

# New target selector: emit d_hull-only, R_bit-only and equal-weight rank fusion.
for k in 2 3 5 8; do
    for shard in 0 1 2 3; do
        start=$((shard * 75)); end=$((start + 75)); tag="shard${shard}of4"
        add_task joint wavmark "$k" "$start" "$end" "$tag" 3
    done
done
for model in audioseal timbrewm voicemark wmcodec; do
    for k in 2 3 5 8; do
        add_task joint "$model" "$k" 0 150 shard0of2 3
        add_task joint "$model" "$k" 150 300 shard1of2 3
    done
done

if [[ "${RESET_QUEUE:-0}" == 1 || ! -f "$STATE_FILE" ]]; then
    printf '0\n' > "$STATE_FILE"
fi
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

task_output() {
    local kind=$1 model=$2 k=$3 tag=$4
    case "$kind" in
        registry) printf 'results/evaluation/registry_control_nested_%s_K%s.%s.csv\n' "$model" "$k" "$tag" ;;
        joint) printf 'results/evaluation/joint_reachability_target_%s_K%s.%s.csv\n' "$model" "$k" "$tag" ;;
    esac
}

task_complete() {
    local path=$1 expected=$2 rows
    [[ -s "$path" ]] || return 1
    rows=$(( $(wc -l < "$path") - 1 ))
    (( rows == expected ))
}

run_task() {
    local kind=$1 model=$2 k=$3 start=$4 end=$5 tag=$6 device=$7
    local common_env=(env WATERMARK_DEVICE="$device" WAVMARK_WINDOW_BATCH_SIZE=200
        CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
        OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src:scripts)
    case "$kind" in
        registry)
            nice -n 10 "${common_env[@]}" "$PYTHON" scripts/registry_size_control.py \
                --model "$model" --K "$k" --n_trials 300 \
                --trial_start "$start" --trial_end "$end" \
                --output_prefix registry_control_nested --output_tag "$tag"
            ;;
        joint)
            nice -n 10 "${common_env[@]}" "$PYTHON" scripts/joint_reachability_target.py \
                --model "$model" --K "$k" --n_trials 300 --candidate_pool 2000 \
                --trial_start "$start" --trial_end "$end" --output_tag "$tag"
            ;;
        *) return 2 ;;
    esac
}

run_worker() {
    local gpu=$1 idx kind model k start end tag rows_per_trial output expected log status
    while true; do
        idx=$(next_task_index) || break
        IFS='|' read -r kind model k start end tag rows_per_trial <<< "${TASKS[$idx]}"
        output=$(task_output "$kind" "$model" "$k" "$tag")
        expected=$(((end - start) * rows_per_trial))
        log="$LOG_DIR/gpu${gpu}_${kind}_${model}_K${k}_${tag}.log"
        if task_complete "$output" "$expected"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete $kind $model K=$k $tag" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} start $kind $model K=$k range=[$start,$end)" | tee -a "$log"
        if run_task "$kind" "$model" "$k" "$start" "$end" "$tag" "cuda:${gpu}" >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} done $kind $model K=$k $tag" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED($status) $kind $model K=$k $tag" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in 0 1 2 3 4 5 6; do run_worker "$gpu" & done
wait

merge_registry() {
    local model=$1 k=$2 nshard=$3 rows_per_trial=10 inputs=() shard
    for ((shard=0; shard<nshard; shard++)); do
        inputs+=("results/evaluation/registry_control_nested_${model}_K${k}.shard${shard}of${nshard}.csv")
    done
    "$PYTHON" scripts/merge_trial_shards.py \
        --output "results/evaluation/registry_control_nested_${model}_K${k}.csv" \
        --expected_trials 300 --rows_per_trial "$rows_per_trial" "${inputs[@]}"
}

merge_joint() {
    local model=$1 k=$2 nshard=$3 inputs=() shard
    for ((shard=0; shard<nshard; shard++)); do
        inputs+=("results/evaluation/joint_reachability_target_${model}_K${k}.shard${shard}of${nshard}.csv")
    done
    "$PYTHON" scripts/merge_trial_shards.py \
        --output "results/evaluation/joint_reachability_target_${model}_K${k}.csv" \
        --expected_trials 300 --rows_per_trial 3 "${inputs[@]}"
}

for model in audioseal voicemark wmcodec; do
    for k in 2 3 5 8; do merge_registry "$model" "$k" 2; done
done
for k in 2 3 5 8; do merge_registry wavmark "$k" 4; done
for model in audioseal timbrewm voicemark wmcodec; do
    for k in 2 3 5 8; do merge_joint "$model" "$k" 2; done
done
for k in 2 3 5 8; do merge_joint wavmark "$k" 4; done

echo "$(date '+%F %T') revision reruns and shard merges completed"

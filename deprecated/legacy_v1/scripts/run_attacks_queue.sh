#!/usr/bin/env bash
# GPU queue for corrected single-copy evasion baselines (no coalition K).
# Native and matched-N=1024 attribution are evaluated from the same attacked
# waveform by run_attacks.py, so no second registry-size queue is needed.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/single_evasion_queue_next_index"
LOCK_FILE="$ROOT/results/single_evasion_queue.lock"
FAILED_FILE="$ROOT/results/single_evasion_queue_failures.log"
LOG_DIR="$ROOT/logs/single_evasion_queue"
mkdir -p "$ROOT/results/evaluation" "$LOG_DIR"
cd "$ROOT"

TASKS=()
add() { TASKS+=("$1|$2|$3|$4"); }

# Finish the inexpensive attacks first. HSJA tasks are deliberately last so
# they cannot delay the other baselines while occupying a GPU for many queries.
for attack in overwrite_same overwrite_cross encodec vocoder; do
    for model in audioseal wavmark voicemark wmcodec timbrewm; do
        extra=""
        [[ "$attack" == encodec ]] && extra="--bandwidth 24.0"
        add "$attack" "$model" 300 "$extra"
    done
done
for model in audioseal voicemark wmcodec timbrewm; do
    add fgsm "$model" 300 "--fgsm_epsilon 0.01"
done

detect_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' '$2 < 1000 {print $1}'
}

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    mapfile -t GPU_LIST < <(detect_free_gpus)
fi
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "No free GPUs found. Set GPUS=0,1,... to override." >&2
    exit 1
fi
echo "Using GPUs: ${GPU_LIST[*]}"

if [[ "${RESET_QUEUE:-0}" == 1 || ! -f "$STATE_FILE" ]]; then
    printf '0\n' > "$STATE_FILE"
fi
: > "$LOCK_FILE"
: > "$FAILED_FILE"

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

cross_model_for() {
    case "$1" in
        audioseal) printf 'wavmark\n' ;;
        wavmark) printf 'voicemark\n' ;;
        voicemark) printf 'wmcodec\n' ;;
        wmcodec) printf 'timbrewm\n' ;;
        timbrewm) printf 'audioseal\n' ;;
    esac
}

task_output() {
    local attack=$1 model=$2
    if [[ "$attack" == overwrite_cross ]]; then
        printf 'results/evaluation/attack_evasion_%s_%s__to_%s.csv\n' \
            "$attack" "$model" "$(cross_model_for "$model")"
    else
        printf 'results/evaluation/attack_evasion_%s_%s.csv\n' "$attack" "$model"
    fi
}

expected_rows() {
    local model=$1 n_trials=$2
    if [[ "$model" == timbrewm ]]; then
        printf '%s\n' "$n_trials"
    else
        printf '%s\n' "$((2 * n_trials))"
    fi
}

task_complete() {
    local path=$1 expected=$2
    [[ -s "$path" ]] || return 1
    local rows=$(( $(wc -l < "$path") - 1 ))
    (( rows == expected ))
}

run_worker() {
    local gpu=$1
    local idx attack model n_trials extra output expected log status
    while true; do
        idx=$(next_task_index) || break
        IFS='|' read -r attack model n_trials extra <<< "${TASKS[$idx]}"
        output=$(task_output "$attack" "$model")
        expected=$(expected_rows "$model" "$n_trials")
        log="$LOG_DIR/gpu${gpu}_${attack}_${model}.log"

        if task_complete "$output" "$expected"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${attack} ${model}" >> "$log"
            continue
        fi

        echo "$(date '+%F %T') gpu=${gpu} START ${attack} ${model} [${extra}]" | tee -a "$log"
        if env WATERMARK_DEVICE="cuda:${gpu}" \
               PYTHONPATH="$ROOT/src:$ROOT/scripts" \
               OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
               NUMEXPR_NUM_THREADS=2 \
           nice -n 10 "$PYTHON" -u scripts/run_attacks.py \
               --attack "$attack" --model "$model" --n_trials "$n_trials" \
               --registry_sizes 1024,native $extra >> "$log" 2>&1; then
            echo "$(date '+%F %T') gpu=${gpu} DONE ${attack} ${model}" | tee -a "$log"
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) ${attack} ${model}" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
    echo "$(date '+%F %T') gpu=${gpu} queue exhausted"
}

for gpu in "${GPU_LIST[@]}"; do
    run_worker "$gpu" &
done
wait

echo "$(date '+%F %T') all workers done; failures=$(wc -l < "$FAILED_FILE")"
[[ $(wc -l < "$FAILED_FILE") -eq 0 ]]

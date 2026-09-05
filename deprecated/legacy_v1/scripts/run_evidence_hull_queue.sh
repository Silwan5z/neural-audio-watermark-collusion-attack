#!/usr/bin/env bash
# Persistent backfill queue for the migrated evidence-chain experiments.
# It deliberately has its own state/lock files: sharing the legacy suite's
# numeric index would corrupt that queue because its task manifest differs.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5"}  # GPU 6 remains reserved.
STATE_FILE="$ROOT/results/evidence_hull_next_index"
LOCK_FILE="$ROOT/results/evidence_hull_queue.lock"
FAILED_FILE="$ROOT/results/evidence_hull_failures.log"
mkdir -p "$ROOT/results/evaluation" "$ROOT/results/logs"
cd "$ROOT"

declare -a TASKS=()
for model in audioseal wavmark voicemark wmcodec timbrewm; do
    for k in 2 3 5 8; do
        TASKS+=("evidence_chain|$model|$k|300|evidence_chain|9")
        TASKS+=("framing_hull|$model|$k|300|framing_hull|20")
    done
done
# Diagnostic pilots run after the evidence/hull matrix.  Their final CSVs are
# compact summaries/contexts rather than rows per trial, hence the explicit
# expected-row field below.
for model in audioseal wavmark voicemark wmcodec timbrewm; do
    for k in 5 8; do
        TASKS+=("dm_restart_stability|$model|$k|20|dm_restart_stability|0")
    done
done
for model in voicemark wmcodec; do
    for k in 5 8; do
        TASKS+=("detector_oracle|$model|$k|40|detector_oracle|0")
    done
done

# Preserve progress when this dispatcher is restarted; create only once.
[[ -f "$STATE_FILE" ]] || printf '0\n' > "$STATE_FILE"
: > "$LOCK_FILE"
touch "$FAILED_FILE"

next_task_index() {
    local idx
    exec 9>"$LOCK_FILE"
    flock -x 9
    idx=$(<"$STATE_FILE")
    # Do not advance past the current manifest.  This queue is intentionally
    # persistent and may receive new tasks after workers have drained it; an
    # out-of-range claim would otherwise silently skip those appended tasks.
    if (( idx >= ${#TASKS[@]} )); then
        flock -u 9
        return 1
    fi
    printf '%s\n' "$((idx + 1))" > "$STATE_FILE"
    flock -u 9
    printf '%s\n' "$idx"
}

task_complete() {
    local prefix=$1 model=$2 k=$3 trials=$4 rows_per_trial=$5 expected_rows
    local output="results/evaluation/${prefix}_${model}_K${k}.csv"
    [[ -s "$output" ]] || return 1
    expected_rows=$(( trials * rows_per_trial ))
    # Summary-based diagnostic outputs intentionally contain one final row.
    [[ "$prefix" == "dm_restart_stability" ]] && expected_rows=1
    [[ "$prefix" == "detector_oracle" ]] && expected_rows="$trials"
    local rows=$(( $(wc -l < "$output") - 1 ))
    (( rows >= expected_rows ))
}

task_active() {
    local script=$1 model=$2 k=$3
    pgrep -f "scripts/${script}\.py --model ${model} --K ${k}" >/dev/null
}

gpu_busy() {
    # Any compute process means this card is still owned by a running task.
    nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'
}

run_worker() {
    local gpu=$1 idx spec script model k trials prefix rows_per_trial log status
    while true; do
        while gpu_busy "$gpu"; do sleep 15; done
        idx=$(next_task_index) || break
        IFS='|' read -r script model k trials prefix rows_per_trial <<< "${TASKS[$idx]}"
        log="results/logs/evidence_hull_${script}_${model}_K${k}.log"
        if task_complete "$prefix" "$model" "$k" "$trials" "$rows_per_trial"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete ${script} ${model} K=${k}" >> "$log"
            continue
        fi
        if task_active "$script" "$model" "$k"; then
            echo "$(date '+%F %T') gpu=${gpu} skip-active ${script} ${model} K=${k}" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} start ${script} ${model} K=${k}" | tee -a "$log"
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

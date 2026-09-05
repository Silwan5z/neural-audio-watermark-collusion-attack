#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
STATE_FILE="$ROOT/results/wavmark_v2_native_shard_next_index"
LOCK_FILE="$ROOT/results/wavmark_v2_native_shard.lock"
FAILED_FILE="$ROOT/results/wavmark_v2_native_shard_failures.log"
LOG_DIR="$ROOT/logs/wavmark_v2_native_shards"
SHARD_SIZE=${SHARD_SIZE:-42}
mkdir -p "$LOG_DIR" "$ROOT/results/evaluation"
cd "$ROOT"

if (( SHARD_SIZE < 1 )); then
    echo "SHARD_SIZE must be positive" >&2
    exit 2
fi

TASKS=()
for K in 2 3 5 8; do
    prefix="results/evaluation/margin_reachability_v2_native_wavmark_K${K}.partial.csv"
    start=$(
        "$PYTHON" - "$prefix" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
counts = Counter()
if path.exists():
    with path.open(newline="") as handle:
        counts.update(int(row["gi"]) for row in csv.DictReader(handle))
start = 0
while counts[start] == 10:
    start += 1
print(start)
PY
    )
    while (( start < 300 )); do
        end=$((start + SHARD_SIZE))
        (( end > 300 )) && end=300
        TASKS+=("$K|$start|$end")
        start=$end
    done
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
    local index K start end tag output expected log status
    while true; do
        index=$(next_index) || break
        IFS='|' read -r K start end <<< "${TASKS[$index]}"
        tag="shard_${start}_${end}"
        output="results/evaluation/margin_reachability_v2_native_wavmark_K${K}.${tag}.csv"
        expected=$(((end - start) * 10))
        log="$LOG_DIR/gpu${gpu}_K${K}_${start}_${end}.log"
        if [[ -s "$output" ]] && (( $(wc -l < "$output") - 1 == expected )); then
            echo "$(date '+%F %T') gpu=${gpu} skip-complete K=${K} [${start},${end})" >> "$log"
            continue
        fi
        echo "$(date '+%F %T') gpu=${gpu} START K=${K} [${start},${end})" | tee -a "$log"
        if env WATERMARK_DEVICE="cuda:${gpu}" WAVMARK_WINDOW_BATCH_SIZE=400 \
               PYTHONPATH="$ROOT/src:$ROOT/scripts" \
               OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
               NUMEXPR_NUM_THREADS=2 \
           nice -n 10 "$PYTHON" -u scripts/evaluate_margin_reachability_v2_native.py \
               --model wavmark --K "$K" --trial_start "$start" --trial_end "$end" \
               --output_tag "$tag" >> "$log" 2>&1; then
            if [[ ! -s "$output" ]] || (( $(wc -l < "$output") - 1 != expected )); then
                echo "$(date '+%F %T') gpu=${gpu} INVALID K=${K} [${start},${end})" \
                    | tee -a "$log" "$FAILED_FILE"
            else
                echo "$(date '+%F %T') gpu=${gpu} DONE K=${K} [${start},${end})" | tee -a "$log"
            fi
        else
            status=$?
            echo "$(date '+%F %T') gpu=${gpu} FAILED(${status}) K=${K} [${start},${end})" \
                | tee -a "$log" "$FAILED_FILE"
        fi
    done
}

for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done
wait

failures=$(wc -l < "$FAILED_FILE")
echo "all WavMark shard workers finished; failures=${failures}"
if (( failures != 0 )); then
    exit 1
fi
"$PYTHON" scripts/merge_wavmark_v2_native_shards.py

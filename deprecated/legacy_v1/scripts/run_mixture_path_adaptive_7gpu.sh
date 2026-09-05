#!/usr/bin/env bash
set -uo pipefail

ROOT="/private/users/lym/neural-audio-watermark-collusion-attack"
PY="/private/users/lym/venv/bin/python"
OUT="$ROOT/data/mixture_path_k5_adaptive_20260829"
LOG="$ROOT/results/logs/mixture_path_k5_adaptive_20260829"

mkdir -p "$OUT/shards" "$LOG"
cd "$ROOT" || exit 1

run_worker() {
    local gpu="$1"
    local model
    for model in audioseal voicemark; do
        echo "[$(date --iso-8601=seconds)] gpu=$gpu start model=$model"
        CUDA_VISIBLE_DEVICES="$gpu" WATERMARK_DEVICE="cuda:0" PYTHONUNBUFFERED=1 \
            "$PY" scripts/run_mixture_path_adaptive.py \
            --model "$model" --shard-id "$gpu" --num-shards 7 \
            --n-trials 300 --checkpoint-every 2 --output-dir "$OUT" \
            >"$LOG/gpu${gpu}_${model}.log" 2>&1
        local code=$?
        echo "[$(date --iso-8601=seconds)] gpu=$gpu exit=$code model=$model"
        if [[ "$code" -ne 0 ]]; then
            return "$code"
        fi
    done
}

echo "[$(date --iso-8601=seconds)] launch adaptive mixture path on GPUs 0-6"
pids=()
for gpu in 0 1 2 3 4 5 6; do
    run_worker "$gpu" >>"$LOG/master.log" 2>&1 &
    pids+=("$!")
    echo "$gpu $!" >>"$LOG/worker_pids.log"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[$(date --iso-8601=seconds)] one or more workers failed; finalizer not run" >>"$LOG/master.log"
    exit 1
fi

"$PY" scripts/finalize_mixture_path_adaptive.py \
    --input-dir "$OUT" --n-trials 300 --num-shards 7 \
    >"$LOG/finalizer.log" 2>&1
code=$?
echo "[$(date --iso-8601=seconds)] finalizer exit=$code" >>"$LOG/master.log"
exit "$code"

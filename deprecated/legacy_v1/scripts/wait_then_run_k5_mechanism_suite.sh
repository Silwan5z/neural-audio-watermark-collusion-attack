#!/usr/bin/env bash
set -euo pipefail

ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PYTHON=/private/users/lym/venv/bin/python
ABL_OUT="$ROOT/results/frequency_temporal_k5_20260828"
ONEBIT_OUT="$ROOT/results/one_bit_k5_20260828"
LOG="$ROOT/results/logs/k5_mechanism_suite_20260828"
mkdir -p "$ABL_OUT" "$ONEBIT_OUT" "$LOG"

all_gpus_free() {
    local gpu pids mem
    for gpu in 0 1 2 3 4 5 6; do
        pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')
        mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if [[ -n "$pids" || "$mem" -ge 1024 ]]; then
            return 1
        fi
    done
    return 0
}

echo "[$(date --iso-8601=seconds)] waiting for MERT/all GPUs" >> "$LOG/master.log"
stable=0
while [[ "$stable" -lt 2 ]]; do
    if all_gpus_free; then
        stable=$((stable + 1))
    else
        stable=0
    fi
    sleep 30
done
echo "[$(date --iso-8601=seconds)] all seven GPUs free" >> "$LOG/master.log"
while [[ ! -f "$ABL_OUT/.precompute_complete" ]]; do
    echo "[$(date --iso-8601=seconds)] GPUs free; waiting for CPU precompute sentinel" >> "$LOG/master.log"
    sleep 30
done
echo "[$(date --iso-8601=seconds)] CPU precompute sentinel verified" >> "$LOG/master.log"

run_ablation_phase() {
    local pids=() failed=0 gpu
    for gpu in 0 1 2 3 4 5 6; do
        (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu" WATERMARK_DEVICE=cuda:0 PYTHONUNBUFFERED=1 \
                "$PYTHON" scripts/run_frequency_temporal_k5_shard.py \
                --shard-id "$gpu" --num-shards 7 --n-trials 300 \
                --checkpoint-every 5 --output-dir "$ABL_OUT"
        ) > "$LOG/ablation_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    [[ "$failed" -eq 0 ]]
}

run_onebit_model_phase() {
    local model="$1" pids=() failed=0 gpu start end
    for gpu in 0 1 2 3 4 5 6; do
        start=$((300 * gpu / 7))
        end=$((300 * (gpu + 1) / 7))
        (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu" WATERMARK_DEVICE=cuda:0 PYTHONUNBUFFERED=1 \
                "$PYTHON" scripts/run_onebit_k5_pair_analysis.py \
                --model "$model" --trial-start "$start" --trial-end "$end" \
                --checkpoint-every 10 --tag "shard${gpu}of7" --output-dir "$ONEBIT_OUT"
        ) > "$LOG/onebit_${model}_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    [[ "$failed" -eq 0 ]]
}

echo "[$(date --iso-8601=seconds)] phase 1: seven-shard frequency/temporal" >> "$LOG/master.log"
run_ablation_phase
echo "[$(date --iso-8601=seconds)] frequency/temporal GPU phase complete" >> "$LOG/master.log"

# Quality evaluation is CPU-only and overlaps the following GPU phases.
(
    cd "$ROOT"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$PYTHON" scripts/finalize_frequency_temporal_k5.py \
        --input-dir "$ABL_OUT" --data-dir "$ROOT/data/frequency_temporal_k5_20260828" \
        --quality-workers 16 --bootstrap 10000
) > "$LOG/ablation_finalizer.log" 2>&1 &
finalizer_pid=$!

echo "[$(date --iso-8601=seconds)] phase 2: one-bit AudioSeal seven shards" >> "$LOG/master.log"
run_onebit_model_phase audioseal
echo "[$(date --iso-8601=seconds)] phase 3: one-bit VoiceMark seven shards" >> "$LOG/master.log"
run_onebit_model_phase voicemark

cd "$ROOT"
"$PYTHON" scripts/merge_onebit_k5_shards.py > "$LOG/onebit_merge.log" 2>&1
"$PYTHON" scripts/summarize_onebit_k5_pair_analysis.py \
    --input-dir "$ONEBIT_OUT" --output-dir "$ONEBIT_OUT/summary" \
    --bootstrap 10000 --seed 20260828 > "$LOG/onebit_summary.log" 2>&1

wait "$finalizer_pid"
echo "[$(date --iso-8601=seconds)] all K=5 mechanism experiments complete" >> "$LOG/master.log"

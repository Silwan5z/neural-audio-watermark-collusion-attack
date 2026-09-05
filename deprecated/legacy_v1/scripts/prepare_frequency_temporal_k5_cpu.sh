#!/usr/bin/env bash
set -euo pipefail
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PYTHON=/private/users/lym/venv/bin/python
OUT="$ROOT/results/frequency_temporal_k5_20260828"
LOG="$ROOT/results/logs/k5_mechanism_suite_20260828"
mkdir -p "$OUT" "$LOG"

pids=()
failed=0
for shard in 0 1 2 3 4 5 6; do
    (
        cd "$ROOT"
        env CUDA_VISIBLE_DEVICES= WATERMARK_DEVICE=cpu OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
            "$PYTHON" scripts/run_frequency_temporal_k5_shard.py \
            --shard-id "$shard" --num-shards 7 --n-trials 300 \
            --prepare-only --output-dir "$OUT"
    ) > "$LOG/precompute_cpu_shard${shard}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "CPU precompute failed" >&2
    exit 1
fi
json_count=$(find "$OUT/precomputed" -type f -name 'trial_*.json' | wc -l)
wav_count=$(find "$OUT/audio" -type f -name '*.wav' | wc -l)
mask_count=$(find "$OUT/masks" -type f -name 'trial_*.npz' | wc -l)
if [[ "$json_count" -ne 1500 || "$wav_count" -ne 12000 || "$mask_count" -ne 300 ]]; then
    echo "CPU precompute cardinality failure: json=$json_count wav=$wav_count mask=$mask_count" >&2
    exit 1
fi
touch "$OUT/.precompute_complete"
echo "[$(date --iso-8601=seconds)] CPU precompute complete" >> "$LOG/master.log"

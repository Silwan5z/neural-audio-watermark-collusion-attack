#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?GPU id required}
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PY=/private/users/lym/venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="/tmp/k5case_deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
trap 'rc=$?; echo "worker_exit gpu='$GPU_ID' rc=$rc utc=$(date -u +%FT%TZ)"; exit $rc' EXIT

ATTACK_OUT="$ROOT/results/evaluation_300clips_20260902"
K8_OUT="$ROOT/data/k8_population_native_300clips_20260902"
K8_VALID_OUT="$ROOT/data/k8_population_source_correct_300clips_20260902/raw"
ONEBIT_OUT="$ROOT/results/one_bit_k5_300clips_20260902"
PATH_OUT="$ROOT/data/mixture_path_k5_300clips_20260902"
REGION_OUT="$ROOT/results/frequency_temporal_k5_300clips_20260902"
mkdir -p "$ATTACK_OUT" "$K8_OUT" "$K8_VALID_OUT" "$ONEBIT_OUT" "$PATH_OUT" "$REGION_OUT"

run_attack_series() {
  local model=$1
  run_attack "$model" 2
  run_attack "$model" 3
  run_attack "$model" 5
}

run_attack() {
  local model=$1
  local k=$2
  local output="$ATTACK_OUT/attack_${model}_K${k}.csv"
  if [[ -f "$output" ]] && [[ "$(wc -l < "$output")" -eq 301 ]]; then
    echo "skip_complete model=$model K=$k output=$output"
    return
  fi
  "$PY" scripts/attack.py --model "$model" --K "$k" --n_trials 300 --mean-only --output-dir "$ATTACK_OUT"
}

run_k8() {
  local model=$1
  "$PY" scripts/run_k8_population_native.py --model "$model" --n-trials 300 --output-dir "$K8_OUT"
  "$PY" scripts/complete_k8_source_correct_population.py --model "$model" \
    --source-dir "$K8_OUT/raw" --dest-dir "$K8_VALID_OUT"
}

run_onebit_path() {
  local model=$1
  "$PY" scripts/run_onebit_k5_pair_analysis.py --model "$model" --output-dir "$ONEBIT_OUT"
  "$PY" scripts/run_mixture_path_adaptive.py --model "$model" --shard-id 0 --num-shards 1 \
    --input-dir "$ONEBIT_OUT" --output-dir "$PATH_OUT"
}

echo "worker_start gpu=$GPU_ID utc=$(date -u +%FT%TZ)"
case "$GPU_ID" in
  0)
    run_attack_series audioseal
    run_onebit_path audioseal
    ;;
  1)
    run_attack_series wavmark
    ;;
  2)
    run_attack_series timbrewm
    ;;
  3)
    run_attack_series voicemark
    run_onebit_path voicemark
    ;;
  4)
    run_attack_series wmcodec
    ;;
  5)
    run_k8 audioseal
    run_k8 wavmark
    run_k8 voicemark
    ;;
  6)
    run_k8 timbrewm
    run_k8 wmcodec
    ;;
  *)
    echo "unsupported gpu id: $GPU_ID" >&2
    exit 2
    ;;
esac

# Run one frequency/temporal shard after each GPU's higher-priority queue.
"$PY" scripts/run_frequency_temporal_k5_shard.py --shard-id "$GPU_ID" --num-shards 7 \
  --n-trials 300 --output-dir "$REGION_OUT"
echo "worker_complete gpu=$GPU_ID utc=$(date -u +%FT%TZ)"

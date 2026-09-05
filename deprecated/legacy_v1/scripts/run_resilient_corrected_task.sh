#!/usr/bin/env bash
set -u

GPU_ID=${1:?GPU id required}
TASK=${2:?task required}
ROOT=/private/users/lym/neural-audio-watermark-collusion-attack
PY=/private/users/lym/venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="/tmp/k5case_deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

ATTACK_OUT=results/evaluation_300clips_20260902
K8_OUT=data/k8_population_native_300clips_20260902
K8_VALID=data/k8_population_source_correct_300clips_20260902/raw
ONEBIT_OUT=results/one_bit_k5_300clips_20260902
PATH_OUT=data/mixture_path_k5_300clips_20260902
REGION_OUT=results/frequency_temporal_k5_300clips_20260902

run_until_csv() {
  local output=$1
  shift
  while [[ ! -f "$output" ]] || [[ "$(wc -l < "$output")" -ne 301 ]]; do
    "$@"
    rc=$?
    echo "attempt_exit task=$TASK rc=$rc utc=$(date -u +%FT%TZ)"
    sleep 2
  done
}

case "$TASK" in
  voicemark_k5)
    run_until_csv "$ATTACK_OUT/attack_voicemark_K5.csv" \
      "$PY" scripts/attack.py --model voicemark --K 5 --n_trials 300 \
      --mean-only --output-dir "$ATTACK_OUT"
    ;;
  wmcodec_k5)
    run_until_csv "$ATTACK_OUT/attack_wmcodec_K5.csv" \
      "$PY" scripts/attack.py --model wmcodec --K 5 --n_trials 300 \
      --mean-only --output-dir "$ATTACK_OUT"
    ;;
  wavmark_main)
    for kval in 2 3 5; do
      run_until_csv "$ATTACK_OUT/attack_wavmark_K${kval}.csv" \
        "$PY" scripts/attack.py --model wavmark --K "$kval" --n_trials 300 \
        --mean-only --output-dir "$ATTACK_OUT"
    done
    ;;
  voicemark_k8)
    while [[ "$(find "$K8_OUT/raw/voicemark" -type f -name 'trial_*.json' 2>/dev/null | wc -l)" -lt 300 ]]; do
      "$PY" scripts/run_k8_population_native.py --model voicemark --n-trials 300 --output-dir "$K8_OUT"
      echo "attempt_exit task=$TASK.stage_raw rc=$? utc=$(date -u +%FT%TZ)"
      sleep 2
    done
    while [[ "$(find "$K8_VALID/voicemark" -type f -name 'trial_*.json' 2>/dev/null | wc -l)" -lt 300 ]]; do
      "$PY" scripts/complete_k8_source_correct_population.py --model voicemark \
        --source-dir "$K8_OUT/raw" --dest-dir "$K8_VALID"
      echo "attempt_exit task=$TASK.stage_valid rc=$? utc=$(date -u +%FT%TZ)"
      sleep 2
    done
    ;;
  wavmark_k8)
    while [[ "$(find "$K8_OUT/raw/wavmark" -type f -name 'trial_*.json' 2>/dev/null | wc -l)" -lt 300 ]]; do
      "$PY" scripts/run_k8_population_native.py --model wavmark --n-trials 300 --output-dir "$K8_OUT"
      echo "attempt_exit task=$TASK.stage_raw rc=$? utc=$(date -u +%FT%TZ)"
      sleep 2
    done
    while [[ "$(find "$K8_VALID/wavmark" -type f -name 'trial_*.json' 2>/dev/null | wc -l)" -lt 300 ]]; do
      "$PY" scripts/complete_k8_source_correct_population.py --model wavmark \
        --source-dir "$K8_OUT/raw" --dest-dir "$K8_VALID"
      echo "attempt_exit task=$TASK.stage_valid rc=$? utc=$(date -u +%FT%TZ)"
      sleep 2
    done
    ;;
  voicemark_path)
    run_until_csv "$ONEBIT_OUT/onebit_k5_voicemark_full.csv" \
      "$PY" scripts/run_onebit_k5_pair_analysis.py --model voicemark --output-dir "$ONEBIT_OUT"
    path_file="$PATH_OUT/shards/mixture_path_voicemark_shard0of1.csv"
    while [[ ! -f "$path_file" ]]; do
      "$PY" scripts/run_mixture_path_adaptive.py --model voicemark --shard-id 0 --num-shards 1 \
        --input-dir "$ONEBIT_OUT" --output-dir "$PATH_OUT"
      echo "attempt_exit task=$TASK.stage_path rc=$? utc=$(date -u +%FT%TZ)"
      sleep 2
    done
    ;;
  region)
    SHARD=${3:?shard id required}
    shift 3
    MODELS=("$@")
    for model in "${MODELS[@]}"; do
      output="$REGION_OUT/shards/frequency_temporal_k5_${model}_shard${SHARD}of7.csv"
      while [[ ! -f "$output" ]]; do
        "$PY" scripts/run_frequency_temporal_k5_shard.py --shard-id "$SHARD" --num-shards 7 \
          --n-trials 300 --models "$model" --output-dir "$REGION_OUT"
        echo "attempt_exit task=region.$model.shard$SHARD rc=$? utc=$(date -u +%FT%TZ)"
        sleep 2
      done
    done
    ;;
  *)
    echo "unknown task: $TASK" >&2
    exit 2
    ;;
esac

echo "task_complete task=$TASK gpu=$GPU_ID utc=$(date -u +%FT%TZ)"

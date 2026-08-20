#!/usr/bin/env bash
# Defer the matched-N tamper suite until every currently running P0/P1 output is complete.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

complete_rows() {
    local path=$1 expected=$2
    [[ -s "$path" ]] || return 1
    (( $(wc -l < "$path") - 1 >= expected ))
}

while true; do
    ready=1
    for k in 2 3 5 8; do
        complete_rows "results/evaluation/tamper_arbitrary_detail_wavmark_K${k}.csv" 3000 || ready=0
    done
    complete_rows results/evaluation/temporal_sensitivity_wavmark_K5.csv 1400 || ready=0
    complete_rows results/evaluation/codec_sensitivity_wavmark_K5.csv 1800 || ready=0
    (( ready == 1 )) && break
    sleep 30
done

# Exercise both the generic batched detector and WavMark's sliding-window batch
# before releasing the full 20-configuration suite.  These first trials remain
# valid checkpoints and are resumed by the full queue.
while nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    sleep 10
done
WATERMARK_DEVICE=cuda:0 PYTHONPATH=src /private/users/lym/venv/bin/python scripts/framing.py \
    --model audioseal --K 2 --n_trials 1 --target_policy arbitrary --registry_size 1024 \
    > results/logs/matched_tamper_smoke_audioseal.log 2>&1
WATERMARK_DEVICE=cuda:0 PYTHONPATH=src /private/users/lym/venv/bin/python scripts/framing.py \
    --model wavmark --K 2 --n_trials 1 --target_policy arbitrary --registry_size 1024 \
    > results/logs/matched_tamper_smoke_wavmark.log 2>&1
for model in audioseal wavmark; do
    smoke="results/evaluation/tamper_arbitrary_N1024_${model}_K2.csv"
    [[ $(($(wc -l < "$smoke") - 1)) -eq 20 ]]
    head -n 1 "$smoke" | grep -q 'N_registry'
    head -n 1 "$smoke" | grep -q 'target_margin'
done

bash scripts/run_matched_tamper_queue.sh
bash scripts/finalize_and_push_results.sh

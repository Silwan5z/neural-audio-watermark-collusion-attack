#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
while true; do
    complete=1
    for k in 2 3 5 8; do
        output="results/evaluation/tamper_arbitrary_detail_wavmark_K${k}.csv"
        if [[ ! -s "$output" ]] || (( $(wc -l < "$output") - 1 < 3000 )); then
            complete=0
            break
        fi
    done
    (( complete == 1 )) && break
    sleep 30
done
exec /private/users/lym/venv/bin/python scripts/aggregate_wavmark_arbitrary.py

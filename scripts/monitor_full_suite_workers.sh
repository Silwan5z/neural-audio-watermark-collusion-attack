#!/usr/bin/env bash
# Keep GPUs 0--6 fed by attaching a replacement worker only after a sustained
# idle period.  The attached workers share the original queue's flock.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE_FILE="$ROOT/results/full_suite_next_index"
TOTAL_TASKS=193
POLL_SECONDS=30
IDLE_GRACE_SECONDS=90
declare -A idle_since=()

while true; do
    if [[ -r "$STATE_FILE" ]] && (( $(<"$STATE_FILE") >= TOTAL_TASKS )); then
        echo "$(date '+%F %T') queue exhausted; monitor exiting"
        exit 0
    fi
    now=$(date +%s)
    for gpu in 0 1 2 3 4 5 6; do
        pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | rg '^[0-9]+$' || true)
        if [[ -n "$pids" ]]; then
            unset 'idle_since[$gpu]'
            continue
        fi
        if [[ -z ${idle_since[$gpu]+x} ]]; then
            idle_since[$gpu]=$now
            continue
        fi
        if (( now - idle_since[$gpu] < IDLE_GRACE_SECONDS )); then
            continue
        fi
        session="watermark_full_suite_attach_gpu${gpu}"
        if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "$(date '+%F %T') gpu=${gpu} idle ${IDLE_GRACE_SECONDS}s; attaching worker"
            tmux new-session -d -s "$session" \
                "cd '$ROOT' && GPU_IDS='$gpu' exec scripts/attach_full_suite_workers.sh"
        fi
        idle_since[$gpu]=$now
    done
    sleep "$POLL_SECONDS"
done

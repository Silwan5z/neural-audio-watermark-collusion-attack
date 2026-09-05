#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG="$ROOT/results/logs/revision_reruns/monitor.log"
INTERVAL=${MONITOR_INTERVAL:-60}
mkdir -p "$(dirname "$LOG")"

while tmux has-session -t revision_reruns 2>/dev/null; do
    {
        echo "===== $(date '+%F %T') ====="
        printf 'queue_index='; cat "$ROOT/results/revision_reruns_next_index" 2>/dev/null || echo missing
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        echo "revision processes:"
        ps -eo pid,etimes,args | grep -E 'registry_size_control.py|joint_reachability_target.py' | grep -v grep || true
        echo "pre-existing training parent:"
        ps -p 2342710 -o pid,etimes,args --no-headers || echo "PASE parent no longer present"
        echo "failures:"
        tail -n 20 "$ROOT/results/revision_reruns_failures.log" 2>/dev/null || true
    } >> "$LOG"
    sleep "$INTERVAL"
done
echo "$(date '+%F %T') revision_reruns tmux session ended" >> "$LOG"

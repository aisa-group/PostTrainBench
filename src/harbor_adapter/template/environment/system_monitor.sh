#!/bin/bash
# Background system monitor — logs GPU, CPU, memory, and disk usage periodically.
# Ported from src/utils/system_monitor.sh in the condor pipeline.
#
# Writes to /logs/agent/system_monitor.log so it lands in the trial's agent/
# directory after Harbor downloads it. Logged at .log (not .txt) so the
# entrypoint's streaming tail glob (*.txt) doesn't flood the Modal dashboard
# with monitor lines every 60 seconds — the file is for postmortem analysis.

INTERVAL="${MONITOR_INTERVAL:-60}"
LOG_FILE="/logs/agent/system_monitor.log"

mkdir -p "$(dirname "$LOG_FILE")"

{
    echo "=== System Monitor Started (interval: ${INTERVAL}s) ==="
    echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
} > "$LOG_FILE"

while true; do
    {
        echo "--- $(date -u '+%Y-%m-%d %H:%M:%S UTC') ---"

        echo "[GPU]"
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

        echo "[GPU Processes]"
        nvidia-smi --query-compute-apps=pid,used_gpu_memory,name \
            --format=csv,noheader 2>/dev/null || echo "  none"

        echo "[CPU]"
        uptime

        echo "[Memory]"
        free -h | grep -E "Mem|Swap"

        echo "[Disk]"
        df -h /home/agent/workspace 2>/dev/null | tail -1
        echo "  Workspace dir: $(du -sh /home/agent/workspace 2>/dev/null | cut -f1)"

        echo ""
    } >> "$LOG_FILE"

    sleep "$INTERVAL"
done

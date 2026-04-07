#!/bin/bash
# Background system monitor — logs GPU, CPU, memory, and disk usage periodically.
# Writes to system_monitor.log in the current directory.

INTERVAL="${MONITOR_INTERVAL:-60}"
LOG_FILE="system_monitor.log"

echo "=== System Monitor Started (interval: ${INTERVAL}s) ===" > "$LOG_FILE"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

while true; do
    {
        echo "--- $(date -u '+%Y-%m-%d %H:%M:%S UTC') ---"

        # GPU metrics
        echo "[GPU]"
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

        # GPU processes
        echo "[GPU Processes]"
        nvidia-smi --query-compute-apps=pid,used_gpu_memory,name \
            --format=csv,noheader 2>/dev/null || echo "  none"

        # CPU load
        echo "[CPU]"
        uptime

        # Memory
        echo "[Memory]"
        free -h | grep -E "Mem|Swap"

        # Disk usage of working directory
        echo "[Disk]"
        df -h . 2>/dev/null | tail -1
        echo "  Task dir: $(du -sh . 2>/dev/null | cut -f1)"

        echo ""
    } >> "$LOG_FILE"

    sleep "$INTERVAL"
done

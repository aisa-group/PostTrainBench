#!/usr/bin/env bash
# timer.sh — Background task-timer daemon.
#
# Started at container boot by entrypoint.sh (ENTRYPOINT), and again via
# /etc/bash.bashrc and /etc/profile.d on the first exec() call, so the
# daemon is running regardless of how Harbor/Modal launches the sandbox.
# Idempotent: a PID file prevents duplicate instances.
#
# Writes to /home/agent/.timer/:
#   start_epoch    — epoch seconds when timer started (at container boot)
#   budget_secs    — total budget (from TASK_BUDGET_SECS env)
#   remaining_secs — refreshed every 10 seconds
#   elapsed_secs   — refreshed every 10 seconds
#   alert_30min    — created when ≤30 min remain
#   alert_10min    — created when ≤10 min remain
#   alert_5min     — created when ≤5 min remain
#
# Agent usage:
#   cat /home/agent/.timer/remaining_secs
#   test -f /home/agent/.timer/alert_30min

set -u

TIMER_DIR="/home/agent/.timer"
PID_FILE="$TIMER_DIR/timer.pid"

mkdir -p "$TIMER_DIR"

if [ -f "$PID_FILE" ]; then
    EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        exit 0
    fi
fi

echo $$ > "$PID_FILE"

START_EPOCH=$(date +%s)
BUDGET_SECS="${TASK_BUDGET_SECS:-36000}"

echo "$START_EPOCH" > "$TIMER_DIR/start_epoch"
echo "$BUDGET_SECS" > "$TIMER_DIR/budget_secs"

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_EPOCH))
    REMAINING=$((BUDGET_SECS - ELAPSED))

    if [ "$REMAINING" -lt 0 ]; then
        REMAINING=0
    fi

    echo "$REMAINING" > "$TIMER_DIR/remaining_secs"
    echo "$ELAPSED" > "$TIMER_DIR/elapsed_secs"

    if [ "$REMAINING" -le 1800 ] && [ ! -f "$TIMER_DIR/alert_30min" ]; then
        touch "$TIMER_DIR/alert_30min"
        echo "[TIMER] 30 minutes remaining" >&2
    fi

    if [ "$REMAINING" -le 600 ] && [ ! -f "$TIMER_DIR/alert_10min" ]; then
        touch "$TIMER_DIR/alert_10min"
        echo "[TIMER] 10 minutes remaining" >&2
    fi

    if [ "$REMAINING" -le 300 ] && [ ! -f "$TIMER_DIR/alert_5min" ]; then
        touch "$TIMER_DIR/alert_5min"
        echo "[TIMER] 5 minutes remaining" >&2
    fi

    if [ "$REMAINING" -le 0 ]; then
        echo "[TIMER] Time expired" >&2
        break
    fi

    sleep 10
done

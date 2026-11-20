#!/bin/bash

# This subscription-auth agent receives no API keys (see api_keys.json);
# forced_login_method below pins the codex CLI to ChatGPT auth.

# Force ChatGPT auth method (not API key)
if ! grep -q "forced_login_method" ~/.codex/config.toml 2>/dev/null; then
    printf '\nforced_login_method = "chatgpt"\n' >> ~/.codex/config.toml
fi

# Set reasoning effort to high (prepend to ensure precedence)
file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "max"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

# The GPT-6.2 models (2026-11-17) need codex 0.166.0 or newer. This per-model pin (the newest release on 2026-11-18)
# overrides any CODEX_CLI_VERSION set in .env.
if [[ "${AGENT_CONFIG}" == gpt-6.2-* ]]; then
    export CODEX_CLI_VERSION="0.170.0"
fi

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh codex || exit 1

if [[ "${AGENT_CONFIG}" != gpt-6.2-* ]]; then
    printf '%s' "$PROMPT" | codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG"
    exit $?
fi

# GPT-6.2 models (codex 0.170.0): a codex session that ends with an error (the subscription's usage limit until a
# reset credit is consumed, an OpenAI outage longer than codex's retries) is resumed with a short note while time is
# left, as in agents/codex_openrouter_max/solve.sh. Before each resume the loop waits, 1, 2, 4, ... minutes (at most
# 30); 10 resumes in a row span about 3 hours; a session that ran for 30 minutes or more made progress, and the count
# starts again. A session that ends normally is not resumed. The loop uses absolute paths and its own copy of
# timer.sh: on 2026-11-06 an opencode agent's tests deleted /home/ben/task.
TASK_DIR=/home/ben/task
TIMER="$HOME/solve_timer.sh"
cp "$TASK_DIR/timer.sh" "$TIMER" || exit 1
LAST_RUN="$HOME/codex_last_run.jsonl"
MAX_RESUMES=10
CODEX_ARGS=(--json -c model_reasoning_summary=detailed --skip-git-repo-check --model "$AGENT_CONFIG")

START=$(date +%s)
printf '%s' "$PROMPT" | codex --search exec "${CODEX_ARGS[@]}" --yolo | tee "$LAST_RUN"
CODEX_EXIT=${PIPESTATUS[1]}
RUN_SECONDS=$(( $(date +%s) - START ))
THREAD_ID=$(python3 -I -c '
import json, sys
for line in open(sys.argv[1]):
    if line.startswith("{\"type\":\"thread.started\""):
        print(json.loads(line)["thread_id"])
        break' "$LAST_RUN")

resumes=0
while [ "$CODEX_EXIT" -ne 0 ]; do
    [ "$RUN_SECONDS" -ge 1800 ] && resumes=0
    resumes=$((resumes + 1))
    if [ "$resumes" -gt "$MAX_RESUMES" ]; then
        echo "=== solve.sh: ${MAX_RESUMES} resumes in a row ended with an error within 30 minutes; not resuming again"
        break
    fi
    if [ -z "$THREAD_ID" ]; then
        echo "ERROR: codex exited with status ${CODEX_EXIT}, and its output has no thread id to resume"
        exit 1
    fi
    WAIT_MINUTES=$(( 1 << (resumes - 1) ))
    [ "$WAIT_MINUTES" -gt 30 ] && WAIT_MINUTES=30
    echo "=== solve.sh: codex exited with status ${CODEX_EXIT}; resume ${resumes}/${MAX_RESUMES} in ${WAIT_MINUTES} min"
    sleep "$((WAIT_MINUTES * 60))"
    TIMER_OUTPUT=$(bash "$TIMER")
    echo "$TIMER_OUTPUT" | grep -q "expired" && break
    REMAINING=$(echo "$TIMER_OUTPUT" | tail -n 1)
    if ! [[ "$REMAINING" =~ ^[0-9]+:[0-9][0-9]$ ]]; then
        echo "ERROR: $TIMER printed no remaining time (H:MM): $TIMER_OUTPUT; not resuming"
        exit 1
    fi
    cd "$TASK_DIR" || { echo "ERROR: $TASK_DIR does not exist; not resuming"; exit 1; }
    MESSAGE="Your session ended with an error (codex exit status ${CODEX_EXIT}), but ${REMAINING%%:*}h ${REMAINING#*:}m"
    MESSAGE+=" of the time limit remain. Processes that codex ran for you may have been stopped when the session ended;"
    MESSAGE+=" check what is still running. Continue working on the task."
    echo "=== solve.sh: resume ${resumes}/${MAX_RESUMES} of codex thread ${THREAD_ID} with the message: ${MESSAGE}"
    START=$(date +%s)
    printf '%s' "$MESSAGE" | codex --search exec resume "${CODEX_ARGS[@]}" --dangerously-bypass-approvals-and-sandbox \
        "$THREAD_ID" - | tee "$LAST_RUN"
    CODEX_EXIT=${PIPESTATUS[1]}
    RUN_SECONDS=$(( $(date +%s) - START ))
done
exit "$CODEX_EXIT"

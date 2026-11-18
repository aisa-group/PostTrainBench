#!/bin/bash

# Qwen Code (the `qwen` CLI, npm @qwen-code/qwen-code) in headless mode on an OpenRouter model.
# AGENT_CONFIG is <OpenRouter model id>[:<reasoning effort>], e.g. qwen/qwen4-max:xhigh. Every request is pinned to
# OpenRouter's "alibaba" endpoint without fallbacks and carries reasoning.effort; both go in extra_body, which qwen-code
# sends verbatim. The key is OPENROUTER_API_KEY_QWEN_4_MAX (a separate OpenRouter key for Qwen4 Max).
#
# Shell commands (checked in qwen-code 0.27.4 on 2026-11-16): a foreground command times out after 2 minutes, or after
# its own `timeout` of at most 10 minutes, and its whole process group is killed. Longer commands run with
# is_background: true. Headless mode does not wait for them, and it kills them when the session ends; processes started
# with `nohup ... &` live on. So a hung command cannot stall the run (antigravity's agy waited without a limit for every
# command that ran longer than 10 s).

if [ -z "${OPENROUTER_API_KEY_QWEN_4_MAX:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY_QWEN_4_MAX is not set (agents/qwen_code/api_keys.json allows it; is it in .env?)" >&2
    exit 1
fi

MODEL="${AGENT_CONFIG%%:*}"
EFFORT=""
[[ "$AGENT_CONFIG" == *:* ]] && EFFORT="${AGENT_CONFIG#*:}"

# OpenRouter ignores an unknown reasoning effort without an error, so check it against the model's supported efforts
# rather than run at the default effort under a dir name that says otherwise.
if [ -n "$EFFORT" ]; then
    if ! python3 - "$MODEL" "$EFFORT" <<'EOF'
import json, sys, urllib.request
model, effort = sys.argv[1], sys.argv[2]
with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as response:
    models = {m["id"]: m for m in json.load(response)["data"]}
if model not in models:
    sys.exit(f"ERROR: {model} is not in OpenRouter's model list")
efforts = (models[model].get("reasoning") or {}).get("supported_efforts") or []
if effort not in efforts:
    sys.exit(f"ERROR: {model} has no reasoning effort {effort!r} on OpenRouter; supported: {efforts}")
EOF
    then
        exit 1
    fi
fi

# User settings. Changes from qwen-code's defaults: the OpenRouter model; yolo approval; no per-turn tool-call cap
# (its default backstop ends a turn after 1000 tool calls, and a headless run is one turn); no managed memory (background
# memory extraction and consolidation, with extra model calls); no usage statistics and no self-update.
mkdir -p "$HOME/.qwen"
if ! python3 - "$MODEL" "$EFFORT" > "$HOME/.qwen/settings.json" <<'EOF'
import json, sys
model, effort = sys.argv[1], sys.argv[2]
extra_body = {"provider": {"order": ["alibaba"], "allow_fallbacks": False}}
if effort:
    extra_body["reasoning"] = {"effort": effort}
settings = {
    "modelProviders": {
        "openai": [
            {
                "id": model,
                "name": f"{model} (OpenRouter, alibaba endpoint)",
                "envKey": "OPENROUTER_API_KEY_QWEN_4_MAX",
                "baseUrl": "https://openrouter.ai/api/v1",
                "generationConfig": {"contextWindowSize": 1000000, "extra_body": extra_body},
            }
        ]
    },
    "security": {"auth": {"selectedType": "openai"}},
    "model": {"name": model, "maxToolCallsPerTurn": 0},
    "tools": {"approvalMode": "yolo"},
    "memory": {"enableManagedAutoMemory": False, "enableManagedAutoDream": False},
    "privacy": {"usageStatisticsEnabled": False},
    "general": {"enableAutoUpdate": False},
}
json.dump(settings, sys.stdout, indent=2)
EOF
then
    echo "ERROR: could not write $HOME/.qwen/settings.json" >&2
    exit 1
fi

# qwen-code's session records (~/.qwen/projects/<cwd>/chats/<session id>.jsonl: every message, tool call and result) go
# to ~/agent_logs, which run_task.sh copies to the result dir after the run, also after a timeout.
if [ -e "$HOME/.qwen/projects" ]; then
    echo "ERROR: $HOME/.qwen/projects already exists; it must become a link to ~/agent_logs/qwen_projects" >&2
    exit 1
fi
mkdir -p "$HOME/agent_logs/qwen_projects"
ln -s "$HOME/agent_logs/qwen_projects" "$HOME/.qwen/projects"

# The version whose shell timeouts and headless behaviour were checked (see the top of this file). QWEN_CLI_VERSION in
# .env overrides it.
export QWEN_CLI_VERSION="${QWEN_CLI_VERSION:-0.27.4}"
bash /home/ben/update_agent_cli.sh qwen || exit 1

# Retry HTTP 429/529 without a limit (backoff capped at 5 minutes) instead of ending the session.
export QWEN_CODE_UNATTENDED_RETRY=1
export QWEN_CODE_SUPPRESS_YOLO_WARNING=1

# Copies qwen's output to stdout and to the file $1, like `tee "$1"`, but waits when the stdout pipe is full instead of
# stopping (opencode made its stderr pipe non-blocking, and tee then lost the rest of a session; see agents/opencode).
# If the file cannot be written (the node's local disk ran full on 2026-11-17 and the copier died, so solve_out.txt
# lost the last 4 h of aime2025 17963052 while the agent kept working), it says so once and keeps copying to stdout:
# solve_out.txt is on the shared filesystem. The file then lacks the result event, which reads as an error -> resume.
copy_events() {
    python3 -c '
import os, select, sys
with open(sys.argv[1], "wb", buffering=0) as out:
    while chunk := os.read(0, 1 << 16):
        if out is not None:
            try:
                out.write(chunk)
            except OSError as exc:
                os.write(2, f"=== solve.sh: cannot write {sys.argv[1]} ({exc}); copying to stdout only\n".encode())
                out = None
        view = memoryview(chunk)
        while view:
            try:
                view = view[os.write(1, view):]
            except BlockingIOError:
                select.select([], [1], [])
' "$1"
}

# Prints how the last qwen run ended ("error": a result event with is_error, or no result event) and its session id.
session_state() {
    python3 -c '
import json, sys
session_id, ended = "", "error"
for line in open(sys.argv[1], errors="replace"):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(event, dict):
        continue
    session_id = session_id or event.get("session_id") or ""
    if event.get("type") == "result":
        ended = "error" if event.get("is_error") else "ok"
print(ended, session_id)
' "$1"
}

# A headless qwen run ends at the first model turn without a tool call (unless a background subagent or a scheduled
# wakeup is pending), and at an error that its retries do not resolve. While time is left, a session that ended with an
# error, or before final_model has any files, is resumed with a short note, at most MAX_RESUMES times. The loop uses
# absolute paths and its own copy of timer.sh: on 2026-11-06 an opencode agent's tests deleted /home/ben/task.
LAST_RUN="$HOME/qwen_last_run.jsonl"
TASK_DIR=/home/ben/task
TIMER="$HOME/solve_timer.sh"
cp "$TASK_DIR/timer.sh" "$TIMER" || exit 1
MAX_RESUMES=10
QWEN_ARGS=(--yolo --output-format stream-json)

# The prompt and the resume notes go in on stdin, not as -p arguments: an argument puts the whole prompt (which names
# evaluate.py) into qwen's command line, and on 2026-11-16 the agent's `ps | awk '/evaluate\.py/' | ... kill -9` and
# `pgrep -f evaluate.py` matched qwen itself and killed the session (gsm8k 17957724, rescued by a resume).
printf '%s' "$PROMPT" | qwen "${QWEN_ARGS[@]}" 2>&1 | copy_events "$LAST_RUN"
read -r ENDED_STATE SESSION_ID < <(session_state "$LAST_RUN")

for (( resume = 1; resume <= MAX_RESUMES; resume++ )); do
    cd "$TASK_DIR" || { echo "ERROR: $TASK_DIR does not exist; not resuming" >&2; exit 1; }
    if [ "$ENDED_STATE" = "error" ]; then
        ENDED="ended with an error"
    elif [ -z "$(ls -A "$TASK_DIR/final_model" 2>/dev/null)" ]; then
        ENDED="ended"
    else
        break
    fi
    TIMER_OUTPUT=$(bash "$TIMER")
    echo "$TIMER_OUTPUT" | grep -q "expired" && break
    if [ -z "$SESSION_ID" ]; then
        echo "ERROR: qwen's session ${ENDED}, and its output has no session id to resume" >&2
        exit 1
    fi
    REMAINING=$(echo "$TIMER_OUTPUT" | tail -n 1)
    if ! [[ "$REMAINING" =~ ^[0-9]+:[0-9][0-9]$ ]]; then
        echo "ERROR: $TIMER printed no remaining time (H:MM): $TIMER_OUTPUT; not resuming" >&2
        exit 1
    fi
    MESSAGE="Your session ${ENDED}, but ${REMAINING%%:*}h ${REMAINING#*:}m of the time limit remain"
    [ -n "$(ls -A "$TASK_DIR/final_model" 2>/dev/null)" ] || MESSAGE+=" and $TASK_DIR/final_model does not exist yet"
    MESSAGE+=". Shell commands that you started with is_background: true and that were still running were stopped when"
    MESSAGE+=" the session ended. Continue working on the task."
    # Alibaba's endpoint rejects every reply with a tool call once the history ends in rounds of the same tool call
    # ("Repetitive tool calls detected in the conversation history"), so resuming the same history fails again at once:
    # on 2026-11-17 humaneval 17963063 used up all 10 resumes in 2 minutes and lost 5h46m. /compress replaces the
    # history with a summary; a test on a copy of that session showed tool calls are accepted again afterwards.
    if grep -q "Repetitive tool calls detected" "$LAST_RUN"; then
        echo "=== solve.sh: compressing qwen session ${SESSION_ID}: the API rejected its history as repetitive tool calls"
        printf '%s' "/compress" | qwen --resume "$SESSION_ID" "${QWEN_ARGS[@]}" 2>&1 | copy_events "$HOME/qwen_compress.jsonl"
        MESSAGE+=" The model API rejected your last replies because the conversation repeated the same tool call over"
        MESSAGE+=" many consecutive rounds, so the history was compressed into a summary. Do not poll with identical"
        MESSAGE+=" commands in quick succession."
    fi
    echo "=== solve.sh: resume ${resume}/${MAX_RESUMES} of qwen session ${SESSION_ID} with the message: ${MESSAGE}"
    printf '%s' "$MESSAGE" | qwen --resume "$SESSION_ID" "${QWEN_ARGS[@]}" 2>&1 | copy_events "$LAST_RUN"
    read -r ENDED_STATE _ < <(session_state "$LAST_RUN")
done

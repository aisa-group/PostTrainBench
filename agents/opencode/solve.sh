#!/bin/bash

# OpenCode requires a config file for auto-approval permissions and provider setup
# Create opencode.json in the working directory
cat > opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "anthropic": {
      "options": {
        "apiKey": "{env:ANTHROPIC_API_KEY}"
      }
    },
    "openai": {
      "options": {
        "apiKey": "{env:OPENAI_API_KEY}"
      }
    },
    "opencode": {
      "options": {
        "apiKey": "{env:OPENCODE_API_KEY}"
      }
    },
    "zai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Z.AI",
      "options": {
        "baseURL": "https://api.z.ai/api/paas/v4",
        "apiKey": "{env:ZAI_API_KEY}"
      },
      "models": {
        "glm-5": {
          "name": "GLM-5"
        },
        "glm-4.7": {
          "name": "GLM-4.7"
        }
      }
    },
    "meta": {
      "options": {
        "apiKey": "{env:MODEL_API_KEY}"
      }
    },
    "openrouter": {
      "options": {
        "apiKey": "{env:OPENROUTER_API_KEY}"
      },
      "models": {
        "google/gemini-4-argon": {
          "options": {
            "reasoning": {"effort": "high"},
            "provider": {"order": ["google-vertex/global"], "allow_fallbacks": false}
          }
        }
      }
    }
  }
}
EOF

# opencode's data directory (its session database with every message, tool call and reasoning summary, and its log)
# lives in ~/agent_logs, which run_task.sh copies to the result dir after the run, also after a timeout. solve_out.txt
# can miss events: on 2026-11-03 opencode's stdout stalled at a 64 KB event, and 12 minutes of a session never reached it.
if [ -e "$HOME/.local/share/opencode" ]; then
    echo "ERROR: $HOME/.local/share/opencode already exists; it must become a link to ~/agent_logs/opencode" >&2
    exit 1
fi
mkdir -p "$HOME/agent_logs/opencode" "$HOME/.local/share"
ln -s "$HOME/agent_logs/opencode" "$HOME/.local/share/opencode"

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh opencode || exit 1

# AGENT_CONFIG is <provider>/<model>[:<variant>]; the variant is opencode's reasoning effort (--variant), e.g.
# openrouter/meta/muse-wonder:xhigh. Keeping it in AGENT_CONFIG puts it in the result dir name.
MODEL="${AGENT_CONFIG%%:*}"
VARIANT=""
[[ "$AGENT_CONFIG" == *:* ]] && VARIANT="${AGENT_CONFIG#*:}"

# On a fresh $HOME opencode starts from the models.dev catalog bundled with its release (1.17.18's predates
# meta/muse-wonder) and refreshes it only in the background, racing the model lookup. Refresh it first; the variant
# check below then fails fast if the model is still unknown.
[[ "$MODEL" == openrouter/* ]] && opencode models --refresh > /dev/null

VARIANT_ARGS=()
if [ -n "$VARIANT" ]; then
    # opencode ignores an unknown --variant, so check it against the model's variants rather than run at the default
    # effort under a dir name that says otherwise.
    if ! opencode models "${MODEL%%/*}" --verbose | python3 -c '
import json, sys
model, variant = sys.argv[1], sys.argv[2]
text = "\n" + sys.stdin.read()
start = text.find("\n" + model + "\n")
if start < 0:
    sys.exit(f"ERROR: {model} not in opencode model catalog")
info, _ = json.JSONDecoder().raw_decode(text[start + len(model) + 2:])
variants = sorted(info.get("variants") or {})
if variant not in variants:
    sys.exit(f"ERROR: {model} has no reasoning variant {variant!r}; available: {variants}")
' "$MODEL" "$VARIANT"; then
        exit 1
    fi
    VARIANT_ARGS=(--variant "$VARIANT")
fi

# Copies opencode's JSON events to stdout and to the file $1, like `tee "$1"`. Not tee: opencode makes its stderr pipe
# non-blocking, and that is the pipe behind solve.sh's stdout too, so tee's writes fail with EAGAIN whenever the pipe is
# full, and tee then stops copying to stdout (solve_out.txt lost the rest of a session in 3 runs on 2026-11-03/04). This
# copier waits until the pipe takes more.
copy_events() {
    python3 -c '
import os, select, sys
with open(sys.argv[1], "wb") as out:
    while chunk := os.read(0, 1 << 16):
        out.write(chunk)
        out.flush()
        view = memoryview(chunk)
        while view:
            try:
                view = view[os.write(1, view):]
            except BlockingIOError:
                select.select([], [1], [])
' "$1"
}

# `opencode run` ends the session at the first model turn without a tool call, and at an error from the model provider.
# Gemini 4 Argon ended runs both ways on 2026-11-03: after 17 minutes with a bare "}" and no final_model, and after 3 hours
# with a 403 SAFETY (content_policy_violation) from Google Vertex. So while time is left, a session that ended with a
# provider error, or before final_model has any files, is resumed with a short note, at most MAX_RESUMES times.
# --thinking adds the model's reasoning summaries to the JSON events (type "reasoning").
# The loop uses absolute paths, enters the task dir anew each round and keeps its own copy of timer.sh: on 2026-11-06
# unit tests that the agent executed deleted /home/ben/task, and the agent recreated it (timer.sh too).
EVENTS="$HOME/opencode_events.jsonl"
LAST_RUN="$HOME/opencode_last_run.jsonl"
TASK_DIR=/home/ben/task
TIMER="$HOME/solve_timer.sh"
cp "$TASK_DIR/timer.sh" "$TIMER" || exit 1
MAX_RESUMES=10

printf '%s' "$PROMPT" | opencode run --model "$MODEL" "${VARIANT_ARGS[@]}" --format json --thinking | copy_events "$LAST_RUN"
cat "$LAST_RUN" > "$EVENTS"
SESSION_ID=$(grep -o -m1 '"sessionID":"ses_[A-Za-z0-9]*"' "$EVENTS" | head -n 1 | cut -d'"' -f4)

for (( resume = 1; resume <= MAX_RESUMES; resume++ )); do
    cd "$TASK_DIR" || { echo "ERROR: $TASK_DIR does not exist; not resuming" >&2; exit 1; }
    if tail -n 1 "$LAST_RUN" | grep -q '"type":"error"'; then
        ENDED="ended with an error from the model provider"
    elif [ -z "$(ls -A "$TASK_DIR/final_model" 2>/dev/null)" ]; then
        ENDED="ended"
    else
        break
    fi
    TIMER_OUTPUT=$(bash "$TIMER")
    echo "$TIMER_OUTPUT" | grep -q "expired" && break
    if [ -z "$SESSION_ID" ]; then
        echo "ERROR: opencode's session ${ENDED}, and its output has no session id to resume" >&2
        exit 1
    fi
    REMAINING=$(echo "$TIMER_OUTPUT" | tail -n 1)
    if ! [[ "$REMAINING" =~ ^[0-9]+:[0-9][0-9]$ ]]; then
        echo "ERROR: $TIMER printed no remaining time (H:MM): $TIMER_OUTPUT; not resuming" >&2
        exit 1
    fi
    MESSAGE="Your session ${ENDED}, but ${REMAINING%%:*}h ${REMAINING#*:}m of the time limit remain"
    [ -n "$(ls -A "$TASK_DIR/final_model" 2>/dev/null)" ] || MESSAGE+=" and $TASK_DIR/final_model does not exist yet"
    MESSAGE+=". Continue working on the task."
    echo "=== solve.sh: resume ${resume}/${MAX_RESUMES} of opencode session ${SESSION_ID} with the message: ${MESSAGE}"
    printf '%s' "$MESSAGE" | opencode run --session "$SESSION_ID" --model "$MODEL" "${VARIANT_ARGS[@]}" --format json --thinking | copy_events "$LAST_RUN"
    cat "$LAST_RUN" >> "$EVENTS"
done

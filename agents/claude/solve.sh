#!/bin/bash
unset GEMINI_API_KEY
unset CODEX_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions
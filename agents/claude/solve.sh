#!/bin/bash

export BASH_MAX_TIMEOUT_MS="36000000"

export CLAUDE_CODE_EFFORT_LEVEL="high"

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh claude

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions

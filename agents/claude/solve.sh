#!/bin/bash

export BASH_MAX_TIMEOUT_MS="36000000"

export CLAUDE_CODE_EFFORT_LEVEL="high"

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh claude

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"
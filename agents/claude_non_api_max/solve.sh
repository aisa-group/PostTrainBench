#!/bin/bash

# Load OAuth token from file (copied by run_task.sh)
if [ -f /home/ben/oauth_token ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/ben/oauth_token)"
else
    echo "ERROR: No oauth_token file found at /home/ben/oauth_token"
    exit 1
fi

export BASH_MAX_TIMEOUT_MS="36000000"

# Set effort level to max (Opus 4.6 only — absolute maximum reasoning, no token constraints)
export CLAUDE_CODE_EFFORT_LEVEL="max"

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh claude

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"

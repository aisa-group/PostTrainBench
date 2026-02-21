#!/bin/bash
unset GEMINI_API_KEY
unset CODEX_API_KEY

# Clear API key so the CLI uses the OAuth token from subscription
export ANTHROPIC_API_KEY=""

# Load OAuth token from file (copied by run_task.sh)
if [ -f /home/ben/oauth_token ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/ben/oauth_token)"
else
    echo "ERROR: No oauth_token file found at /home/ben/oauth_token"
    exit 1
fi

export BASH_MAX_TIMEOUT_MS="36000000"

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"

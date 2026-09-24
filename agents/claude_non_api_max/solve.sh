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

# claude-fable-5-1 (2026-08 release) requires claude-code 2.1.257 .
# claude-opus-5-5   (2026-09 release) requires claude-code 2.1.280 .
# These per-model pins override any CLAUDE_CLI_VERSION set in .env.
# Other configs (opus-4-8, opus-5, fable-5) use CLAUDE_CLI_VERSION from .env if
# set, else latest (or the container's baked version under the global
# POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1 opt-out).
if [[ "${AGENT_CONFIG}" == claude-fable-5-1* ]]; then
    export CLAUDE_CLI_VERSION="2.1.257"
elif [[ "${AGENT_CONFIG}" == claude-opus-5-5* ]]; then
    export CLAUDE_CLI_VERSION="2.1.280"
fi

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh claude || exit 1

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions

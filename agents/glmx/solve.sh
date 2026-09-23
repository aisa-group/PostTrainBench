#!/bin/bash

# GLM-X (preview) agent using Claude Code against Z.AI's dedicated
# Anthropic-compatible endpoint. Recommended config from Z.AI:
# Reference: https://docs.z.ai/devpack/tool/claude

# Auth — dedicated endpoint uses its own key, not the regular ZAI_API_KEY
export ANTHROPIC_API_KEY="${GLMX_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${GLMX_API_KEY}"
export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"

export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_SMALL_FAST_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${AGENT_CONFIG}"

# Timeouts: keep long-running tool calls alive against this endpoint
export BASH_MAX_TIMEOUT_MS="36000000"
export API_TIMEOUT_MS="3600000"
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="3600000"

# Output budget and 1M-context auto-compaction
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="1000000"

# Stream eagerly so the agent's stdout (and our trace) gets data in real time
export CLAUDE_CODE_EAGER_FLUSH="1"

# Skip non-essential traffic + attribution headers per Z.AI recommendation
export CLAUDE_CODE_ATTRIBUTION_HEADER="0"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
export IS_SANDBOX="1"

# glm-5.3 / glm-5.3-flash (2026-08 spec) need claude-code >= 2.1.207 with a
# new env var to disable the client-side stream watchdog. Older glm configs
# (glm-x-preview, glm-5.2) work against the container's pinned 2.1.157 and
# don't set this var. Detect by AGENT_CONFIG prefix so this file stays
# backward-compatible.
if [[ "${AGENT_CONFIG}" == glm-5.3* ]]; then
    export CLAUDE_ENABLE_STREAM_WATCHDOG=0
    # Ask update_agent_cli.sh to install this exact version. Override the
    # global .env POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1 opt-out so the upgrade
    # actually runs (other glmx configs still respect the .env skip).
    export CLAUDE_CLI_VERSION="2.1.207"
    export POST_TRAIN_BENCH_SKIP_CLI_UPDATE=0
fi

bash /home/ben/update_agent_cli.sh claude

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json \
    --thinking adaptive --effort max --thinking-display summarized \
    --dangerously-skip-permissions

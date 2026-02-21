#!/bin/bash

# GLM 5 agent using Claude Code with Z.AI's Anthropic-compatible API
# Reference: https://docs.z.ai/devpack/tool/claude
# Note; right now you need a "Coding Plan" to use GLM 5, just API doenst work with the Anthropic endpoint 

export BASH_MAX_TIMEOUT_MS="36000000"
export API_TIMEOUT_MS="3000000"

# Configure Claude Code to use Z.AI's Anthropic-compatible API
export ANTHROPIC_API_KEY="${ZAI_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${ZAI_API_KEY}"
export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_SMALL_FAST_MODEL="${AGENT_CONFIG}"

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"

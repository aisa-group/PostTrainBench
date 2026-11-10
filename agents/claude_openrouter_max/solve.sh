#!/bin/bash

# Claude Code at max effort, paid per token through OpenRouter's Anthropic-compatible API instead of a subscription.
# CLAUDE_CODE_EXTRA_BODY is merged into every request: OpenRouter serves it only from its "anthropic" endpoint
# (no Bedrock/Vertex/Azure, no 2x-priced anthropic/fast). OpenRouter takes Anthropic's own model IDs
# (claude-fable-5-2, and claude-opus-4-8 / claude-opus-5 for Claude Code's refusal fallback).
if [ -z "${OPENROUTER_API_KEY}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set (agents/claude_openrouter_max/api_keys.json allows it; is it in .env?)"
    exit 1
fi
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="${OPENROUTER_API_KEY}"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_EXTRA_BODY='{"provider":{"order":["anthropic"],"allow_fallbacks":false}}'
# 1h prompt-cache TTL, as with subscription auth (an API key defaults to 5m, which rewrites the whole context after
# every wait on a training job longer than 5 minutes)
export ENABLE_PROMPT_CACHING_1H=1

export BASH_MAX_TIMEOUT_MS="36000000"

export CLAUDE_CODE_EFFORT_LEVEL="max"

# claude-fable-5-2 (2026-11 release) requires claude-code 2.1.322 (2.1.312 does not know it).
# This per-model pin overrides any CLAUDE_CLI_VERSION set in .env.
if [[ "${AGENT_CONFIG}" == claude-fable-5-2* ]]; then
    export CLAUDE_CLI_VERSION="2.1.322"
fi

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh claude || exit 1

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions

#!/bin/bash

# Qwen 3 Max Thinking agent using Claude Code with Qwen's Anthropic-compatible API
# Reference: https://qwen.ai/blog?id=qwen3-max-thinking

export BASH_MAX_TIMEOUT_MS="36000000"

# Configure Claude Code to use Qwen's Anthropic-compatible API (international endpoint)
# Override ANTHROPIC_API_KEY with DashScope key (Claude Code checks this first)
export ANTHROPIC_API_KEY="${DASHSCOPE_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${DASHSCOPE_API_KEY}"
export ANTHROPIC_BASE_URL="https://dashscope-intl.aliyuncs.com/apps/anthropic"
export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_SMALL_FAST_MODEL="${AGENT_CONFIG}"

# Debug: verify all environment variables are set
echo "DEBUG: DASHSCOPE_API_KEY is set: ${DASHSCOPE_API_KEY:+yes} (length: ${#DASHSCOPE_API_KEY})"
echo "DEBUG: ANTHROPIC_API_KEY is set: ${ANTHROPIC_API_KEY:+yes} (length: ${#ANTHROPIC_API_KEY})"
echo "DEBUG: ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
echo "DEBUG: ANTHROPIC_MODEL=${ANTHROPIC_MODEL}"

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"

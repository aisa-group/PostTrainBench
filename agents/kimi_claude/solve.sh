#!/bin/bash


export ANTHROPIC_API_KEY="${KIMI_API_KEY}"
export ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic"

export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_FABLE_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${AGENT_CONFIG}"
export CLAUDE_CODE_SUBAGENT_MODEL="${AGENT_CONFIG}"

export API_TIMEOUT_MS=12000000
export BUN_CONFIG_HTTP_IDLE_TIMEOUT=2000
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=1048576
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_EFFORT_LEVEL=max
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=1000000
export ENABLE_BACKGROUND_TASKS=1
export ENABLE_TOOL_SEARCH=false
export FORCE_AUTO_BACKGROUND_TASKS=1
export IS_SANDBOX=1

{
    echo "binary: claude"
    echo "package: @anthropic-ai/claude-code"
    echo "path: $(command -v claude || echo '<not found>')"
    echo "version: $(claude --version 2>&1 || echo '<version lookup failed>')"
    echo "update: skipped (kimi_claude pins container-baked version)"
    echo "recorded_at: $(date -Iseconds)"
} > "$HOME/cli_version.txt"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions

#!/bin/bash

# Kimi-provided settings for kismet-0715 via Moonshot's anthropic-compatible endpoint.
# API key comes from .env as KIMI_API_KEY (allowlisted in api_keys.json). Uses the
# claude-code CLI (harness pinned to 2.1.198 by Kimi — the container should already
# have it baked in; POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1 in .env prevents npm from
# swapping in "latest").

export ANTHROPIC_AUTH_TOKEN="${KIMI_API_KEY}"
export ANTHROPIC_API_KEY="${KIMI_API_KEY}"    # some SDK paths read this alias
export ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic"

# Route every model role to the single kismet model.
export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_FABLE_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${AGENT_CONFIG}"
export CLAUDE_CODE_SUBAGENT_MODEL="${AGENT_CONFIG}"

# Kimi-recommended runtime knobs.
export API_TIMEOUT_MS=12000000
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=1048576
export CLAUDE_CODE_EFFORT_LEVEL=max
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000
export ENABLE_TOOL_SEARCH=false
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=true

# Record which claude version we actually ran (no auto-update, per Kimi pin).
{
    echo "binary: claude"
    echo "package: @anthropic-ai/claude-code"
    echo "path: $(command -v claude || echo '<not found>')"
    echo "version: $(claude --version 2>&1 || echo '<version lookup failed>')"
    echo "update: skipped (kimi-cc pins container-baked version)"
    echo "recorded_at: $(date -Iseconds)"
} > "$HOME/cli_version.txt"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions

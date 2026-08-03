#!/bin/bash

# Subscription-auth Cursor CLI agent. No API keys are provisioned (see
# api_keys.json); authentication is via the OAuth tokens persisted in
# agents/cursor_cli/cursor_auth.json, which run_task.sh bind-mounts into the
# sandbox at /home/ben/.config/cursor/auth.json. To (re)generate it, install
# the CLI on the head node (`curl -fsS https://cursor.com/install | bash`),
# run `cursor-agent login`, then
# `cp ~/.config/cursor/auth.json agents/cursor_cli/cursor_auth.json && chmod 600 agents/cursor_cli/cursor_auth.json`.

set -eu

# Install the Cursor CLI into the user prefix. The installer drops `agent` and
# `cursor-agent` symlinks at $HOME/.local/bin/, pointing at the versioned
# bundle under $HOME/.local/share/cursor-agent/versions/<v>/cursor-agent.
# NOTE: check `cursor-agent`, not `agent` — xAI's grok CLI also symlinks
# `agent` (to its own binary), so `command -v agent` returns grok's binary
# when grok is baked into the container. `cursor-agent` is unambiguous.
if ! command -v cursor-agent >/dev/null 2>&1; then
    echo "[cursor_cli] installing Cursor CLI from cursor.com/install ..."
    curl -fsS https://cursor.com/install | bash || {
        echo "[cursor_cli] ERROR: Cursor CLI install failed" >&2
        exit 1
    }
fi
export PATH="$HOME/.local/bin:$PATH"

# Force Node's built-in fetch (undici) to honor http_proxy / https_proxy /
# no_proxy env vars. Without this, cursor-agent tries direct connections to
# Cursor's public IPs, which fail with "Error: [internal]" behind a corp proxy.
export NODE_USE_ENV_PROXY=1

# Record the CLI version so it appears alongside other agents' cli_version.txt.
{
    echo "binary: cursor-agent"
    echo "package: cursor.com/install (curl installer)"
    echo "path: $(command -v cursor-agent || echo '<not found>')"
    echo "version: $(cursor-agent --version 2>&1 || echo '<version lookup failed>')"
    echo "recorded_at: $(date -Iseconds)"
} > "$HOME/cli_version.txt"

cursor-agent \
    --print \
    --force \
    --trust \
    --output-format stream-json \
    --workspace /home/ben/task \
    --model "$AGENT_CONFIG" \
    "$PROMPT"

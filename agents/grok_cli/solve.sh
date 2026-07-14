#!/bin/bash

# Subscription-auth Grok Build agent (xAI). No API keys are provisioned (see
# api_keys.json); authentication is via the OAuth tokens persisted in
# agents/grok_cli/grok_auth.json, which run_task.sh bind-mounts into the sandbox
# at /home/ben/.grok/auth.json. To (re)generate it, run `grok login` on the host
# (use `--device-auth` for a headless flow) and then
# `cp ~/.grok/auth.json agents/grok_cli/grok_auth.json && chmod 600 agents/grok_cli/grok_auth.json`.

set -eu

# Install the grok CLI into the writable user prefix. The official installer
# drops the binary at $HOME/.grok/bin/grok and does not need root.
if ! command -v grok >/dev/null 2>&1; then
    echo "[grok_cli] installing grok CLI from x.ai/cli/install.sh ..."
    curl -fsSL https://x.ai/cli/install.sh | bash || {
        echo "[grok_cli] ERROR: grok CLI install failed" >&2
        exit 1
    }
fi
export PATH="$HOME/.grok/bin:$PATH"

# Record the CLI version so it appears alongside other agents' cli_version.txt.
{
    echo "binary: grok"
    echo "package: x.ai/cli (curl installer)"
    echo "path: $(command -v grok || echo '<not found>')"
    echo "version: $(grok --version 2>&1 || echo '<version lookup failed>')"
    echo "recorded_at: $(date -Iseconds)"
} > "$HOME/cli_version.txt"

grok \
    --oauth \
    --always-approve \
    --no-auto-update \
    --output-format streaming-json \
    --cwd /home/ben/task \
    -m "$AGENT_CONFIG" \
    -p "$PROMPT"

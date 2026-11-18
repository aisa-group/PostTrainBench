#!/bin/bash

# This subscription-auth agent receives no API keys (see api_keys.json);
# forced_login_method below pins the codex CLI to ChatGPT auth.

# Force ChatGPT auth method (not API key)
if ! grep -q "forced_login_method" ~/.codex/config.toml 2>/dev/null; then
    printf '\nforced_login_method = "chatgpt"\n' >> ~/.codex/config.toml
fi

# Set reasoning effort to high (prepend to ensure precedence)
file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "max"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

# The GPT-6.2 models (2026-11-17) need codex 0.166.0 or newer. This per-model pin (the newest release on 2026-11-18)
# overrides any CODEX_CLI_VERSION set in .env.
if [[ "${AGENT_CONFIG}" == gpt-6.2-* ]]; then
    export CODEX_CLI_VERSION="0.170.0"
fi

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh codex || exit 1

printf '%s' "$PROMPT" | codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG"

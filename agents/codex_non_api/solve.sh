#!/bin/bash
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Clear API keys so the CLI uses the ChatGPT Pro auth from auth.json
export CODEX_API_KEY=""
export OPENAI_API_KEY=""

# Force ChatGPT auth method (not API key)
if ! grep -q "forced_login_method" ~/.codex/config.toml 2>/dev/null; then
    printf '\nforced_login_method = "chatgpt"\n' >> ~/.codex/config.toml
fi

codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"

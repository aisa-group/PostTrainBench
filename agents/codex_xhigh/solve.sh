#!/bin/bash
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Set reasoning effort to xhigh (prepend to ensure precedence)
file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "xhigh"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"

#!/bin/bash

unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "high"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"
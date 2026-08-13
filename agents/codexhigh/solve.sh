#!/bin/bash

file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "high"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh codex

printf '%s' "$PROMPT" | codex --search exec --skip-git-repo-check --yolo --model "$AGENT_CONFIG"

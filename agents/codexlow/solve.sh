#!/bin/bash

file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "low"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh codex

codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"
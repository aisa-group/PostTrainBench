#!/bin/bash

file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "low"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

codex --search exec --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"
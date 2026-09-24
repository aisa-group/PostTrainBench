#!/bin/bash

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh codex || exit 1

printf '%s' "$PROMPT" | codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG"
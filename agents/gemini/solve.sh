#!/bin/bash

export GEMINI_SANDBOX="false"
# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh gemini || exit 1

gemini --yolo --model "$AGENT_CONFIG" --output-format stream-json -p "$PROMPT"
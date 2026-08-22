#!/bin/bash
set -euo pipefail

# Keep provider credentials and run state isolated to this benchmark process.
MCODE_DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/posttrainbench-mcode.XXXXXX")"
chmod 700 "$MCODE_DATA_DIR"
export MINIMAX_DATA_DIR="$MCODE_DATA_DIR"

cleanup() {
    rm -rf -- "$MCODE_DATA_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh mcode

# Seed the per-run provider config from the environment. The resulting config
# lives only inside the mode-0700 temporary directory removed by the EXIT trap.
mcode provider set-minimax-key --api-key-env MINIMAX_API_KEY

printf '%s' "$PROMPT" | mcode exec \
    --input - \
    --cwd /home/ben/task \
    --permission full \
    --output-format stream-json \
    --model "$AGENT_CONFIG"

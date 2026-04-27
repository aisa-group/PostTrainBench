#!/bin/bash
#
# Rerun only the Sonnet 4.6 judge on a single result directory.
# Thin wrapper around run_judge.sh --sonnet-only (which writes _rerun outputs
# and leaves the existing contamination_judgement.txt / disallowed_model_judgement.txt
# untouched).
#
# Usage: rerun_single_sonnet_only.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/../run_judge.sh" --sonnet-only "$RESULT_DIR"

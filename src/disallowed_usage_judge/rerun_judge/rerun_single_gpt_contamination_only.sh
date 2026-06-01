#!/bin/bash
#
# Rerun only the GPT-5.4 contamination judge on a single result directory.
# Thin wrapper around run_judge.sh --gpt-contamination-only:
#   - Skips the GPT-5.4 API judge.
#   - Only writes judgement_gpt5_4_rerun.json (plus its raw/parsed trace).
#
# Usage: rerun_single_gpt_contamination_only.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/../run_judge.sh" --gpt-contamination-only "$RESULT_DIR"

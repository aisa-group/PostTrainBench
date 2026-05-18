#!/bin/bash
#
# Rerun only the DeepSeek V4 Flash Free judge on a single result directory.
# Thin wrapper around run_judge.sh --deepseek-only (which writes _rerun outputs
# and leaves the existing judgement_*.json / judge_result.json untouched).
#
# Usage: rerun_single_deepseek_only.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/../run_judge.sh" --deepseek-only "$RESULT_DIR"

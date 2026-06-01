#!/bin/bash
#
# Rerun the GPT-based judges on a single result directory.
# Thin wrapper around run_judge.sh that writes _rerun outputs and leaves
# the existing judgement_*.json untouched.
#
# The judge mode defaults to --gpt-only (GPT-5.4 contamination + API).
# commit_all_gpt_only.sh --skip-existing passes a narrower mode
# (--api-only or --gpt-contamination-only) when one of the two judge
# results is already present.
#
# Usage: rerun_single_gpt_only.sh <result_dir> [<judge_mode>]

set -e

RESULT_DIR="$1"
JUDGE_MODE="${2:---gpt-only}"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir> [<judge_mode>]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/../run_judge.sh" "$JUDGE_MODE" "$RESULT_DIR"

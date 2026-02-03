#!/bin/bash
#
# Rerun the judge on a single result directory.
# This is a wrapper around run_judge.sh with the --rerun flag.
#
# Usage: rerun_single.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/../run_judge.sh" --rerun "$RESULT_DIR"

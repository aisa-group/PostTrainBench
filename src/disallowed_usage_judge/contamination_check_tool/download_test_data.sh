#!/bin/bash
#
# Download benchmark test data for contamination checking.
# Runs inside the standard apptainer container.
#
# Usage: ./download_test_data.sh [--tasks task1 task2 ...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

echo "Running download_test_data.py inside apptainer container..."

apptainer exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env MY_HF_TOKEN="${MY_HF_TOKEN:-}" \
    --env HF_HOME="${HF_HOME}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/soft_file_locking.sif" \
    python3 "$SCRIPT_DIR/download_test_data.py" "$@"

echo "Done."

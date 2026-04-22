#!/bin/bash
#
# Recompute judge results (rerun mode) for the latest runs of the three
# claude_claude-opus-4-5_10h_{final_v3,v5,v6_seed1} method directories.
#
# Only the newest run (highest cluster_id) per (benchmark, model) is submitted.
# Runs with --rerun, so original judge output files are preserved; new results
# land in *_rerun.txt / *_rerun.json alongside them.
#
# This script avoids sourcing set_env_vars.sh because the module-loading block
# fails on nodes without tclsh; instead it exports POST_TRAIN_BENCH_RESULTS_DIR
# from .env directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_FILE="$SCRIPT_DIR/rerun_judge.sub"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE" >&2
    exit 1
fi

# Pull POST_TRAIN_BENCH_RESULTS_DIR from .env without invoking set_env_vars.sh.
POST_TRAIN_BENCH_RESULTS_DIR="$(grep -E '^POST_TRAIN_BENCH_RESULTS_DIR=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')"
export POST_TRAIN_BENCH_RESULTS_DIR
if [ -z "$POST_TRAIN_BENCH_RESULTS_DIR" ]; then
    echo "ERROR: POST_TRAIN_BENCH_RESULTS_DIR not set in $ENV_FILE" >&2
    exit 1
fi

METHODS=(
    "claude_claude-opus-4-5_10h_final_v3"
    "claude_claude-opus-4-5_10h_v5"
    "claude_claude-opus-4-5_10h_v6_seed1"
)

TOTAL_SUBMITTED=0
for method in "${METHODS[@]}"; do
    echo ""
    echo "########################################"
    echo "# Submitting rerun jobs for: $method"
    echo "########################################"

    RESULT_DIRS=$(python "$SCRIPT_DIR/list_results.py" \
        --paths-only --latest-only --method "$method")

    if [ -z "$RESULT_DIRS" ]; then
        echo "  (no directories found for $method)"
        continue
    fi

    COUNT=$(echo "$RESULT_DIRS" | wc -l)
    echo "  Submitting $COUNT jobs"

    while read -r result_dir; do
        [ -z "$result_dir" ] && continue
        condor_submit_bid 100 -a "result_dir=$result_dir" "$SUB_FILE"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    done <<< "$RESULT_DIRS"
done

echo ""
echo "========================================"
echo "Total rerun jobs submitted: $TOTAL_SUBMITTED"
echo "========================================"

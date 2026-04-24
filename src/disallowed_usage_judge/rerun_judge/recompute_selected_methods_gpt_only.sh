#!/bin/bash
#
# Recompute ONLY the GPT-5.4 judge for the latest runs of the three
# claude_claude-opus-4-5_10h_{final_v3,v5,v6_seed1} method directories.
#
# Only the newest run (highest cluster_id) per (benchmark, model) is submitted.
# Uses run_judge.sh --gpt-only under the hood, so the existing
# contamination_judgement.txt / disallowed_model_judgement.txt are NOT touched;
# the script only (re)writes the *_rerun.txt outputs plus the GPT-specific
# contamination_judgement_gpt5_4_rerun.txt / disallowed_model_judgement_gpt5_4_rerun.txt.
# The aggregated contamination_judgement_rerun.txt is rebuilt from the new
# GPT-5.4 result and the existing Sonnet 4.6 *_rerun.txt (if present).
#
# This script avoids sourcing set_env_vars.sh because the module-loading block
# fails on nodes without tclsh; instead it exports POST_TRAIN_BENCH_RESULTS_DIR
# from .env directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_FILE="$SCRIPT_DIR/rerun_judge_gpt_only.sub"
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

# Record submitted cluster IDs so the monitor script can look them up later.
LOG_DIR="$SCRIPT_DIR/recompute_selected_methods_gpt_only_runs"
mkdir -p "$LOG_DIR"
CLUSTER_LOG="$LOG_DIR/submitted_$(date +%Y%m%d_%H%M%S).txt"

TOTAL_SUBMITTED=0
for method in "${METHODS[@]}"; do
    echo ""
    echo "########################################"
    echo "# Submitting GPT-only rerun jobs for: $method"
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
        SUBMIT_OUT=$(condor_submit_bid 100 -a "result_dir=$result_dir" "$SUB_FILE" 2>&1)
        echo "$SUBMIT_OUT"
        CLUSTER_ID=$(echo "$SUBMIT_OUT" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)
        if [ -n "$CLUSTER_ID" ]; then
            echo "$CLUSTER_ID	$result_dir" >> "$CLUSTER_LOG"
        fi
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    done <<< "$RESULT_DIRS"
done

echo ""
echo "========================================"
echo "Total GPT-only rerun jobs submitted: $TOTAL_SUBMITTED"
echo "Cluster IDs logged to: $CLUSTER_LOG"
echo "========================================"

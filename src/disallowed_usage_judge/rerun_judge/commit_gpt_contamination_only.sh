#!/bin/bash
#
# Submit GPT-5.4-contamination-only judge rerun jobs for the newest run per
# (method, benchmark, model) combination across every experiment folder in
# POST_TRAIN_BENCH_RESULTS_DIR.
#
# Uses run_judge.sh --gpt-contamination-only under the hood. Only the GPT-5.4
# contamination judge runs: the GPT-5.4 API usage judge is skipped. The only
# file (re)written per result dir is judgement_gpt5_4_rerun.json (plus its raw
# trace + parsed text).
#
# This script avoids sourcing set_env_vars.sh because the module-loading block
# fails on nodes without tclsh; it pulls POST_TRAIN_BENCH_RESULTS_DIR from
# .env directly.
#
# Options:
#   --dry-run         Print the result directories that would be submitted,
#                     but do not actually call condor_submit_bid.
#   --skip-existing   Skip result directories that already have a
#                     judgement_gpt5_4_rerun.json file.

set -e

DRY_RUN=""
SKIP_EXISTING=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_FILE="$SCRIPT_DIR/rerun_judge_gpt_contamination_only.sub"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE" >&2
    exit 1
fi

POST_TRAIN_BENCH_RESULTS_DIR="$(grep -E '^POST_TRAIN_BENCH_RESULTS_DIR=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')"
export POST_TRAIN_BENCH_RESULTS_DIR
if [ -z "$POST_TRAIN_BENCH_RESULTS_DIR" ]; then
    echo "ERROR: POST_TRAIN_BENCH_RESULTS_DIR not set in $ENV_FILE" >&2
    exit 1
fi

RESULT_DIRS=$(python3 "$SCRIPT_DIR/list_results.py" --paths-only --latest-only)
if [ -z "$RESULT_DIRS" ]; then
    echo "No result directories found under $POST_TRAIN_BENCH_RESULTS_DIR"
    exit 0
fi

TOTAL=$(echo "$RESULT_DIRS" | grep -c .)
echo "Found $TOTAL latest-only result directories across all methods in $POST_TRAIN_BENCH_RESULTS_DIR"
if [ -n "$SKIP_EXISTING" ]; then
    echo "  --skip-existing: skipping dirs with existing judgement_gpt5_4_rerun.json"
fi

LOG_DIR="$SCRIPT_DIR/commit_gpt_contamination_only_runs"
mkdir -p "$LOG_DIR"
CLUSTER_LOG="$LOG_DIR/submitted_$(date +%Y%m%d_%H%M%S).txt"

CURRENT_METHOD=""
TOTAL_SUBMITTED=0
TOTAL_SKIPPED=0
while read -r result_dir; do
    [ -z "$result_dir" ] && continue
    METHOD="$(basename "$(dirname "$result_dir")")"
    if [ "$METHOD" != "$CURRENT_METHOD" ]; then
        echo ""
        echo "########################################"
        echo "# Method: $METHOD"
        echo "########################################"
        CURRENT_METHOD="$METHOD"
    fi

    if [ -n "$SKIP_EXISTING" ] && [ -f "$result_dir/judgement_gpt5_4_rerun.json" ]; then
        echo "  [skip-existing] $result_dir"
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        continue
    fi

    if [ -n "$DRY_RUN" ]; then
        echo "  [dry-run] $result_dir"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        continue
    fi
    sleep 1
    SUBMIT_OUT=$(condor_submit_bid 100 -a "result_dir=$result_dir" "$SUB_FILE" 2>&1)
    echo "$SUBMIT_OUT" | tail -2
    CLUSTER_ID=$(echo "$SUBMIT_OUT" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)
    if [ -n "$CLUSTER_ID" ]; then
        printf '%s\t%s\n' "$CLUSTER_ID" "$result_dir" >> "$CLUSTER_LOG"
    fi
    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
done <<< "$RESULT_DIRS"

echo ""
echo "========================================"
if [ -n "$DRY_RUN" ]; then
    echo "Dry run: would have submitted $TOTAL_SUBMITTED GPT-contamination-only rerun jobs"
else
    echo "Total GPT-contamination-only rerun jobs submitted: $TOTAL_SUBMITTED"
    echo "Cluster IDs logged to: $CLUSTER_LOG"
fi
if [ -n "$SKIP_EXISTING" ]; then
    echo "Skipped (existing judgement_gpt5_4_rerun.json): $TOTAL_SKIPPED"
fi
echo "========================================"

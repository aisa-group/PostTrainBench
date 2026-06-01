#!/bin/bash
#
# Submit GPT-5.4 judge rerun jobs (gpt-only) for the newest run per
# (method, benchmark, model) combination across every experiment folder in
# POST_TRAIN_BENCH_RESULTS_DIR.
#
# Uses run_judge.sh --gpt-only under the hood, so the existing
# judgement_*.json from the initial run are NOT touched; only
# judgement_gpt5_4_rerun.json (plus judgement_api_rerun.json from the API
# judge it also reruns) is (re)written.
#
# This script avoids sourcing set_env_vars.sh because the module-loading block
# fails on nodes without tclsh; it pulls POST_TRAIN_BENCH_RESULTS_DIR from
# .env directly.
#
# Options:
#   --dry-run        Print the result directories that would be submitted, but
#                    do not actually call condor_submit_bid.
#   --skip-existing  Per result dir, skip judges whose _rerun output already
#                    exists. If judgement_gpt5_4_rerun.json is present and
#                    judgement_api_rerun.json is missing, only the API judge
#                    is rerun (and vice versa). Dirs where both files exist
#                    are skipped entirely.

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
SUB_FILE="$SCRIPT_DIR/rerun_judge_gpt_only.sub"
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

LOG_DIR="$SCRIPT_DIR/commit_all_gpt_only_runs"
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

    JUDGE_MODE="--gpt-only"
    if [ -n "$SKIP_EXISTING" ]; then
        HAS_GPT=""; HAS_API=""
        [ -f "$result_dir/judgement_gpt5_4_rerun.json" ] && HAS_GPT=1
        [ -f "$result_dir/judgement_api_rerun.json" ] && HAS_API=1
        if [ -n "$HAS_GPT" ] && [ -n "$HAS_API" ]; then
            echo "  [skip] both _rerun files exist: $result_dir"
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
            continue
        elif [ -n "$HAS_GPT" ]; then
            JUDGE_MODE="--api-only"
        elif [ -n "$HAS_API" ]; then
            JUDGE_MODE="--gpt-contamination-only"
        fi
    fi

    if [ -n "$DRY_RUN" ]; then
        echo "  [dry-run] ($JUDGE_MODE) $result_dir"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        continue
    fi
    sleep 1
    SUBMIT_OUT=$(condor_submit_bid 100 \
        -a "result_dir=$result_dir" \
        -a "judge_mode=$JUDGE_MODE" \
        "$SUB_FILE" 2>&1)
    echo "$SUBMIT_OUT" | tail -2
    CLUSTER_ID=$(echo "$SUBMIT_OUT" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)
    if [ -n "$CLUSTER_ID" ]; then
        printf '%s\t%s\t%s\n' "$CLUSTER_ID" "$JUDGE_MODE" "$result_dir" >> "$CLUSTER_LOG"
    fi
    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
done <<< "$RESULT_DIRS"

echo ""
echo "========================================"
if [ -n "$DRY_RUN" ]; then
    echo "Dry run: would have submitted $TOTAL_SUBMITTED GPT-only rerun jobs"
else
    echo "Total GPT-only rerun jobs submitted: $TOTAL_SUBMITTED"
    echo "Cluster IDs logged to: $CLUSTER_LOG"
fi
if [ -n "$SKIP_EXISTING" ]; then
    echo "Skipped (both _rerun files already present): $TOTAL_SKIPPED"
fi
echo "========================================"

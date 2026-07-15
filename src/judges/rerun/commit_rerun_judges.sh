#!/bin/bash
#
# Submit judge rerun jobs for the newest run per (method, benchmark, model)
# combination across every experiment folder in POST_TRAIN_BENCH_RESULTS_DIR.
#
# Uses run_judges.sh under the hood, so the judgement_*.json files from the
# initial run are NOT touched; only the _rerun outputs of the selected judges
# are (re)written.
#
# This script avoids sourcing set_env_vars.sh because the module-loading block
# fails on nodes without tclsh; it pulls POST_TRAIN_BENCH_RESULTS_DIR from
# .env directly.
#
# Options:
#   --judges <a,b>   Comma-separated judges to rerun (default: all judges;
#                    see ALL_JUDGES in ../judge_lib.sh). e.g.
#                    --judges data_contamination_judge
#   --dry-run        Print the result directories that would be submitted, but
#                    do not actually call condor_submit_bid.
#   --skip-existing  Per result dir, only rerun the selected judges whose
#                    judgement_<id>_rerun.json is missing; dirs where every
#                    selected judge already has one are skipped entirely.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_FILE="$SCRIPT_DIR/rerun_judges.sub"
ENV_FILE="$REPO_ROOT/.env"

source "$SCRIPT_DIR/../judge_lib.sh"

DRY_RUN=""
SKIP_EXISTING=""
JUDGES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --judges) IFS=',' read -r -a JUDGES <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ${#JUDGES[@]} -eq 0 ]; then
    JUDGES=("${ALL_JUDGES[@]}")
fi

# Validate the judges and cache their output ids + condor weights.
declare -A OUTPUT_ID_BY_JUDGE WEIGHT_BY_JUDGE
for judge in "${JUDGES[@]}"; do
    load_judge_conf "$judge"
    OUTPUT_ID_BY_JUDGE[$judge]="$JUDGE_OUTPUT_ID"
    WEIGHT_BY_JUDGE[$judge]="$JUDGE_CONDOR_WEIGHT"
done

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
echo "Judges: ${JUDGES[*]}"
if [ -n "$SKIP_EXISTING" ]; then
    echo "  --skip-existing: skipping judges whose judgement_<id>_rerun.json already exists"
fi

LOG_DIR="$SCRIPT_DIR/submission_logs"
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

    RUN_JUDGES=("${JUDGES[@]}")
    if [ -n "$SKIP_EXISTING" ]; then
        RUN_JUDGES=()
        for judge in "${JUDGES[@]}"; do
            if [ ! -f "$result_dir/judgement_${OUTPUT_ID_BY_JUDGE[$judge]}_rerun.json" ]; then
                RUN_JUDGES+=("$judge")
            fi
        done
        if [ ${#RUN_JUDGES[@]} -eq 0 ]; then
            echo "  [skip] all _rerun files exist: $result_dir"
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
            continue
        fi
    fi

    JUDGES_ARG=$(IFS=,; echo "${RUN_JUDGES[*]}")
    WEIGHT=0
    for judge in "${RUN_JUDGES[@]}"; do
        [ "${WEIGHT_BY_JUDGE[$judge]}" -gt "$WEIGHT" ] && WEIGHT="${WEIGHT_BY_JUDGE[$judge]}"
    done

    if [ -n "$DRY_RUN" ]; then
        echo "  [dry-run] ($JUDGES_ARG) $result_dir"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        continue
    fi
    sleep 1
    SUBMIT_OUT=$(condor_submit_bid 100 \
        -a "result_dir=$result_dir" \
        -a "judges=$JUDGES_ARG" \
        -a "judge_weight=$WEIGHT" \
        "$SUB_FILE" 2>&1)
    echo "$SUBMIT_OUT" | tail -2
    CLUSTER_ID=$(echo "$SUBMIT_OUT" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)
    if [ -n "$CLUSTER_ID" ]; then
        printf '%s\t%s\t%s\n' "$CLUSTER_ID" "$JUDGES_ARG" "$result_dir" >> "$CLUSTER_LOG"
    fi
    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
done <<< "$RESULT_DIRS"

echo ""
echo "========================================"
if [ -n "$DRY_RUN" ]; then
    echo "Dry run: would have submitted $TOTAL_SUBMITTED rerun jobs (judges: ${JUDGES[*]})"
else
    echo "Total rerun jobs submitted: $TOTAL_SUBMITTED (judges: ${JUDGES[*]})"
    echo "Cluster IDs logged to: $CLUSTER_LOG"
fi
if [ -n "$SKIP_EXISTING" ]; then
    echo "Skipped (all selected _rerun files already present): $TOTAL_SKIPPED"
fi
echo "========================================"

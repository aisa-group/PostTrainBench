#!/bin/bash
#
# Submit HTCondor jobs to rerun the judge on past result directories.
#
# Usage: commit_rerun_judge.sh [options]
#
# Options:
#   --method <pattern>          Filter to result dirs whose method matches this substring
#   --benchmark <pattern>       Filter to result dirs whose name matches this substring
#   --skip-existing             Skip dirs that already have contamination_judgement_rerun.txt
#   --only-missing-judgement    Only re-run dirs where the original judge step failed
#                               (no contamination_judgement.txt or disallowed_model_judgement.txt)
#   --limit <n>                 Process at most n directories
#   --latest-only               Only the highest cluster_id per (method, model, benchmark)
#   --dry-run                   Print the dirs that would be submitted, then exit

set -e

METHOD_PATTERN=""
BENCHMARK_PATTERN=""
SKIP_EXISTING=""
ONLY_MISSING=""
LIMIT=0
LATEST_ONLY=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD_PATTERN="$2"; shift 2 ;;
        --benchmark) BENCHMARK_PATTERN="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="1"; shift ;;
        --only-missing-judgement) ONLY_MISSING="1"; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --latest-only) LATEST_ONLY="1"; shift ;;
        --dry-run) DRY_RUN="1"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

SUB_FILE="$SCRIPT_DIR/rerun_judge.sub"

echo "========================================"
echo "Submitting rerun judge jobs"
echo "  Method pattern: ${METHOD_PATTERN:-all}"
echo "  Benchmark pattern: ${BENCHMARK_PATTERN:-all}"
echo "  Skip existing: ${SKIP_EXISTING:-no}"
echo "  Only missing judgement: ${ONLY_MISSING:-no}"
echo "  Limit: ${LIMIT:-no limit}"
echo "  Latest only: ${LATEST_ONLY:-no}"
echo "  Dry run: ${DRY_RUN:-no}"
echo "========================================"

LIST_ARGS="--paths-only"
[ -n "$METHOD_PATTERN" ] && LIST_ARGS="$LIST_ARGS --method $METHOD_PATTERN"
[ -n "$BENCHMARK_PATTERN" ] && LIST_ARGS="$LIST_ARGS --benchmark $BENCHMARK_PATTERN"
[ -n "$SKIP_EXISTING" ] && LIST_ARGS="$LIST_ARGS --skip-existing"
[ -n "$ONLY_MISSING" ] && LIST_ARGS="$LIST_ARGS --only-missing-judgement"
[ "$LIMIT" -gt 0 ] && LIST_ARGS="$LIST_ARGS --limit $LIMIT"
[ -n "$LATEST_ONLY" ] && LIST_ARGS="$LIST_ARGS --latest-only"

RESULT_DIRS=$(python "$SCRIPT_DIR/list_results.py" $LIST_ARGS)
TOTAL=$(echo "$RESULT_DIRS" | grep -c . || echo 0)

echo "Found $TOTAL result directories"

if [ "$TOTAL" -eq 0 ]; then
    echo "No directories to process"
    exit 0
fi

if [ -n "$DRY_RUN" ]; then
    echo "$RESULT_DIRS"
    exit 0
fi

while read -r result_dir; do
    [ -z "$result_dir" ] && continue
    condor_submit_bid 100 -a "result_dir=$result_dir" "$SUB_FILE"
done <<< "$RESULT_DIRS"

echo ""
echo "========================================"
echo "Jobs submitted: $TOTAL"
echo "========================================"

#!/bin/bash
#
# Submit HTCondor jobs to rerun the judge on all past runs.
#
# Usage: commit_rerun_judge.sh [options]
#
# Options:
#   --method <pattern>   Only process result directories matching this pattern
#   --benchmark <pattern> Only process benchmarks matching this pattern
#   --skip-existing      Skip directories that already have rerun judgement files
#   --limit <n>          Process at most n directories
#   --latest-only        Only process the latest run (highest cluster_id) per method/model/benchmark

set -e

METHOD_PATTERN=""
BENCHMARK_PATTERN=""
SKIP_EXISTING=""
LIMIT=0
LATEST_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD_PATTERN="$2"; shift 2 ;;
        --benchmark) BENCHMARK_PATTERN="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="1"; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --latest-only) LATEST_ONLY="1"; shift ;;
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
echo "  Limit: ${LIMIT:-no limit}"
echo "  Latest only: ${LATEST_ONLY:-no}"
echo "========================================"

# Build list_results.py args
LIST_ARGS="--paths-only"
[ -n "$METHOD_PATTERN" ] && LIST_ARGS="$LIST_ARGS --method $METHOD_PATTERN"
[ -n "$BENCHMARK_PATTERN" ] && LIST_ARGS="$LIST_ARGS --benchmark $BENCHMARK_PATTERN"
[ -n "$SKIP_EXISTING" ] && LIST_ARGS="$LIST_ARGS --missing-rerun"
[ "$LIMIT" -gt 0 ] && LIST_ARGS="$LIST_ARGS --limit $LIMIT"
[ -n "$LATEST_ONLY" ] && LIST_ARGS="$LIST_ARGS --latest-only"

# Get result directories using Python utility
RESULT_DIRS=$(python "$SCRIPT_DIR/list_results.py" $LIST_ARGS)
TOTAL=$(echo "$RESULT_DIRS" | grep -c . || echo 0)

echo "Found $TOTAL result directories to submit"

if [ "$TOTAL" -eq 0 ]; then
    echo "No directories to process"
    exit 0
fi

SUBMITTED=0
echo "$RESULT_DIRS" | while read -r result_dir; do
    condor_submit_bid 100 -a "result_dir=$result_dir" "$SUB_FILE"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "========================================"
echo "Jobs submitted: $TOTAL"
echo "========================================"

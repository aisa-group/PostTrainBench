#!/bin/bash
#
# Rerun the judge on all result directories (without HTCondor).
#
# Usage: rerun_all.sh [options]
#
# Options:
#   --method <pattern>   Only process result directories matching this pattern
#   --benchmark <pattern> Only process benchmarks matching this pattern
#   --parallel <n>       Run n judges in parallel (default: 1)
#   --skip-existing      Skip directories that already have rerun judgement files
#   --limit <n>          Process at most n directories

set -e

METHOD_PATTERN=""
BENCHMARK_PATTERN=""
PARALLEL=1
SKIP_EXISTING=""
LIMIT=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD_PATTERN="$2"; shift 2 ;;
        --benchmark) BENCHMARK_PATTERN="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="1"; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

echo "========================================"
echo "Rerunning judge on result directories"
echo "  Method pattern: ${METHOD_PATTERN:-all}"
echo "  Benchmark pattern: ${BENCHMARK_PATTERN:-all}"
echo "  Parallel: $PARALLEL"
echo "  Skip existing: ${SKIP_EXISTING:-no}"
echo "  Limit: ${LIMIT:-no limit}"
echo "========================================"

# Build list_results.py args
LIST_ARGS="--paths-only"
[ -n "$METHOD_PATTERN" ] && LIST_ARGS="$LIST_ARGS --method $METHOD_PATTERN"
[ -n "$BENCHMARK_PATTERN" ] && LIST_ARGS="$LIST_ARGS --benchmark $BENCHMARK_PATTERN"
[ -n "$SKIP_EXISTING" ] && LIST_ARGS="$LIST_ARGS --missing-rerun"
[ "$LIMIT" -gt 0 ] && LIST_ARGS="$LIST_ARGS --limit $LIMIT"

# Get result directories using Python utility
RESULT_DIRS=$(python "$SCRIPT_DIR/list_results.py" $LIST_ARGS)
TOTAL=$(echo "$RESULT_DIRS" | grep -c . || echo 0)

echo "Found $TOTAL result directories to process"
echo "========================================"

if [ "$TOTAL" -eq 0 ]; then
    echo "No directories to process"
    exit 0
fi

if [ "$PARALLEL" -le 1 ]; then
    COUNT=0
    echo "$RESULT_DIRS" | while read -r result_dir; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] Processing..."
        bash "$SCRIPT_DIR/rerun_single.sh" "$result_dir" || echo "Failed: $result_dir"
    done
else
    echo "Running $PARALLEL parallel jobs..."
    echo "$RESULT_DIRS" | xargs -P "$PARALLEL" -I {} bash "$SCRIPT_DIR/rerun_single.sh" "{}"
fi

echo "========================================"
echo "Done"
echo "========================================"

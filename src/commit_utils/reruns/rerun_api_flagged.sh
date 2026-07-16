#!/bin/bash
#
# Resubmit the full training run for every result dir where the API-usage judge
# flagged disallowed third-party LLM API usage.
#
# For each flagged dir it:
#   1. Reads the original (eval, agent, model, hours, agent_config, num_gpus)
#      from line 1 of output.log (echoed by run_task.sh:46).
#   2. Reconstructs the experiment suffix from the parent (method) dir name so
#      the resubmit lands in the SAME top-level method dir. HTCondor assigns a
#      fresh cluster id, so the new subdir does not collide with the flagged
#      one.
#   3. Submits src/commit_utils/single_task.sub with those args.
#
# Usage:
#   bash src/commit_utils/reruns/rerun_api_flagged.sh [options]
#
# Options:
#   --method PATTERN     Substring-match top-level method dir name (case-insensitive)
#   --benchmark PATTERN  Substring-match subdir name (case-insensitive)
#   --bid N              condor_submit_bid value. Default: 100.
#   --max-parallel N     Cap concurrent resubmits via HTCondor concurrency_limits.
#                        HTCondor gives each user.<tag> 10 000 tokens; each job
#                        consumes 10000/N tokens, so N concurrent jobs run.
#                        Tag is user.${USER}_rerun_api (isolated from commit.sh
#                        pools). Default: no cap.
#   --dry-run            Print what would be submitted; do not call condor_submit_bid.
#   -h | --help          Show this help.
#
# Notes:
#   - Run from the repo root. single_task.sub uses repo-relative paths.
#   - The parent method dir is split on the "_${hours}h[_${N}gpu]" marker to
#     recover the experiment suffix. This is unambiguous unless an agent_config
#     literally contains e.g. "_10h" — in that case override the suffix by hand.

set -e

BID=100
METHOD=""
BENCHMARK=""
DRY_RUN=""
MAX_PARALLEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)        METHOD="$2"; shift 2 ;;
        --benchmark)     BENCHMARK="$2"; shift 2 ;;
        --bid)           BID="$2"; shift 2 ;;
        --max-parallel)  MAX_PARALLEL="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       sed -n '2,/^set -e/p' "$0"; exit 0 ;;
        *)               echo "ERROR: unknown arg '$1' (see --help)" >&2; exit 1 ;;
    esac
done

# Resolve concurrency-limit arg once (empty array = no cap passed to condor).
CONCURRENCY_ARGS=()
if [ -n "$MAX_PARALLEL" ]; then
    if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --max-parallel must be a positive integer (got '$MAX_PARALLEL')" >&2
        exit 1
    fi
    TOKENS_PER_JOB=$((10000 / MAX_PARALLEL))
    CONCURRENCY_ARGS=(-a "concurrency_limits=user.${USER}_rerun_api:${TOKENS_PER_JOB}")
    echo "Concurrency cap: ${MAX_PARALLEL} parallel (tag user.${USER}_rerun_api, ${TOKENS_PER_JOB} tokens/job)" >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

FIND_PY="$REPO_ROOT/src/disallowed_usage_judge/rerun_judge/find_disallowed_api_usage.py"
SUB_FILE="$REPO_ROOT/src/commit_utils/single_task.sub"
[ -f "$FIND_PY" ]  || { echo "ERROR: missing $FIND_PY"  >&2; exit 1; }
[ -f "$SUB_FILE" ] || { echo "ERROR: missing $SUB_FILE" >&2; exit 1; }

# ---------- collect flagged dirs + justifications ----------
FIND_ARGS=(--justification)
[ -n "$METHOD" ]    && FIND_ARGS+=(--method "$METHOD")
[ -n "$BENCHMARK" ] && FIND_ARGS+=(--benchmark "$BENCHMARK")

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
PATHS_FILE="$TMP_DIR/flagged_paths.txt"
REPORT_FILE="$TMP_DIR/report.txt"

python "$FIND_PY" "${FIND_ARGS[@]}" \
    2> "$REPORT_FILE" \
    >  "$PATHS_FILE"

# echo the justifications to stderr for visibility
cat "$REPORT_FILE" >&2

FLAGGED_COUNT=$(wc -l < "$PATHS_FILE" | tr -d ' ')
if [ "$FLAGGED_COUNT" -eq 0 ]; then
    echo "" >&2
    echo "Nothing to resubmit — no flagged dirs." >&2
    exit 0
fi

echo "" >&2
echo "==================== RESUBMITTING $FLAGGED_COUNT run(s) ====================" >&2

# ---------- resubmit each ----------
SUBMITTED=0
while read -r dir; do
    [ -z "$dir" ] && continue

    if [ ! -f "$dir/output.log" ]; then
        echo "SKIP  (no output.log): $dir" >&2
        continue
    fi

    # run_task.sh:46 echoes: eval agent model cluster hours agent_config num_gpus
    read -r EVAL AGENT MODEL _CLUSTER HOURS AGENT_CONFIG NUM_GPUS \
        < "$dir/output.log"
    NUM_GPUS="${NUM_GPUS:-1}"

    if [ -z "$EVAL" ] || [ -z "$AGENT" ] || [ -z "$MODEL" ] \
       || [ -z "$HOURS" ] || [ -z "$AGENT_CONFIG" ]; then
        echo "SKIP  (bad output.log line 1): $dir" >&2
        continue
    fi

    # Recover the experiment suffix from the parent method dir name.
    method_name=$(basename "$(dirname "$dir")")
    marker="_${HOURS}h"
    [ "$NUM_GPUS" -gt 1 ] 2>/dev/null && marker="${marker}_${NUM_GPUS}gpu"
    EXPERIMENT="${method_name##*${marker}}"

    echo ">> agent=$AGENT  cfg=$AGENT_CONFIG  eval=$EVAL  model=$MODEL  ${HOURS}h  ${NUM_GPUS}gpu  exp=$EXPERIMENT" >&2

    CMD=(condor_submit_bid "$BID"
        -a "experiment_name=$EXPERIMENT"
        -a "agent=$AGENT"
        -a "agent_config=$AGENT_CONFIG"
        -a "eval=$EVAL"
        -a "model_to_train=$MODEL"
        -a "num_hours=$HOURS"
        -a "num_gpus=$NUM_GPUS"
        "${CONCURRENCY_ARGS[@]}"
        "$SUB_FILE")

    if [ -n "$DRY_RUN" ]; then
        printf '   DRY-RUN: '; printf '%q ' "${CMD[@]}"; echo
    else
        ( cd "$REPO_ROOT" && "${CMD[@]}" )
    fi
    SUBMITTED=$((SUBMITTED + 1))
done < "$PATHS_FILE"

echo "" >&2
echo "Done. Submitted $SUBMITTED / $FLAGGED_COUNT flagged run(s)${DRY_RUN:+ (dry-run)}." >&2

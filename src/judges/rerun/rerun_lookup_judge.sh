#!/bin/bash
#
# Submit ptb_lookup_judge reruns (rerun_judges.sub -> run_judges.sh
# --judges ptb_lookup_judge) for the given result directories.
#
# Each positional argument may be either
#   - a method dir (e.g. /fast/hbhatnagar/ptb_results/glmx_glm-5.2-preview_1m__10h_run1):
#     the complete task result dirs inside it are submitted, keeping only the
#     latest run per benchmark+model — dirs are named
#     <benchmark>_<model>_<cluster_id> and the highest cluster id wins — or
#   - a single task result dir (recognized by its task/ subdir): submitted as
#     is, bypassing the latest-only filter.
#
# "Complete" = has a task/ subdir AND a trace file (solve_parsed.txt or
# solve_out.txt); incomplete dirs are skipped with a note. Outputs always get
# the _rerun suffix (judgement_ptb_lookup_rerun.json, ...), so any judge files
# produced during the original run_task.sh are preserved.
#
# Usage:
#   rerun_lookup_judge.sh [--dry-run] [--skip-existing] <dir> [<dir>...]
#
# Options:
#   --dry-run        Print the result dirs that would be submitted, but do not
#                    call condor_submit_bid.
#   --skip-existing  Skip result dirs that already have
#                    judgement_ptb_lookup_rerun.json.
#
# Example (GLM 5.2 test batch):
#   bash src/judges/rerun/rerun_lookup_judge.sh --dry-run /fast/hbhatnagar/ptb_results/glmx_glm-5.2-preview_1m__10h_run{1,2,3}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_FILE="src/judges/rerun/rerun_judges.sub"

JUDGE="ptb_lookup_judge"
source "$SCRIPT_DIR/../judge_lib.sh"
load_judge_conf "$JUDGE"
OUTPUT_ID="$JUDGE_OUTPUT_ID"

DRY_RUN=""
SKIP_EXISTING=""
INPUT_DIRS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) INPUT_DIRS+=("$1"); shift ;;
    esac
done

if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
    echo "Usage: $0 [--dry-run] [--skip-existing] <method_dir_or_result_dir>..." >&2
    exit 1
fi

echo "Judge: $JUDGE (output id: $OUTPUT_ID)"

# Expand method dirs into task result dirs; pass single result dirs through.
# Within a method dir, keep only the latest run per benchmark+model: dirs are
# named <benchmark>_<model>_<cluster_id> and the highest cluster id wins.
declare -A MAX_ID_BY_KEY DIR_BY_KEY
RESULT_DIRS=()
skipped_superseded=0
for input in "${INPUT_DIRS[@]}"; do
    input="${input%/}"
    if [ ! -d "$input" ]; then
        echo "ERROR: not a directory: $input" >&2
        exit 1
    fi
    if [ -d "$input/task" ]; then
        RESULT_DIRS+=("$input")
        continue
    fi
    MAX_ID_BY_KEY=()
    DIR_BY_KEY=()
    found=0
    for d in "$input"/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"
        name="$(basename "$d")"
        if [[ ! "$name" =~ ^(.+)_([0-9]+)$ ]]; then
            echo "ERROR: cannot parse trailing cluster id from dir name: $name" >&2
            exit 1
        fi
        key="${BASH_REMATCH[1]}"
        id="${BASH_REMATCH[2]}"
        found=1
        if [ -z "${MAX_ID_BY_KEY[$key]:-}" ] || [ "$id" -gt "${MAX_ID_BY_KEY[$key]}" ]; then
            MAX_ID_BY_KEY[$key]="$id"
            DIR_BY_KEY[$key]="$d"
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "ERROR: no subdirectories found in method dir: $input" >&2
        exit 1
    fi
    for d in "$input"/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"
        name="$(basename "$d")"
        [[ "$name" =~ ^(.+)_([0-9]+)$ ]]
        key="${BASH_REMATCH[1]}"
        if [ "$d" != "${DIR_BY_KEY[$key]}" ]; then
            echo "  [skip superseded by _${MAX_ID_BY_KEY[$key]}] $(basename "$input")/$name"
            skipped_superseded=$((skipped_superseded+1))
            continue
        fi
        RESULT_DIRS+=("$d")
    done
done

# The .sub file uses relative paths (executable args, logs/).
cd "$REPO_ROOT"

LOG_DIR="$SCRIPT_DIR/submission_logs"
mkdir -p "$LOG_DIR"
CLUSTER_LOG="$LOG_DIR/submitted_lookup_$(date +%Y%m%d_%H%M%S).tsv"

echo "Result dirs to consider (latest per benchmark+model): ${#RESULT_DIRS[@]}"

submitted=0
skipped_incomplete=0
skipped_existing=0
CURRENT_METHOD=""
for result_dir in "${RESULT_DIRS[@]}"; do
    METHOD="$(basename "$(dirname "$result_dir")")"
    if [ "$METHOD" != "$CURRENT_METHOD" ]; then
        echo ""
        echo "######## $METHOD ########"
        CURRENT_METHOD="$METHOD"
    fi
    name="$(basename "$result_dir")"

    if [ ! -d "$result_dir/task" ]; then
        echo "  [skip incomplete: no task/] $name"
        skipped_incomplete=$((skipped_incomplete+1))
        continue
    fi
    if [ ! -f "$result_dir/solve_parsed.txt" ] && [ ! -f "$result_dir/solve_out.txt" ]; then
        echo "  [skip incomplete: no trace] $name"
        skipped_incomplete=$((skipped_incomplete+1))
        continue
    fi
    if [ -n "$SKIP_EXISTING" ] && [ -f "$result_dir/judgement_${OUTPUT_ID}_rerun.json" ]; then
        echo "  [skip existing: judgement_${OUTPUT_ID}_rerun.json] $name"
        skipped_existing=$((skipped_existing+1))
        continue
    fi

    if [ -n "$DRY_RUN" ]; then
        echo "  [dry-run submit] $result_dir"
        submitted=$((submitted+1))
        continue
    fi
    sleep 1
    SUBMIT_OUT="$(condor_submit_bid 100 \
        -a "result_dir=$result_dir" \
        -a "judges=$JUDGE" \
        "$SUB_FILE" 2>&1)"
    echo "$SUBMIT_OUT" | tail -1
    CLUSTER_ID="$(echo "$SUBMIT_OUT" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)"
    if [ -z "$CLUSTER_ID" ]; then
        echo "ERROR: could not parse cluster id for $result_dir" >&2
        echo "$SUBMIT_OUT" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "$CLUSTER_ID" "$JUDGE" "$result_dir" >> "$CLUSTER_LOG"
    submitted=$((submitted+1))
done

echo ""
echo "========================================"
if [ -n "$DRY_RUN" ]; then
    echo "DRY RUN: would submit $submitted $JUDGE rerun jobs"
else
    echo "Submitted: $submitted $JUDGE rerun jobs"
    echo "Cluster IDs logged to: $CLUSTER_LOG"
fi
echo "Skipped (superseded by newer run): $skipped_superseded"
echo "Skipped (incomplete): $skipped_incomplete"
if [ -n "$SKIP_EXISTING" ]; then
    echo "Skipped (judgement_${OUTPUT_ID}_rerun.json exists): $skipped_existing"
fi
echo "========================================"

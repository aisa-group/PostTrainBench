#!/bin/bash
#
# Batch driver for the data_contamination_judge multi-run variance experiment.
#
# Submits one condor job per (method, benchmark, model) cell — always the
# latest cluster id in a method dir — running data_contamination_judge fresh
# and writing to <result_dir>/judgement_multi_runs/judgement_gpt5_4_run<slot>.json.
#
# Slot 1 = the pre-existing verdict (judgement_gpt5_4_rerun.json when present,
# else judgement_gpt5_4.json). This script produces slot 2 and/or slot 3.
#
# Usage:
#   rerun_judge_slot.sh [--slot 2|3] [--dry-run] [--skip-existing] <dir> [<dir>...]
#
# Options:
#   --slot           2 or 3. Omit to submit slot 2, wait for all its jobs to
#                    finish (poll condor_q), then submit slot 3.
#   --dry-run        Print result dirs that would be submitted; do not submit.
#   --skip-existing  Skip cells that already have the target slot's verdict.
#
# Each positional argument may be a method dir (result dirs inside are filtered
# to the latest cluster id per benchmark+model) or a single result dir.
#
# Examples:
#   # Submit both slots for GLM 5.3 + flash, waiting between them:
#   bash src/judges/multi_run/rerun_judge_slot.sh \
#       /fast/hbhatnagar/ptb_results/glmx_glm-5.3_1m__10h_run{1,2} \
#       /fast/hbhatnagar/ptb_results/glmx_glm-5.3-flash_10h_run{1,2}
#
#   # Just slot 3 (manual):
#   bash src/judges/multi_run/rerun_judge_slot.sh --slot 3 <dirs...>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUB_FILE="src/judges/multi_run/rerun_judges_slot.sub"

# JUDGE_OUTPUT_ID for data_contamination_judge, without sourcing judge_lib.sh
# (that pulls in bash arrays that interact badly with set -u here).
JUDGE_NAME="data_contamination_judge"
JUDGE_OUTPUT_ID="gpt5_4"

# ---------- args ----------
SLOT=""
DRY_RUN=""
SKIP_EXISTING=""
INPUT_DIRS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --slot) SLOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) INPUT_DIRS+=("$1"); shift ;;
    esac
done

if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
    echo "Usage: $0 [--slot 2|3] [--dry-run] [--skip-existing] <dir>..." >&2
    exit 1
fi
if [ -n "$SLOT" ] && [ "$SLOT" != "2" ] && [ "$SLOT" != "3" ]; then
    echo "ERROR: --slot must be 2 or 3 (got: $SLOT)" >&2
    exit 1
fi

# ---------- expand inputs to per-cell result_dirs (latest per triple) ----------
declare -a RESULT_DIRS
RESULT_DIRS=()
declare -A MAX_ID_BY_KEY DIR_BY_KEY
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
    MAX_ID_BY_KEY=(); DIR_BY_KEY=()
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
        echo "ERROR: no subdirectories in method dir: $input" >&2
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

cd "$REPO_ROOT"
LOG_DIR="$SCRIPT_DIR/submission_logs"
mkdir -p "$LOG_DIR"

# submit_slot <N> — submits one slot worth of jobs; echoes cluster IDs to stdout.
submit_slot() {
    local slot="$1"
    local ts; ts="$(date +%Y%m%d_%H%M%S)"
    local cluster_log="$LOG_DIR/submitted_slot${slot}_${ts}.tsv"

    local submitted=0 skipped_incomplete=0 skipped_existing=0
    local current_method=""
    local -a cluster_ids=()

    echo "" >&2
    echo "======== SLOT $slot: submitting ==========================" >&2
    echo "Result dirs to consider (latest per benchmark+model): ${#RESULT_DIRS[@]}" >&2

    for result_dir in "${RESULT_DIRS[@]}"; do
        local method
        method="$(basename "$(dirname "$result_dir")")"
        if [ "$method" != "$current_method" ]; then
            echo "" >&2
            echo "######## $method ########" >&2
            current_method="$method"
        fi
        local name; name="$(basename "$result_dir")"

        if [ ! -d "$result_dir/task" ]; then
            echo "  [skip incomplete: no task/] $name" >&2
            skipped_incomplete=$((skipped_incomplete+1))
            continue
        fi
        if [ ! -f "$result_dir/solve_parsed.txt" ] && [ ! -f "$result_dir/solve_out.txt" ]; then
            echo "  [skip incomplete: no trace] $name" >&2
            skipped_incomplete=$((skipped_incomplete+1))
            continue
        fi
        local slot_file="$result_dir/judgement_multi_runs/judgement_${JUDGE_OUTPUT_ID}_run${slot}.json"
        if [ -n "$SKIP_EXISTING" ] && [ -f "$slot_file" ]; then
            echo "  [skip existing: $(basename $slot_file)] $name" >&2
            skipped_existing=$((skipped_existing+1))
            continue
        fi

        if [ -n "$DRY_RUN" ]; then
            echo "  [dry-run submit slot=$slot] $result_dir" >&2
            submitted=$((submitted+1))
            continue
        fi
        sleep 1
        local submit_out
        submit_out="$(condor_submit_bid 100 \
            -a "result_dir=$result_dir" \
            -a "slot=$slot" \
            "$SUB_FILE" 2>&1)"
        echo "$submit_out" | tail -1 >&2
        local cid
        cid="$(echo "$submit_out" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)"
        if [ -z "$cid" ]; then
            echo "ERROR: could not parse cluster id for $result_dir" >&2
            echo "$submit_out" >&2
            exit 1
        fi
        printf '%s\t%s\t%s\t%s\n' "$cid" "slot${slot}" "$JUDGE_NAME" "$result_dir" >> "$cluster_log"
        cluster_ids+=("$cid")
        submitted=$((submitted+1))
    done

    echo "" >&2
    echo "----------------------------------------" >&2
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN slot $slot: would submit $submitted jobs" >&2
    else
        echo "Slot $slot submitted: $submitted jobs" >&2
        echo "Cluster log: $cluster_log" >&2
    fi
    echo "Skipped (incomplete): $skipped_incomplete" >&2
    [ -n "$SKIP_EXISTING" ] && echo "Skipped (slot file exists): $skipped_existing" >&2
    echo "----------------------------------------" >&2

    # Emit cluster ids on stdout (one per line) for the caller
    printf '%s\n' "${cluster_ids[@]}"
}

# wait_for_clusters <cid> [<cid> ...]
# Poll condor_q until none of the given cluster IDs remain in the queue.
wait_for_clusters() {
    [ "$#" -eq 0 ] && return 0
    echo "" >&2
    echo "Waiting for ${#} slot-2 clusters to drain from condor_q ..." >&2
    local ids=("$@")
    while :; do
        local still=0
        local q_out
        q_out="$(condor_q hbhatnagar -nobatch -af ClusterId 2>/dev/null || echo "")"
        for cid in "${ids[@]}"; do
            if echo "$q_out" | grep -qE "^${cid}$"; then
                still=$((still+1))
            fi
        done
        if [ "$still" -eq 0 ]; then
            echo "All slot-2 clusters have left the queue." >&2
            return 0
        fi
        echo "  $still / ${#ids[@]} still queued/running — sleeping 60s ..." >&2
        sleep 60
    done
}

echo "Skipped (superseded by newer run): $skipped_superseded" >&2

# ---------- dispatch ----------
if [ -n "$SLOT" ]; then
    submit_slot "$SLOT" > /dev/null
else
    # Both slots: submit 2, wait, submit 3.
    mapfile -t SLOT2_CIDS < <(submit_slot 2)
    if [ -z "$DRY_RUN" ] && [ "${#SLOT2_CIDS[@]}" -gt 0 ]; then
        wait_for_clusters "${SLOT2_CIDS[@]}"
    fi
    submit_slot 3 > /dev/null
fi

echo "" >&2
echo "Done." >&2

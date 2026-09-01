#!/bin/bash
#
# Rerun the GPT-5.4 judges (contamination + API) on a hand-picked list of result
# directories. Throttles via an HTCondor concurrency-limit tag so we don't race
# the codex auth.json refresh-token rotation.
#
# Usage:
#   bash rerun_specific_runs.sh [options] <paths_file>
#   bash rerun_specific_runs.sh [options] -            # read paths from stdin
#
# Paths in <paths_file> may be:
#   - absolute (used as-is)
#   - relative ("<method>/<run_dir>", joined onto --results-root)
#
# Options:
#   --results-root DIR     Prefix for relative paths.
#                          Default: POST_TRAIN_BENCH_RESULTS_DIR from .env / env.
#   --tag NAME             Concurrency-limit tag (must start with "user.").
#                          Default: user.judge_rerun_$USER
#   --max-concurrent N     Max concurrent jobs on this tag (1..1000).
#                          tokens-per-job = 10000/N, so N=10 → 1000 each.
#                          Default: 10.
#   --bid N                condor_submit_bid value. Default: 100.
#   --skip-existing        Per dir, skip judges already done; downgrade mode
#                          when only one of the two _rerun files is missing.
#   --contamination-only   Run only the contamination judge (skip API judge).
#   --dry-run              Print what would be submitted; don't call condor_submit_bid.
#
# Notes:
#   - Run from the repo root (the .sub files use repo-relative script paths).
#   - logs/ must exist (mkdir -p logs once).
#   - agents/codex_non_api/auth.json must exist (ChatGPT Pro auth for GPT-5.4).

set -e

# ---------- defaults ----------
TAG="user.judge_rerun_${USER}"
MAX_CONCURRENT=10
BID=50
SKIP_EXISTING=""
CONTAM_ONLY=""
DRY_RUN=""
RESULTS_ROOT=""
PATHS_FILE=""

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-root)      RESULTS_ROOT="$2"; shift 2 ;;
        --tag)               TAG="$2"; shift 2 ;;
        --max-concurrent)    MAX_CONCURRENT="$2"; shift 2 ;;
        --bid)               BID="$2"; shift 2 ;;
        --skip-existing)     SKIP_EXISTING=1; shift ;;
        --contamination-only) CONTAM_ONLY=1; shift ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help)           sed -n '2,/^set -e/p' "$0"; exit 0 ;;
        -|*)                 PATHS_FILE="$1"; shift ;;
    esac
done

if [ -z "$PATHS_FILE" ]; then
    echo "ERROR: missing <paths_file>. See --help." >&2
    exit 1
fi

# Validate tag and concurrency
case "$TAG" in
    user.*) ;;
    *) echo "ERROR: --tag must start with 'user.' (got '$TAG')" >&2; exit 1 ;;
esac
if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [ "$MAX_CONCURRENT" -lt 1 ] || [ "$MAX_CONCURRENT" -gt 10000 ]; then
    echo "ERROR: --max-concurrent must be an integer in 1..10000 (got '$MAX_CONCURRENT')" >&2
    exit 1
fi
TOKENS_PER_JOB=$(( 10000 / MAX_CONCURRENT ))
CONCURRENCY="${TAG}:${TOKENS_PER_JOB}"

# Resolve results root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -z "$RESULTS_ROOT" ]; then
    RESULTS_ROOT="${POST_TRAIN_BENCH_RESULTS_DIR:-}"
fi
if [ -z "$RESULTS_ROOT" ] && [ -f "$ENV_FILE" ]; then
    RESULTS_ROOT="$(grep -E '^POST_TRAIN_BENCH_RESULTS_DIR=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')"
fi
if [ -z "$RESULTS_ROOT" ]; then
    echo "ERROR: --results-root not given and POST_TRAIN_BENCH_RESULTS_DIR not set" >&2
    exit 1
fi

# Choose the underlying .sub file based on whether we need judge_mode threading
if [ -n "$SKIP_EXISTING" ] || [ -n "$CONTAM_ONLY" ]; then
    SUB_FILE="src/disallowed_usage_judge/rerun_judge/rerun_judge_gpt_only.sub"
else
    SUB_FILE="src/disallowed_usage_judge/rerun_judge/rerun_judge.sub"
fi

mkdir -p logs

echo "Settings:"
echo "  results root:    $RESULTS_ROOT"
echo "  concurrency tag: $CONCURRENCY  (max $MAX_CONCURRENT concurrent)"
echo "  bid:             $BID"
echo "  sub file:        $SUB_FILE"
echo "  paths file:      $PATHS_FILE"
[ -n "$DRY_RUN" ]       && echo "  DRY RUN"
[ -n "$SKIP_EXISTING" ] && echo "  --skip-existing"
[ -n "$CONTAM_ONLY" ]   && echo "  --contamination-only"
echo

# ---------- submission loop ----------
LOG_DIR="$SCRIPT_DIR/rerun_specific_runs_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
CLUSTER_LOG="$LOG_DIR/submitted_${TS}.tsv"
[ -n "$DRY_RUN" ] || printf 'cluster_id\tmode\tresult_dir\n' > "$CLUSTER_LOG"

submitted=0; skipped=0; missing=0; no_trace=0

# Allow `-` to mean stdin
if [ "$PATHS_FILE" = "-" ]; then
    INPUT="/dev/stdin"
else
    [ -r "$PATHS_FILE" ] || { echo "ERROR: cannot read $PATHS_FILE" >&2; exit 1; }
    INPUT="$PATHS_FILE"
fi

while IFS= read -r raw; do
    # strip comments + blanks + whitespace
    line="${raw%%#*}"
    line="$(echo "$line" | xargs)"
    [ -z "$line" ] && continue

    case "$line" in
        /*) d="$line" ;;
         *) d="$RESULTS_ROOT/$line" ;;
    esac

    if [ ! -d "$d/task" ]; then
        echo "MISSING (no task/): $d"
        missing=$((missing + 1))
        continue
    fi
    if [ ! -f "$d/solve_parsed.txt" ] && [ ! -f "$d/solve_out.txt" ]; then
        echo "MISSING (no trace): $d"
        no_trace=$((no_trace + 1))
        continue
    fi

    # Resume / mode-downgrade logic
    mode=""
    if [ -n "$CONTAM_ONLY" ]; then
        mode="--gpt-contamination-only"
    fi
    if [ -n "$SKIP_EXISTING" ]; then
        has_gpt=0; has_api=0
        [ -f "$d/judgement_gpt5_4_rerun.json" ] && has_gpt=1
        [ -f "$d/judgement_api_rerun.json" ]    && has_api=1
        if [ "$has_gpt" = 1 ] && { [ "$has_api" = 1 ] || [ -n "$CONTAM_ONLY" ]; }; then
            echo "SKIP (already done): $d"
            skipped=$((skipped + 1))
            continue
        fi
        if [ -z "$CONTAM_ONLY" ]; then
            if   [ "$has_gpt" = 1 ]; then mode="--api-only"
            elif [ "$has_api" = 1 ]; then mode="--gpt-contamination-only"
            else                          mode="--gpt-only"
            fi
        fi
    fi

    # Build condor_submit_bid command
    cmd=(condor_submit_bid "$BID"
        -a "result_dir=$d"
        -a "concurrency_limits=$CONCURRENCY")
    [ -n "$mode" ] && cmd+=( -a "judge_mode=$mode" )
    cmd+=( "$SUB_FILE" )

    if [ -n "$DRY_RUN" ]; then
        printf 'WOULD SUBMIT (%s): %s\n' "${mode:-both}" "$d"
        submitted=$((submitted + 1))
        continue
    fi

    out=$("${cmd[@]}" 2>&1)
    echo "$out" | tail -2
    cluster=$(echo "$out" | grep -oE 'cluster [0-9]+' | awk '{print $2}' | tail -1)
    if [ -n "$cluster" ]; then
        printf '%s\t%s\t%s\n' "$cluster" "${mode:-both}" "$d" >> "$CLUSTER_LOG"
    fi
    submitted=$((submitted + 1))
done < "$INPUT"

echo
echo "========================================"
if [ -n "$DRY_RUN" ]; then
    echo "DRY RUN: would have submitted $submitted job(s)"
else
    echo "Submitted: $submitted"
    echo "Cluster log: $CLUSTER_LOG"
fi
echo "Skipped (already done):  $skipped"
echo "Missing (no task/ dir):  $missing"
echo "Missing (no trace file): $no_trace"
echo "========================================"

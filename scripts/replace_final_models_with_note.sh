#!/bin/bash
#
# Replace every results/.../final_model/ directory with a small placeholder
# note, reclaiming the disk space used by the trained model checkpoints.
#
# Each final_model/ is emptied and a single marker file (NOT_COPIED.txt) is
# written in its place, mirroring:
#   <results>/<experiment>/<run>/final_model/NOT_COPIED.txt
#
# DRY-RUN BY DEFAULT: prints which final_model dirs would be cleared and how
# much space would be reclaimed, but changes nothing. Pass --apply to actually
# delete the model files and write the markers.
#
# Idempotent: a final_model/ that already contains only the marker file is
# skipped.
#
# Reads POST_TRAIN_BENCH_RESULTS_DIR from .env directly (does NOT source
# set_env_vars.sh, whose module-loading block fails on nodes without tclsh).
#
# Options:
#   --apply                    Actually delete model files and write markers
#                              (default: dry-run, no changes).
#   --results-dir DIR          Override the results dir (default: from .env).
#   --only-methods REGEX       Only process method dirs whose name matches
#                              this bash-extended regex. Applied first.
#   --exclude-methods REGEX    Skip method dirs whose name matches this regex.
#                              Applied after --only-methods.
#   --preserve-benchmark NAME  Skip cell dirs starting with "<NAME>_" — e.g.
#                              --preserve-benchmark bfcl keeps bfcl checkpoints
#                              for later re-scoring under PR 63. Can be
#                              repeated: pipe-separated in a single flag or
#                              pass the flag multiple times.
#   -h, --help                 Show this help and exit.

set -euo pipefail

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

APPLY=""
RESULTS_DIR_OVERRIDE=""
ONLY_METHODS_RE=""
EXCLUDE_METHODS_RE=""
PRESERVE_BENCH_RE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --results-dir) RESULTS_DIR_OVERRIDE="${2:?--results-dir needs an argument}"; shift 2 ;;
        --only-methods) ONLY_METHODS_RE="${2:?--only-methods needs a regex}"; shift 2 ;;
        --exclude-methods) EXCLUDE_METHODS_RE="${2:?--exclude-methods needs a regex}"; shift 2 ;;
        --preserve-benchmark)
            arg="${2:?--preserve-benchmark needs a benchmark name}"
            if [ -z "$PRESERVE_BENCH_RE" ]; then
                PRESERVE_BENCH_RE="$arg"
            else
                PRESERVE_BENCH_RE="${PRESERVE_BENCH_RE}|${arg}"
            fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
else
    if [ ! -f "$ENV_FILE" ]; then
        echo "ERROR: .env file not found at $ENV_FILE" >&2
        exit 1
    fi
    RESULTS_DIR="$(grep -E '^POST_TRAIN_BENCH_RESULTS_DIR=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')"
fi

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: results dir not set (POST_TRAIN_BENCH_RESULTS_DIR missing from $ENV_FILE)" >&2
    exit 1
fi
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: results dir does not exist: $RESULTS_DIR" >&2
    exit 1
fi
# Normalise to an absolute, trailing-slash-free path for the safety guard below.
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

MARKER_NAME="NOT_COPIED.txt"
read -r -d '' MARKER_TEXT <<'EOF' || true
The original final_model directory was produced during the run but has been
removed to reclaim disk space.
This file is a placeholder marker only.
EOF

# final_model dirs live at <results>/<experiment>/<run>/final_model (depth 3).
# -maxdepth 3 both restricts the search and avoids descending into the model
# files themselves (depth 4).
mapfile -t ALL_DIRS < <(find "$RESULTS_DIR" -mindepth 3 -maxdepth 3 -type d -name final_model | sort)
ALL_TOTAL=${#ALL_DIRS[@]}

# Apply --only-methods / --exclude-methods / --preserve-benchmark filters.
# $fm = <results>/<method>/<cell>/final_model — so method = 2nd-to-last dirname
# and cell (=<benchmark>_<model>_<cid>) = last dirname.
FINAL_MODEL_DIRS=()
filtered_out=0
for fm in "${ALL_DIRS[@]}"; do
    cell="$(basename "$(dirname "$fm")")"
    method="$(basename "$(dirname "$(dirname "$fm")")")"
    if [ -n "$ONLY_METHODS_RE" ] && ! [[ $method =~ $ONLY_METHODS_RE ]]; then
        filtered_out=$((filtered_out + 1)); continue
    fi
    if [ -n "$EXCLUDE_METHODS_RE" ] && [[ $method =~ $EXCLUDE_METHODS_RE ]]; then
        filtered_out=$((filtered_out + 1)); continue
    fi
    if [ -n "$PRESERVE_BENCH_RE" ] && [[ $cell =~ ^(${PRESERVE_BENCH_RE})_ ]]; then
        filtered_out=$((filtered_out + 1)); continue
    fi
    FINAL_MODEL_DIRS+=("$fm")
done

TOTAL=${#FINAL_MODEL_DIRS[@]}
echo "Results dir: $RESULTS_DIR"
echo "Found $ALL_TOTAL final_model directories total"
[ -n "$ONLY_METHODS_RE" ]   && echo "  --only-methods:       $ONLY_METHODS_RE"
[ -n "$EXCLUDE_METHODS_RE" ] && echo "  --exclude-methods:    $EXCLUDE_METHODS_RE"
[ -n "$PRESERVE_BENCH_RE" ] && echo "  --preserve-benchmark: $PRESERVE_BENCH_RE"
if [ "$filtered_out" -gt 0 ]; then
    echo "Filters kept:                  $TOTAL"
    echo "Filters excluded (preserved):  $filtered_out"
fi
if [ -z "$APPLY" ]; then
    echo "MODE: dry-run (no changes will be made). Pass --apply to execute."
else
    echo "MODE: APPLY (model files WILL be deleted)"
fi
echo ""

processed=0
skipped=0
errors=0
declare -a TO_PROCESS=()

for fm in "${FINAL_MODEL_DIRS[@]}"; do
    # --- Safety guards: never rm -rf anything that is not a final_model dir
    #     living strictly inside the results dir. ---
    if [ "$(basename "$fm")" != "final_model" ]; then
        echo "REFUSE (not a final_model dir): $fm" >&2
        errors=$((errors + 1)); continue
    fi
    case "$fm/" in
        "$RESULTS_DIR"/*/*) : ;;  # inside results dir, at least two levels deep
        *) echo "REFUSE (outside results dir): $fm" >&2; errors=$((errors + 1)); continue ;;
    esac

    # Idempotency: skip if final_model already contains only the marker.
    shopt -s nullglob dotglob
    entries=("$fm"/*)
    shopt -u nullglob dotglob
    if [ ${#entries[@]} -eq 1 ] && [ "$(basename "${entries[0]}")" = "$MARKER_NAME" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    if [ -z "$APPLY" ]; then
        size="$(du -sh "$fm" 2>/dev/null | cut -f1)"
        echo "[dry-run] would clear (${size:-?}): $fm"
        TO_PROCESS+=("$fm")
        processed=$((processed + 1))
        continue
    fi

    # --- APPLY: replace the dir wholesale, then drop the marker. ---
    rm -rf "${fm:?refusing to rm an empty path}"
    mkdir -p "$fm"
    printf '%s\n' "$MARKER_TEXT" > "$fm/$MARKER_NAME"
    echo "cleared: $fm"
    processed=$((processed + 1))
done

echo ""
echo "================ summary ================"
echo "final_model dirs found:        $TOTAL"
if [ -z "$APPLY" ]; then
    echo "would clear:                   $processed"
else
    echo "cleared:                       $processed"
fi
echo "already markers (skipped):     $skipped"
echo "refused (safety guard):        $errors"

if [ -z "$APPLY" ] && [ ${#TO_PROCESS[@]} -gt 0 ]; then
    echo ""
    echo "Total space that would be reclaimed:"
    du -sch "${TO_PROCESS[@]}" 2>/dev/null | tail -1
    echo ""
    echo "Re-run with --apply to perform the deletion."
fi

if [ "$errors" -gt 0 ]; then
    exit 1
fi

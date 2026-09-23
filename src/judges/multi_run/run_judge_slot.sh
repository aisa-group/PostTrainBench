#!/bin/bash
#
# Run one slot of the data_contamination_judge multi-run variance experiment.
#
# Fresh gpt-5.6-terra invocation of data_contamination_judge, writing its output
# under <result_dir>/judgement_multi_runs/ with the _run<slot> suffix so it
# stays out of the way of the original judge file and the rerun file.
#
# Slot 1 is the pre-existing verdict (judgement_gpt5_4_rerun.json when present,
# else judgement_gpt5_4.json). This script writes slot 2 or slot 3. The
# per-field majority verdict is later materialized into
# judgement_multi_runs/judgement_gpt5_4_final.json on demand by
# scripts/utils.py::judgement_path().
#
# The emitted judgement is post-processed to embed a "_meta" block recording
# the judge model, codex version, slot, and timestamp — the file naming alone
# does not identify which model produced the verdict (both gpt-5.4 and
# gpt-5.6-terra can appear as "gpt5_4" in the output id historically).
#
# Usage: run_judge_slot.sh --slot <2|3> <result_dir>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$JUDGES_DIR/judge_lib.sh"

SLOT=""
RESULT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --slot) SLOT="$2"; shift 2 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) RESULT_DIR="$1"; shift ;;
    esac
done

if [ -z "$SLOT" ] || [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 --slot <2|3> <result_dir>" >&2
    exit 1
fi
if [ "$SLOT" != "2" ] && [ "$SLOT" != "3" ]; then
    echo "ERROR: --slot must be 2 or 3 (got: $SLOT)" >&2
    exit 1
fi
if [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: result directory does not exist: $RESULT_DIR" >&2
    exit 1
fi
if [ ! -d "$RESULT_DIR/task" ]; then
    echo "ERROR: no task/ subdir found in $RESULT_DIR" >&2
    exit 1
fi

JUDGE_NAME="data_contamination_judge"
load_judge_conf "$JUDGE_NAME"

# Multi-run outputs live in <result_dir>/judgement_multi_runs/ so the slot
# files never collide with the top-level per-judge files.
MULTI_DIR="$RESULT_DIR/judgement_multi_runs"
mkdir -p "$MULTI_DIR"

SUFFIX="_run${SLOT}"
JUDGEMENT_PATH="$MULTI_DIR/judgement_${JUDGE_OUTPUT_ID}${SUFFIX}.json"
OUTPUT_JSON_PATH="$MULTI_DIR/judge_output_${JUDGE_OUTPUT_ID}${SUFFIX}.json"

# Pick trace file (solve_parsed.txt preferred, solve_out.txt as fallback)
if [ -f "$RESULT_DIR/solve_parsed.txt" ]; then
    TRACE_FILE="$RESULT_DIR/solve_parsed.txt"
    TRACE_NAME="solve_parsed.txt"
elif [ -f "$RESULT_DIR/solve_out.txt" ]; then
    TRACE_FILE="$RESULT_DIR/solve_out.txt"
    TRACE_NAME="solve_out.txt"
else
    echo "ERROR: no trace file (solve_parsed.txt or solve_out.txt) found in $RESULT_DIR" >&2
    exit 1
fi

source "$JUDGES_REPO_ROOT/src/commit_utils/set_env_vars.sh"

# Parse result directory to get benchmark and model
DIRNAME=$(basename "$RESULT_DIR")
BENCHMARK=$(echo "$DIRNAME" | sed -E 's/^([^_]+)_.*/\1/')
MODEL_PART=$(echo "$DIRNAME" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')

METHOD_DIR=$(basename "$(dirname "$RESULT_DIR")")
AGENT_AND_CONFIG=$(echo "$METHOD_DIR" | sed -E 's/_[0-9]+h.*$//')
AGENT=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^([^_]+)_.*/\1/')
AGENT_CONFIG=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^[^_]+_(.*)$/\1/')

echo "Multi-run slot $SLOT for: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF | Agent: $AGENT ($AGENT_CONFIG) | Trace: $TRACE_NAME"
echo "  Judge: $JUDGE_NAME (${JUDGE_MODEL}, codex ${JUDGE_CODEX_VERSION:-container-default})"
echo "  Output: $JUDGEMENT_PATH"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

JOB_DIR="$TMP_DIR/job_dir"
JOB_TMP="$TMP_DIR/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

cp -r "$RESULT_DIR/task" "$JOB_DIR/task"
rm -f "$JOB_DIR/task/judgement.json"
cp "$TRACE_FILE" "$JOB_DIR/$TRACE_NAME"

prepare_judge_sandbox "$JOB_DIR" "$BENCHMARK" "$RESULT_DIR/final_model/config.json"
setup_judge_codex_auth "$JOB_DIR"

# Wipe stale outputs for this slot
rm -f "$JUDGEMENT_PATH" "$OUTPUT_JSON_PATH" "${OUTPUT_JSON_PATH%.json}.txt"

JUDGE_EXTRA_APPTAINER_ARGS=()
JUDGE_PROMPT=$(build_judge_prompt "$JUDGE_NAME" "$BENCHMARK" "$MODEL_HF" "$AGENT" "$AGENT_CONFIG")

run_judge_exec "$JOB_DIR" "$JOB_TMP" "$OUTPUT_JSON_PATH" "$JUDGE_PROMPT"
collect_judge_output "$JOB_DIR" "$MULTI_DIR" "$SUFFIX" 1
tag_judgement_meta "$JUDGEMENT_PATH" "$SLOT"

echo ""
echo "Multi-run slot $SLOT complete: $JUDGEMENT_PATH"

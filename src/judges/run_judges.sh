#!/bin/bash
#
# Run the reward-hacking judges on an existing result directory.
#
# Judges live in src/judges/<judge_name>/ (see judge_lib.sh); by default all
# of them run, in the order given by ALL_JUDGES:
#   1. data_contamination_judge -> judgement_gpt5_4_rerun.json
#      (contamination/disallowed_model verdict; this is the canonical
#      contamination verdict consumed downstream)
#   2. api_usage_judge          -> judgement_api_rerun.json
#      (separate `disallowed_api_usage` schema; archival only, not consumed
#      by the scoring pipeline)
#
# All outputs are always saved with the _rerun suffix so original judge
# outputs produced by src/run_task.sh are preserved.
#
# Usage: run_judges.sh [--judges <name>[,<name>...]] <result_dir>
#
# Options:
#   --judges   Comma-separated subset of judges to run (default: all).
#              e.g. --judges data_contamination_judge
#                   --judges api_usage_judge

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/judge_lib.sh"

# Parse arguments
JUDGES=()
RESULT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --judges)
            IFS=',' read -r -a JUDGES <<< "$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--judges <name>[,<name>...]] <result_dir>" >&2
            exit 1
            ;;
        *)
            RESULT_DIR="$1"
            shift
            ;;
    esac
done

if [ ${#JUDGES[@]} -eq 0 ]; then
    JUDGES=("${ALL_JUDGES[@]}")
fi

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 [--judges <name>[,<name>...]] <result_dir>" >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR" ]; then
    echo "Error: Result directory does not exist: $RESULT_DIR" >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR/task" ]; then
    echo "Error: No task directory found in $RESULT_DIR" >&2
    exit 1
fi

# Validate the requested judges early (before any expensive work).
for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"
done

# Find trace file (solve_parsed.txt preferred, solve_out.txt as fallback)
if [ -f "$RESULT_DIR/solve_parsed.txt" ]; then
    TRACE_FILE="$RESULT_DIR/solve_parsed.txt"
    TRACE_NAME="solve_parsed.txt"
elif [ -f "$RESULT_DIR/solve_out.txt" ]; then
    TRACE_FILE="$RESULT_DIR/solve_out.txt"
    TRACE_NAME="solve_out.txt"
else
    echo "Error: No trace file (solve_parsed.txt or solve_out.txt) found in $RESULT_DIR" >&2
    exit 1
fi

source "$JUDGES_REPO_ROOT/src/commit_utils/set_env_vars.sh"

# Parse result directory to get benchmark and model
# Format: {benchmark}_{provider}_{model}_{cluster_id}
DIRNAME=$(basename "$RESULT_DIR")
BENCHMARK=$(echo "$DIRNAME" | sed -E 's/^([^_]+)_.*/\1/')
MODEL_PART=$(echo "$DIRNAME" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')

# Parse the parent (method) directory to get the agent + its harness model.
# Format: {agent}_{agent_config}_{num_hours}h[_{num_gpus}gpu][{experiment_name}]
METHOD_DIR=$(basename "$(dirname "$RESULT_DIR")")
AGENT_AND_CONFIG=$(echo "$METHOD_DIR" | sed -E 's/_[0-9]+h.*$//')
AGENT=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^([^_]+)_.*/\1/')
AGENT_CONFIG=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^[^_]+_(.*)$/\1/')

echo "Running judges on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF | Agent: $AGENT ($AGENT_CONFIG) | Trace: $TRACE_NAME"
echo "  Judges: ${JUDGES[*]} (outputs suffixed with _rerun)"

# Create temporary working directory
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

JOB_DIR="$TMP_DIR/job_dir"
JOB_TMP="$TMP_DIR/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

# Copy task directory
cp -r "$RESULT_DIR/task" "$JOB_DIR/task"

# Remove any pre-existing judgement file from the task dir so stale values
# from earlier runs can't leak into this judge's output when the CLI crashes.
rm -f "$JOB_DIR/task/judgement.json"

# Copy trace file to parent directory (not task directory)
cp "$TRACE_FILE" "$JOB_DIR/$TRACE_NAME"

# Copy judge helper tooling and benchmark metadata into the sandbox.
prepare_judge_sandbox "$JOB_DIR" "$BENCHMARK" "$RESULT_DIR/final_model/config.json"

# Set up codex config + ChatGPT Pro subscription auth (JUDGE_CODEX_AUTH_SRC).
setup_judge_codex_auth "$JOB_DIR"

# Remove any pre-existing per-judge output files in the result dir for the
# judges we are about to rerun, so stale values from earlier runs can't be
# confused with fresh output when a CLI fails. Leave the skipped judges'
# files alone.
for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"
    rm -f "$RESULT_DIR/judgement_${JUDGE_OUTPUT_ID}_rerun.json"
done

JUDGE_EXTRA_APPTAINER_ARGS=()

for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"

    echo ""
    echo "========================================="
    echo "=== ${JUDGE_LABEL} ==="
    echo "========================================="

    # Clean judgement file so each judge starts fresh
    rm -f "$JOB_DIR/task/judgement.json"

    JUDGE_PROMPT=$(build_judge_prompt "$JUDGE_NAME" "$BENCHMARK" "$MODEL_HF" "$AGENT" "$AGENT_CONFIG")

    run_judge_exec "$JOB_DIR" "$JOB_TMP" "$RESULT_DIR/judge_output_${JUDGE_OUTPUT_ID}_rerun.json" "$JUDGE_PROMPT"

    collect_judge_output "$JOB_DIR" "$RESULT_DIR" "_rerun" 1
done

echo ""
echo "Judges completed successfully: ${JUDGES[*]}"

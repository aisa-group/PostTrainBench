#!/bin/bash
#
# Run the contamination judge on a result directory.
#
# Usage: run_judge.sh [--rerun] <result_dir>
#
# Options:
#   --rerun    Save results with _rerun suffix (doesn't overwrite original judgements)
#
# This script can be used to:
# 1. Rerun the judge on completed results
# 2. Run the judge manually for debugging
#
# The judge analyzes the task directory and ../solve_parsed.txt to determine:
# - Whether benchmark data was used for training (contamination)
# - Whether only the allowed base model was fine-tuned

set -e

# Parse arguments
RERUN_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --rerun)
            RERUN_MODE=true
            shift
            ;;
        *)
            RESULT_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 [--rerun] <result_dir>" >&2
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

if [ ! -f "$RESULT_DIR/solve_parsed.txt" ]; then
    echo "Error: No solve_parsed.txt found in $RESULT_DIR" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

# Parse result directory to get benchmark and model
# Format: {benchmark}_{provider}_{model}_{cluster_id}
DIRNAME=$(basename "$RESULT_DIR")
BENCHMARK=$(echo "$DIRNAME" | sed -E 's/^([^_]+)_.*/\1/')
MODEL_PART=$(echo "$DIRNAME" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')

echo "Running judge on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF"
if [ "$RERUN_MODE" = true ]; then
    echo "  Mode: rerun (will save with _rerun suffix)"
else
    echo "  Mode: normal (will overwrite existing judgements)"
fi

# Generate judge prompt
JUDGE_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt.py" \
    --benchmark-id "$BENCHMARK" \
    --model "$MODEL_HF")

# Create temporary working directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

JOB_DIR="$TMP_DIR/job_dir"
JOB_TMP="$TMP_DIR/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

# Copy task directory
cp -r "$RESULT_DIR/task" "$JOB_DIR/task"

# Copy trace file to parent directory (not task directory)
cp "$RESULT_DIR/solve_parsed.txt" "$JOB_DIR/solve_parsed.txt"

# Copy codex config
cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$JOB_DIR/"

# Run judge via codex inside apptainer, capturing output
JUDGE_OUTPUT_FILE="$TMP_DIR/judge_output.json"
apptainer exec \
    -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
    --env CODEX_API_KEY="${OPENAI_API_KEY}" \
    --env PYTHONNOUSERSITE="1" \
    --bind "${JOB_TMP}:/tmp" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
    codex --search -a never exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_FILE"

# Determine output file suffix based on mode
if [ "$RERUN_MODE" = true ]; then
    SUFFIX="_rerun"
else
    SUFFIX=""
fi

# Copy judge output to result directory and convert to human-readable format
if [ -f "$JUDGE_OUTPUT_FILE" ]; then
    cp "$JUDGE_OUTPUT_FILE" "$RESULT_DIR/judge_output${SUFFIX}.json"
    python "$REPO_ROOT/agents/codex/human_readable_trace.py" "$JUDGE_OUTPUT_FILE" -o "$RESULT_DIR/judge_output${SUFFIX}.txt"
    echo "  Judge output saved to: judge_output${SUFFIX}.json and judge_output${SUFFIX}.txt"
fi

# Copy results back
if [ -f "$JOB_DIR/task/contamination_judgement.txt" ]; then
    cp "$JOB_DIR/task/contamination_judgement.txt" "$RESULT_DIR/contamination_judgement${SUFFIX}.txt"
    echo "  Contamination: $(cat "$RESULT_DIR/contamination_judgement${SUFFIX}.txt")"
else
    echo "  Warning: contamination_judgement.txt not created by judge"
fi

if [ -f "$JOB_DIR/task/disallowed_model_judgement.txt" ]; then
    cp "$JOB_DIR/task/disallowed_model_judgement.txt" "$RESULT_DIR/disallowed_model_judgement${SUFFIX}.txt"
    echo "  Model usage: $(cat "$RESULT_DIR/disallowed_model_judgement${SUFFIX}.txt")"
else
    echo "  Warning: disallowed_model_judgement.txt not created by judge"
fi

echo "Judge completed successfully"

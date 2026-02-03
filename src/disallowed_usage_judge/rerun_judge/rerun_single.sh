#!/bin/bash
#
# Rerun the judge on a single result directory with the trace file included.
#
# Usage: rerun_single.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

# Parse result directory to get benchmark and model
DIRNAME=$(basename "$RESULT_DIR")
BENCHMARK=$(echo "$DIRNAME" | sed -E 's/^([^_]+)_.*/\1/')
MODEL_PART=$(echo "$DIRNAME" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')

# Get human-readable benchmark name from info.json
BENCHMARK_FILE="$REPO_ROOT/src/eval/tasks/${BENCHMARK}/info.json"
if [ -f "$BENCHMARK_FILE" ]; then
    BENCHMARK_NAME=$(jq -r '.benchmark' "$BENCHMARK_FILE")
else
    BENCHMARK_NAME="$BENCHMARK"
fi

echo "Rerunning judge: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK_NAME | Model: $MODEL_HF"

# Generate judge prompt
JUDGE_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt_with_trace.py" \
    --benchmark "$BENCHMARK_NAME" \
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

# Copy trace file into task directory
if [ -f "$RESULT_DIR/solve_parsed.txt" ]; then
    cp "$RESULT_DIR/solve_parsed.txt" "$JOB_DIR/task/solve_trace.txt"
    echo "  Using trace: solve_parsed.txt"
elif [ -f "$RESULT_DIR/solve_out.txt" ]; then
    cp "$RESULT_DIR/solve_out.txt" "$JOB_DIR/task/solve_trace.txt"
    echo "  Using trace: solve_out.txt"
else
    echo "  Warning: No trace file found"
fi

# Copy codex config
cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$JOB_DIR/"

# Run judge via codex inside apptainer, capturing output
JUDGE_OUTPUT_FILE="$TMP_DIR/judge_output.txt"
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
    codex --search -a never exec --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_FILE"

# Copy judge output to result directory (don't overwrite existing)
if [ -f "$JUDGE_OUTPUT_FILE" ]; then
    cp "$JUDGE_OUTPUT_FILE" "$RESULT_DIR/judge_output_rerun.txt"
    echo "  Judge output saved to: judge_output_rerun.txt"
fi

# Copy results back
if [ -f "$JOB_DIR/task/contamination_judgement.txt" ]; then
    cp "$JOB_DIR/task/contamination_judgement.txt" "$RESULT_DIR/contamination_judgement_rerun.txt"
    echo "  Contamination: $(cat "$RESULT_DIR/contamination_judgement_rerun.txt")"
fi

if [ -f "$JOB_DIR/task/disallowed_model_judgement.txt" ]; then
    cp "$JOB_DIR/task/disallowed_model_judgement.txt" "$RESULT_DIR/disallowed_model_judgement_rerun.txt"
    echo "  Model usage: $(cat "$RESULT_DIR/disallowed_model_judgement_rerun.txt")"
fi

#!/bin/bash
#
# Run the disallowed-usage judges on a result directory.
#
# Two judges, each written to its own per-judge file (no aggregation):
#   1. GPT-5.4 contamination/base-model judge (via codex CLI)
#      -> judgement_gpt5_4_rerun.json (contamination/disallowed_model verdict;
#         this is the canonical contamination verdict consumed downstream)
#   2. GPT-5.4 third-party API usage judge (via codex CLI)
#      -> judgement_api_rerun.json (separate `disallowed_api_usage` schema;
#         archival only, not consumed by the scoring pipeline)
#
# All outputs are always saved with the _rerun suffix so original judge
# outputs produced by src/run_task.sh are preserved.
#
# Usage: run_judge.sh [--gpt-only|--api-only|--gpt-contamination-only] <result_dir>
#
# Options:
#   --gpt-only      Rerun the GPT-5.4 contamination judge + the API judge.
#   --api-only      Only rerun the GPT-5.4 third-party API usage judge
#                   (skip the contamination judge).
#   --gpt-contamination-only
#                   Only rerun the GPT-5.4 contamination judge
#                   (skip the API judge).

set -e

# Parse arguments
RUN_GPT=true
RUN_API=true
MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpt-only)
            MODE="gpt-only"
            shift
            ;;
        --api-only)
            MODE="api-only"
            RUN_GPT=false
            shift
            ;;
        --gpt-contamination-only)
            MODE="gpt-contamination-only"
            RUN_API=false
            shift
            ;;
        *)
            RESULT_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 [--gpt-only|--api-only|--gpt-contamination-only] <result_dir>" >&2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

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

echo "Running judge on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF | Agent: $AGENT ($AGENT_CONFIG) | Trace: $TRACE_NAME"
case "$MODE" in
    all)                     echo "  Mode: all judges (GPT-5.4 contamination + GPT-5.4 API), outputs suffixed with _rerun" ;;
    gpt-only)                echo "  Mode: GPT-5.4 contamination + GPT-5.4 API, outputs suffixed with _rerun" ;;
    api-only)                echo "  Mode: GPT-5.4 API only (contamination judge skipped), outputs suffixed with _rerun" ;;
    gpt-contamination-only)  echo "  Mode: GPT-5.4 contamination only (API skipped, no aggregation), outputs suffixed with _rerun" ;;
esac

# Generate judge prompts
if [ "$RUN_GPT" = true ]; then
    JUDGE_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt.py" \
        --benchmark-id "$BENCHMARK" \
        --model "$MODEL_HF")
fi
if [ "$RUN_API" = true ]; then
    JUDGE_API_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt.py" \
        --benchmark-id "$BENCHMARK" \
        --model "$MODEL_HF" \
        --agent "$AGENT" \
        --agent-config "$AGENT_CONFIG" \
        --kind api)
fi

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
cp "$REPO_ROOT/src/disallowed_usage_judge/judge_tools/contamination_check.py" "$JOB_DIR/contamination_check.py"
cp "$REPO_ROOT/src/disallowed_usage_judge/judge_tools/model_identity_check.py" "$JOB_DIR/model_identity_check.py"
cp -r "$REPO_ROOT/src/disallowed_usage_judge/judge_tools/reference_configs" "$JOB_DIR/reference_configs"

# Expose final_model/config.json to the judge as ../final_model_config.json so
# model_identity_check.py can compare it against the reference. Only the
# config.json is needed for the architecture-identity check, not the weights.
if [ -f "$RESULT_DIR/final_model/config.json" ]; then
    cp "$RESULT_DIR/final_model/config.json" "$JOB_DIR/final_model_config.json"
fi

if [ -f "$REPO_ROOT/src/eval/tasks/$BENCHMARK/test_data.json" ]; then
    cp "$REPO_ROOT/src/eval/tasks/$BENCHMARK/test_data.json" "$JOB_DIR/test_data.json"
fi

# Set up codex config + ChatGPT Pro subscription auth.
# auth.json itself is bind-mounted from the shared location at apptainer exec
# time so codex can write the rotated refresh token back to the source and the
# next job picks it up instead of reusing a stale single-use refresh token.
CODEX_AUTH_SRC="$REPO_ROOT/agents/codex_non_api/auth.json"
cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$JOB_DIR/"
if [ ! -f "$CODEX_AUTH_SRC" ]; then
    echo "ERROR: agents/codex_non_api/auth.json not found — GPT-5.4 judges need subscription auth" >&2
    exit 1
fi
# Touch a placeholder so apptainer has something to bind onto inside .codex/.
: > "$JOB_DIR/.codex/auth.json"
if ! grep -q "forced_login_method" "$JOB_DIR/.codex/config.toml" 2>/dev/null; then
    printf '\nforced_login_method = "chatgpt"\n' >> "$JOB_DIR/.codex/config.toml"
fi

# Remove any pre-existing per-judge output files in the result dir for the
# judges we are about to rerun, so stale values from earlier runs can't be
# confused with fresh output when a CLI fails. Leave the skipped judge's
# files alone so aggregation can still use them.
if [ "$RUN_GPT" = true ]; then
    rm -f "$RESULT_DIR/judgement_gpt5_4_rerun.json"
fi
if [ "$RUN_API" = true ]; then
    rm -f "$RESULT_DIR/judgement_api_rerun.json"
fi

# ============================================================
# Judge 1: GPT-5.4 via codex CLI
# ============================================================
if [ "$RUN_GPT" = true ]; then
    echo ""
    echo "========================================="
    echo "=== Judge 1: GPT-5.4 (codex CLI) ==="
    echo "========================================="

    JUDGE_OUTPUT_GPT="$RESULT_DIR/judge_output_gpt5_4_rerun.json"
    apptainer exec \
        --containall \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env CODEX_API_KEY="" \
        --env OPENAI_API_KEY="" \
        --env PYTHONNOUSERSITE="1" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${CODEX_AUTH_SRC}:/home/ben/.codex/auth.json" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/gpt_5_5.sif" \
        codex --search -a never exec --json -c model_reasoning_summary=detailed -c model_reasoning_effort=xhigh --skip-git-repo-check --yolo --model "gpt-5.4" "$JUDGE_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_GPT"

    # Decode the codex JSON trace into a human-readable text report
    python "$REPO_ROOT/src/trace_parsing/parse_trace.py" --agent codex "$JUDGE_OUTPUT_GPT" -o "$RESULT_DIR/judge_output_gpt5_4_rerun.txt"
    echo "  GPT-5.4 judge output saved"

    # Save GPT-5.4 judgement JSON with model-specific suffix
    if [ -f "$JOB_DIR/task/judgement.json" ]; then
        cp "$JOB_DIR/task/judgement.json" "$RESULT_DIR/judgement_gpt5_4_rerun.json"
        echo "  GPT-5.4 judgement: $(cat "$RESULT_DIR/judgement_gpt5_4_rerun.json")"
    else
        echo "ERROR: judgement.json not created by GPT-5.4 judge (see $RESULT_DIR/judge_output_gpt5_4_rerun.txt)" >&2
        exit 1
    fi

    # Clean judgement file so the next judge starts fresh
    rm -f "$JOB_DIR/task/judgement.json"
fi

# ============================================================
# Judge 2: third-party API usage (GPT-5.4 only)
# ============================================================
if [ "$RUN_API" = true ]; then
    echo ""
    echo "========================================="
    echo "=== Judge 2: third-party API usage (GPT-5.4) ==="
    echo "========================================="

    JUDGE_OUTPUT_API="$RESULT_DIR/judge_output_api_rerun.json"
    apptainer exec \
        --containall \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env CODEX_API_KEY="" \
        --env OPENAI_API_KEY="" \
        --env PYTHONNOUSERSITE="1" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${CODEX_AUTH_SRC}:/home/ben/.codex/auth.json" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/gpt_5_5.sif" \
        codex --search -a never exec --json -c model_reasoning_summary=detailed -c model_reasoning_effort=xhigh --skip-git-repo-check --yolo --model "gpt-5.4" "$JUDGE_API_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_API"

    python "$REPO_ROOT/src/trace_parsing/parse_trace.py" --agent codex "$JUDGE_OUTPUT_API" -o "$RESULT_DIR/judge_output_api_rerun.txt"
    echo "  API judge output saved"

    if [ -f "$JOB_DIR/task/judgement.json" ]; then
        cp "$JOB_DIR/task/judgement.json" "$RESULT_DIR/judgement_api_rerun.json"
        echo "  API judgement: $(cat "$RESULT_DIR/judgement_api_rerun.json")"
    else
        echo "ERROR: judgement.json not created by API judge (see $RESULT_DIR/judge_output_api_rerun.txt)" >&2
        exit 1
    fi
fi

echo ""
echo "Judge completed successfully (mode: $MODE)"

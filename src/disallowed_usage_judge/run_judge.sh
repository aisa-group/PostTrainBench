#!/bin/bash
#
# Run the disallowed-usage judges on a result directory.
#
# Three judges:
#   1. GPT-5.4 contamination/base-model judge (via codex CLI)
#   2. DeepSeek V4 Flash Free contamination/base-model judge (via opencode CLI)
#   3. GPT-5.4 third-party API usage judge (via codex CLI)
#
# Results from judges 1 and 2 are aggregated into judge_result_rerun.json:
# if either flags an issue, the overall result is flagged.
#
# Judge 3 has a different schema (`disallowed_api_usage` instead of
# `contamination`/`disallowed_model`) and is written to its own files
# `judgement_api_rerun.json` and `judge_output_api_rerun.{json,txt}` — it is
# NOT folded into judge_result_rerun.json.
#
# All outputs are always saved with the _rerun suffix so original judge
# outputs produced by src/run_task.sh are preserved.
#
# Usage: run_judge.sh [--gpt-only|--deepseek-only|--api-only] <result_dir>
#
# Options:
#   --gpt-only      Only rerun the GPT-5.4 contamination judge + the API judge
#                   (skip DeepSeek); aggregation still runs using the existing
#                   DeepSeek _rerun output if present.
#   --deepseek-only Only rerun the DeepSeek V4 Flash Free contamination judge
#                   (skip both GPT-based judges); aggregation still runs using
#                   the existing GPT _rerun output if present.
#   --api-only      Only rerun the GPT-5.4 third-party API usage judge
#                   (skip both contamination judges and skip aggregation).

set -e

# Parse arguments
RUN_GPT=true
RUN_DEEPSEEK=true
RUN_API=true
MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpt-only)
            MODE="gpt-only"
            RUN_DEEPSEEK=false
            shift
            ;;
        --deepseek-only)
            MODE="deepseek-only"
            RUN_GPT=false
            RUN_API=false
            shift
            ;;
        --api-only)
            MODE="api-only"
            RUN_GPT=false
            RUN_DEEPSEEK=false
            shift
            ;;
        *)
            RESULT_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 [--gpt-only|--deepseek-only|--api-only] <result_dir>" >&2
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

echo "Running judge on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF | Trace: $TRACE_NAME"
case "$MODE" in
    all)           echo "  Mode: all judges (GPT-5.4 contamination + DeepSeek V4 Flash Free contamination + GPT-5.4 API), outputs suffixed with _rerun" ;;
    gpt-only)      echo "  Mode: GPT-5.4 contamination + GPT-5.4 API (DeepSeek skipped), outputs suffixed with _rerun" ;;
    deepseek-only) echo "  Mode: DeepSeek V4 Flash Free contamination only (GPT-based judges skipped), outputs suffixed with _rerun" ;;
    api-only)      echo "  Mode: GPT-5.4 API only (contamination judges skipped), outputs suffixed with _rerun" ;;
esac

# Generate judge prompts
if [ "$RUN_GPT" = true ] || [ "$RUN_DEEPSEEK" = true ]; then
    JUDGE_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt.py" \
        --benchmark-id "$BENCHMARK" \
        --model "$MODEL_HF")
fi
if [ "$RUN_API" = true ]; then
    JUDGE_API_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt.py" \
        --benchmark-id "$BENCHMARK" \
        --model "$MODEL_HF" \
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

# Set up codex config + ChatGPT Pro subscription auth (only when a GPT-based judge runs).
# auth.json itself is bind-mounted from the shared location at apptainer exec
# time so codex can write the rotated refresh token back to the source and the
# next job picks it up instead of reusing a stale single-use refresh token.
CODEX_AUTH_SRC="$REPO_ROOT/agents/codex_non_api/auth.json"
if [ "$RUN_GPT" = true ] || [ "$RUN_API" = true ]; then
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
fi

# Ensure OPENCODE_API_KEY is available when the DeepSeek judge runs. opencode
# reads it via {env:OPENCODE_API_KEY} from opencode.json (see solve.sh).
if [ "$RUN_DEEPSEEK" = true ]; then
    if [ -z "${OPENCODE_API_KEY:-}" ]; then
        echo "ERROR: OPENCODE_API_KEY is not set — DeepSeek V4 Flash Free judge needs an opencode API key" >&2
        exit 1
    fi
fi

# Remove any pre-existing per-judge output files in the result dir for the
# judges we are about to rerun, so stale values from earlier runs can't be
# confused with fresh output when a CLI fails. Leave the skipped judge's
# files alone so aggregation can still use them.
if [ "$RUN_GPT" = true ]; then
    rm -f "$RESULT_DIR/judgement_gpt5_4_rerun.json"
fi
if [ "$RUN_DEEPSEEK" = true ]; then
    rm -f "$RESULT_DIR/judgement_deepseek_rerun.json"
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
        -c \
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
    python "$REPO_ROOT/agents/codex/human_readable_trace.py" "$JUDGE_OUTPUT_GPT" -o "$RESULT_DIR/judge_output_gpt5_4_rerun.txt"
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
# Judge 2: DeepSeek V4 Flash Free via opencode CLI
# ============================================================
if [ "$RUN_DEEPSEEK" = true ]; then
    echo ""
    echo "========================================="
    echo "=== Judge 2: DeepSeek V4 Flash Free (opencode CLI) ==="
    echo "========================================="

    # opencode requires opencode.json in the working directory for provider
    # config and auto-approval. Write it into the task pwd that apptainer uses.
    cat > "$JOB_DIR/task/opencode.json" << 'OPENCODE_EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "opencode": {
      "options": {
        "apiKey": "{env:OPENCODE_API_KEY}"
      }
    }
  }
}
OPENCODE_EOF

    JUDGE_OUTPUT_DEEPSEEK="$RESULT_DIR/judge_output_deepseek_rerun.json"
    apptainer exec \
        -c \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env OPENCODE_API_KEY="${OPENCODE_API_KEY}" \
        --env PYTHONNOUSERSITE="1" \
        --bind "${JOB_TMP}:/tmp" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/gpt_5_5.sif" \
        opencode run --model "opencode/deepseek-v4-flash-free" --format json "$JUDGE_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_DEEPSEEK"

    # Decode the opencode JSONL trace into a human-readable text report
    python "$REPO_ROOT/agents/opencode/human_readable_trace.py" "$JUDGE_OUTPUT_DEEPSEEK" -o "$RESULT_DIR/judge_output_deepseek_rerun.txt"
    echo "  DeepSeek V4 Flash Free judge output saved"

    # Save DeepSeek judgement JSON with model-specific suffix
    if [ -f "$JOB_DIR/task/judgement.json" ]; then
        cp "$JOB_DIR/task/judgement.json" "$RESULT_DIR/judgement_deepseek_rerun.json"
        echo "  DeepSeek V4 Flash Free judgement: $(cat "$RESULT_DIR/judgement_deepseek_rerun.json")"
    else
        echo "ERROR: judgement.json not created by DeepSeek V4 Flash Free judge (see $RESULT_DIR/judge_output_deepseek_rerun.txt)" >&2
        exit 1
    fi

    # Clean judgement file so the next judge starts fresh
    rm -f "$JOB_DIR/task/judgement.json"
fi

# ============================================================
# Judge 3: third-party API usage (GPT-5.4 only)
# ============================================================
if [ "$RUN_API" = true ]; then
    echo ""
    echo "========================================="
    echo "=== Judge 3: third-party API usage (GPT-5.4) ==="
    echo "========================================="

    JUDGE_OUTPUT_API="$RESULT_DIR/judge_output_api_rerun.json"
    apptainer exec \
        -c \
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

    python "$REPO_ROOT/agents/codex/human_readable_trace.py" "$JUDGE_OUTPUT_API" -o "$RESULT_DIR/judge_output_api_rerun.txt"
    echo "  API judge output saved"

    if [ -f "$JOB_DIR/task/judgement.json" ]; then
        cp "$JOB_DIR/task/judgement.json" "$RESULT_DIR/judgement_api_rerun.json"
        echo "  API judgement: $(cat "$RESULT_DIR/judgement_api_rerun.json")"
    else
        echo "ERROR: judgement.json not created by API judge (see $RESULT_DIR/judge_output_api_rerun.txt)" >&2
        exit 1
    fi
fi

# ============================================================
# Aggregate: contamination judges OR'd together, API verdict folded in.
# Always re-aggregate so judge_result_rerun.json reflects the latest run.
# All three per-judge rerun files must exist; aggregate_judgement.py fails
# loud if any is missing.
# ============================================================
echo ""
echo "========================================="
echo "=== Aggregating Judge Results ==="
echo "========================================="

python "$SCRIPT_DIR/aggregate_judgement.py" \
    --judge "gpt5_4=$RESULT_DIR/judgement_gpt5_4_rerun.json" \
    --judge "deepseek=$RESULT_DIR/judgement_deepseek_rerun.json" \
    --judge "api=$RESULT_DIR/judgement_api_rerun.json" \
    --output "$RESULT_DIR/judge_result_rerun.json"

echo ""
echo "Judge completed successfully (mode: $MODE)"

#!/bin/bash
#
# Run the contamination judge on a result directory using two models:
#   1. GPT-5.4 (via codex CLI)
#   2. Claude Sonnet 4.6 (via claude CLI)
#
# Results are aggregated: if either model flags an issue, the overall result is flagged.
# All outputs are always saved with the _rerun suffix so original judge outputs
# produced by src/run_task.sh are preserved.
#
# Usage: run_judge.sh [--gpt-only|--sonnet-only] <result_dir>
#
# Options:
#   --gpt-only     Only rerun the GPT-5.4 judge (skip Sonnet); aggregation still runs
#                  using the existing Sonnet _rerun output if present.
#   --sonnet-only  Only rerun the Sonnet 4.6 judge (skip GPT); aggregation still runs
#                  using the existing GPT _rerun output if present.
#
# The judge analyzes the task directory and ../solve_parsed.txt to determine:
# - Whether benchmark data was used for training (contamination)
# - Whether only the allowed base model was fine-tuned

set -e

# Parse arguments
RUN_GPT=true
RUN_SONNET=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpt-only)
            RUN_SONNET=false
            shift
            ;;
        --sonnet-only)
            RUN_GPT=false
            shift
            ;;
        *)
            RESULT_DIR="$1"
            shift
            ;;
    esac
done

if [ "$RUN_GPT" = false ] && [ "$RUN_SONNET" = false ]; then
    echo "Error: cannot combine --gpt-only and --sonnet-only" >&2
    exit 1
fi

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 [--gpt-only|--sonnet-only] <result_dir>" >&2
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
if [ "$RUN_GPT" = true ] && [ "$RUN_SONNET" = true ]; then
    echo "  Mode: both judges (GPT-5.4 + Sonnet 4.6), outputs suffixed with _rerun"
elif [ "$RUN_GPT" = true ]; then
    echo "  Mode: GPT-5.4 only (Sonnet skipped), outputs suffixed with _rerun"
else
    echo "  Mode: Sonnet 4.6 only (GPT skipped), outputs suffixed with _rerun"
fi

# Generate judge prompt
JUDGE_PROMPT=$(python "$SCRIPT_DIR/get_judge_prompt.py" \
    --benchmark-id "$BENCHMARK" \
    --model "$MODEL_HF")

# Create temporary working directory
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

JOB_DIR="$TMP_DIR/job_dir"
JOB_TMP="$TMP_DIR/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

# Copy task directory
cp -r "$RESULT_DIR/task" "$JOB_DIR/task"

# Remove any pre-existing judgement files from the task dir so stale values
# from earlier runs can't leak into this judge's output when the CLI crashes.
rm -f "$JOB_DIR/task/contamination_judgement.txt"
rm -f "$JOB_DIR/task/disallowed_model_judgement.txt"

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

# Set up codex config + ChatGPT Pro subscription auth (only when GPT judge runs)
if [ "$RUN_GPT" = true ]; then
    cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$JOB_DIR/"
    if [ -f "$REPO_ROOT/agents/codex_non_api/auth.json" ]; then
        cp "$REPO_ROOT/agents/codex_non_api/auth.json" "$JOB_DIR/.codex/auth.json"
    else
        echo "ERROR: agents/codex_non_api/auth.json not found — GPT-5.4 judge needs subscription auth" >&2
        exit 1
    fi
    if ! grep -q "forced_login_method" "$JOB_DIR/.codex/config.toml" 2>/dev/null; then
        printf '\nforced_login_method = "chatgpt"\n' >> "$JOB_DIR/.codex/config.toml"
    fi
fi

# Load Claude Max subscription OAuth token for claude judge (only when Sonnet runs)
JUDGE_OAUTH_TOKEN=""
if [ "$RUN_SONNET" = true ]; then
    if [ -f "$REPO_ROOT/agents/claude_non_api/oauth_token" ]; then
        JUDGE_OAUTH_TOKEN="$(cat "$REPO_ROOT/agents/claude_non_api/oauth_token")"
    else
        echo "ERROR: agents/claude_non_api/oauth_token not found — Sonnet 4.6 judge needs subscription auth" >&2
        exit 1
    fi
fi

# Remove any pre-existing per-judge output files in the result dir for the
# judges we are about to rerun, so stale values from earlier runs can't be
# confused with fresh output when a CLI fails. Leave the skipped judge's
# files alone so aggregation can still use them.
if [ "$RUN_GPT" = true ]; then
    rm -f "$RESULT_DIR/contamination_judgement_gpt5_4_rerun.txt"
    rm -f "$RESULT_DIR/disallowed_model_judgement_gpt5_4_rerun.txt"
fi
if [ "$RUN_SONNET" = true ]; then
    rm -f "$RESULT_DIR/contamination_judgement_sonnet4_6_rerun.txt"
    rm -f "$RESULT_DIR/disallowed_model_judgement_sonnet4_6_rerun.txt"
fi

# ============================================================
# Judge 1: GPT-5.4 via codex CLI
# ============================================================
if [ "$RUN_GPT" = true ]; then
    echo ""
    echo "========================================="
    echo "=== Judge 1: GPT-5.4 (codex CLI) ==="
    echo "========================================="

    JUDGE_OUTPUT_GPT="$TMP_DIR/judge_output_gpt5_4.json"
    apptainer exec \
        -c \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env CODEX_API_KEY="" \
        --env OPENAI_API_KEY="" \
        --env PYTHONNOUSERSITE="1" \
        --bind "${JOB_TMP}:/tmp" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/opus_4_6_codex_5_3.sif" \
        codex --search -a never exec -c model_reasoning_summary=detailed -c model_reasoning_effort=xhigh --skip-git-repo-check --yolo --model "gpt-5.4" "$JUDGE_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_GPT"

    # Save GPT-5.4 judge output
    if [ -f "$JUDGE_OUTPUT_GPT" ]; then
        cp "$JUDGE_OUTPUT_GPT" "$RESULT_DIR/judge_output_gpt5_4_rerun.txt"
        echo "  GPT-5.4 judge output saved"
    fi

    # Save GPT-5.4 judgements with model-specific suffix
    if [ -f "$JOB_DIR/task/contamination_judgement.txt" ]; then
        cp "$JOB_DIR/task/contamination_judgement.txt" "$RESULT_DIR/contamination_judgement_gpt5_4_rerun.txt"
        echo "  GPT-5.4 Contamination: $(cat "$RESULT_DIR/contamination_judgement_gpt5_4_rerun.txt")"
    else
        echo "  Warning: contamination_judgement.txt not created by GPT-5.4 judge"
    fi

    if [ -f "$JOB_DIR/task/disallowed_model_judgement.txt" ]; then
        cp "$JOB_DIR/task/disallowed_model_judgement.txt" "$RESULT_DIR/disallowed_model_judgement_gpt5_4_rerun.txt"
        echo "  GPT-5.4 Model usage: $(cat "$RESULT_DIR/disallowed_model_judgement_gpt5_4_rerun.txt")"
    else
        echo "  Warning: disallowed_model_judgement.txt not created by GPT-5.4 judge"
    fi

    # Clean judgement files so the next judge starts fresh
    rm -f "$JOB_DIR/task/contamination_judgement.txt"
    rm -f "$JOB_DIR/task/disallowed_model_judgement.txt"
fi

# ============================================================
# Judge 2: Claude Sonnet 4.6 via claude CLI
# ============================================================
if [ "$RUN_SONNET" = true ]; then
    echo ""
    echo "========================================="
    echo "=== Judge 2: Claude Sonnet 4.6 ==="
    echo "========================================="

    JUDGE_OUTPUT_SONNET="$RESULT_DIR/judge_output_sonnet4_6_rerun.txt"
    apptainer exec \
        -c \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env ANTHROPIC_API_KEY="" \
        --env CLAUDE_CODE_OAUTH_TOKEN="${JUDGE_OAUTH_TOKEN}" \
        --env PYTHONNOUSERSITE="1" \
        --env CLAUDE_CODE_EFFORT_LEVEL="high" \
        --bind "${JOB_TMP}:/tmp" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/opus_4_6_codex_5_3.sif" \
        claude --print --verbose --model claude-sonnet-4-6 --output-format stream-json --dangerously-skip-permissions "$JUDGE_PROMPT" 2>&1 | tee "$JUDGE_OUTPUT_SONNET"

    # Save Sonnet 4.6 judge output
    if [ -f "$JUDGE_OUTPUT_SONNET" ]; then
        cp "$JUDGE_OUTPUT_SONNET" "$RESULT_DIR/judge_output_sonnet4_6_rerun.json"
        python "$REPO_ROOT/agents/claude/human_readable_trace.py" "$JUDGE_OUTPUT_SONNET" -o "$RESULT_DIR/judge_output_sonnet4_6_rerun.txt"
        echo "  Sonnet 4.6 judge output saved"
    fi

    # Save Sonnet 4.6 judgements with model-specific suffix
    if [ -f "$JOB_DIR/task/contamination_judgement.txt" ]; then
        cp "$JOB_DIR/task/contamination_judgement.txt" "$RESULT_DIR/contamination_judgement_sonnet4_6_rerun.txt"
        echo "  Sonnet 4.6 Contamination: $(cat "$RESULT_DIR/contamination_judgement_sonnet4_6_rerun.txt")"
    else
        echo "  Warning: contamination_judgement.txt not created by Sonnet 4.6 judge"
    fi

    if [ -f "$JOB_DIR/task/disallowed_model_judgement.txt" ]; then
        cp "$JOB_DIR/task/disallowed_model_judgement.txt" "$RESULT_DIR/disallowed_model_judgement_sonnet4_6_rerun.txt"
        echo "  Sonnet 4.6 Model usage: $(cat "$RESULT_DIR/disallowed_model_judgement_sonnet4_6_rerun.txt")"
    else
        echo "  Warning: disallowed_model_judgement.txt not created by Sonnet 4.6 judge"
    fi
fi

# ============================================================
# Aggregate results: flag if either judge flags
# ============================================================
echo ""
echo "========================================="
echo "=== Aggregating Judge Results ==="
echo "========================================="

# Aggregate contamination judgement
CONTAM_GPT=$(cat "$RESULT_DIR/contamination_judgement_gpt5_4_rerun.txt" 2>/dev/null || echo "")
CONTAM_SONNET=$(cat "$RESULT_DIR/contamination_judgement_sonnet4_6_rerun.txt" 2>/dev/null || echo "")

if echo "$CONTAM_GPT" | grep -qix "contamination detected" || echo "$CONTAM_SONNET" | grep -qix "contamination detected"; then
    echo "contamination detected" > "$RESULT_DIR/contamination_judgement_rerun.txt"
else
    echo "no contamination detected" > "$RESULT_DIR/contamination_judgement_rerun.txt"
fi
echo "  Aggregated Contamination: $(cat "$RESULT_DIR/contamination_judgement_rerun.txt")"

# Aggregate disallowed model judgement
MODEL_GPT=$(cat "$RESULT_DIR/disallowed_model_judgement_gpt5_4_rerun.txt" 2>/dev/null || echo "")
MODEL_SONNET=$(cat "$RESULT_DIR/disallowed_model_judgement_sonnet4_6_rerun.txt" 2>/dev/null || echo "")

if echo "$MODEL_GPT" | grep -qix "disallowed use detected" || echo "$MODEL_SONNET" | grep -qix "disallowed use detected"; then
    echo "disallowed use detected" > "$RESULT_DIR/disallowed_model_judgement_rerun.txt"
else
    echo "only allowed use detected" > "$RESULT_DIR/disallowed_model_judgement_rerun.txt"
fi
echo "  Aggregated Model usage: $(cat "$RESULT_DIR/disallowed_model_judgement_rerun.txt")"

echo ""
echo "Judge completed successfully (GPT-5.4 + Sonnet 4.6)"

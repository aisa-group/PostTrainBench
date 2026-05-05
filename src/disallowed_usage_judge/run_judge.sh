#!/bin/bash
#
# Re-run the contamination/disallowed-model judge on an existing result directory.
#
# Mirrors the judge invocation in src/run_task.sh (single GPT-5.1-Codex via the
# CODEX_API_KEY) but operates on a result directory that already exists,
# without re-running the agent or eval. Outputs are written with a `_rerun`
# suffix so the originals from the run are preserved:
#   - contamination_judgement_rerun.txt
#   - disallowed_model_judgement_rerun.txt
#   - judge_output_rerun.json / judge_output_rerun.txt
#
# Usage: run_judge.sh <result_dir>

set -e

RESULT_DIR="$1"

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>" >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR" ]; then
    echo "Error: result directory does not exist: $RESULT_DIR" >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR/task" ]; then
    echo "Error: no task directory found in $RESULT_DIR" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/src/commit_utils/set_env_vars.sh"

# The original run_task.sh swaps OPENAI_API_KEY into CODEX_API_KEY before the
# judge runs. Mirror that here so the script works whether the user has only
# OPENAI_API_KEY set, only CODEX_API_KEY set, or both.
if [ -z "${CODEX_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    export CODEX_API_KEY="${OPENAI_API_KEY}"
fi

if [ -z "${CODEX_API_KEY:-}" ]; then
    echo "Error: neither CODEX_API_KEY nor OPENAI_API_KEY is set" >&2
    exit 1
fi

# Parse benchmark/model from result directory name.
# Format: {benchmark}_{provider}_{model}_{cluster_id}
DIRNAME=$(basename "$RESULT_DIR")
BENCHMARK=$(echo "$DIRNAME" | sed -E 's/^([^_]+)_.*/\1/')
MODEL_PART=$(echo "$DIRNAME" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')

echo "Running judge on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF"

JUDGE_TASK=$(python "$REPO_ROOT/src/disallowed_usage_judge/get_judge_prompt.py" \
    --benchmark "$BENCHMARK" \
    --model "$MODEL_HF")

# Sandbox: copy the task dir so any judgement files written by the judge land
# in our temp dir, not the canonical result dir, until we explicitly copy them.
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
JOB_DIR="$TMP_DIR/job_dir"
JOB_TMP="$TMP_DIR/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

cp -r "$RESULT_DIR/task" "$JOB_DIR/task"

# Strip stale judgement files from the sandbox so a CLI failure can't leak old
# values into this judge's output.
rm -f "$JOB_DIR/task/contamination_judgement.txt"
rm -f "$JOB_DIR/task/disallowed_model_judgement.txt"

# Reset codex config (matches src/run_task.sh:196) so any agent-specific
# settings like model_reasoning_effort don't leak into the judge.
cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$JOB_DIR/"

# Strip any pre-existing _rerun outputs so a CLI crash can't leave stale data.
rm -f "$RESULT_DIR/contamination_judgement_rerun.txt"
rm -f "$RESULT_DIR/disallowed_model_judgement_rerun.txt"
rm -f "$RESULT_DIR/judge_output_rerun.json"
rm -f "$RESULT_DIR/judge_output_rerun.txt"

JUDGE_OUTPUT_JSON="$RESULT_DIR/judge_output_rerun.json"

apptainer exec \
    --nv \
    -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env CODEX_API_KEY="${CODEX_API_KEY}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --bind "${JOB_TMP}:/tmp" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
    codex --search -a never exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_TASK" 2>&1 | tee "$JUDGE_OUTPUT_JSON"

# Convert JSON output to human-readable trace.
if [ -f "$JUDGE_OUTPUT_JSON" ]; then
    python "$REPO_ROOT/agents/codex/human_readable_trace.py" "$JUDGE_OUTPUT_JSON" -o "$RESULT_DIR/judge_output_rerun.txt"
fi

# Copy out judgement files (if the judge produced them).
if [ -f "$JOB_DIR/task/contamination_judgement.txt" ]; then
    cp "$JOB_DIR/task/contamination_judgement.txt" "$RESULT_DIR/contamination_judgement_rerun.txt"
    echo "  Contamination: $(cat "$RESULT_DIR/contamination_judgement_rerun.txt")"
else
    echo "  Warning: contamination_judgement.txt not produced by judge"
fi

if [ -f "$JOB_DIR/task/disallowed_model_judgement.txt" ]; then
    cp "$JOB_DIR/task/disallowed_model_judgement.txt" "$RESULT_DIR/disallowed_model_judgement_rerun.txt"
    echo "  Model usage: $(cat "$RESULT_DIR/disallowed_model_judgement_rerun.txt")"
else
    echo "  Warning: disallowed_model_judgement.txt not produced by judge"
fi

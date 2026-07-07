#!/bin/bash
# Modal port of src/run_task.sh's contamination-judge phase (lines 190-219).
# Runs in a fresh container; app.py has already reconstructed the job dir
# (task/ and solve_parsed.txt from the results volume).
#
# Inputs via environment (set by app.py:run_judge):
#   EVALUATION_TASK MODEL_TO_TRAIN LOCAL_EVAL_DIR REPO JOB_DIR HF_RO
#   POST_TRAIN_BENCH_PROMPT plus API keys from the secret.

source "${REPO}/src/modal_adapter/hf_cache_lib.sh"

mkdir -p "${LOCAL_EVAL_DIR}"
cd "${REPO}"

echo "========================================="
echo "=== RUNNING CONTAMINATION JUDGE ==="
echo "========================================="

export HF_HOME_NEW="${JOB_DIR}/hf_cache"
hf_symlink_farm "${HF_RO}" "${HF_HOME_NEW}"

# The judge authenticates with the OpenAI key (run_task.sh:64-65).
export CODEX_API_KEY="${OPENAI_API_KEY:-}"

BENCHMARK=$(cat "src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
JUDGE_TASK=$(python src/disallowed_usage_judge/get_judge_prompt.py --benchmark "${BENCHMARK}" --model "${MODEL_TO_TRAIN}")

# Reset codex config to prevent agent-specific settings (e.g. model_reasoning_effort)
# from leaking into the judge, which uses a different model
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

cd "${JOB_DIR}/task"
env $(modal_env_unsets) \
    HOME="${JOB_DIR}" \
    PATH="/root/.local/bin:${JOB_DIR}/.local/bin:${PATH}" \
    HF_HOME="${HF_HOME_NEW}" \
    CODEX_API_KEY="${CODEX_API_KEY}" \
    VLLM_API_KEY="inspectai" \
    PYTHONNOUSERSITE="1" \
    codex --search -a never exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_TASK" 2>&1 | tee "${LOCAL_EVAL_DIR}/judge_output.json"

cd "${REPO}"

# Convert judge JSON output to human-readable format
python agents/codex/human_readable_trace.py "${LOCAL_EVAL_DIR}/judge_output.json" -o "${LOCAL_EVAL_DIR}/judge_output.txt"

cp "${JOB_DIR}/task/contamination_judgement.txt" "${LOCAL_EVAL_DIR}/contamination_judgement.txt" \
    || echo "Warning: judge produced no contamination_judgement.txt"
cp "${JOB_DIR}/task/disallowed_model_judgement.txt" "${LOCAL_EVAL_DIR}/disallowed_model_judgement.txt" \
    || echo "Warning: judge produced no disallowed_model_judgement.txt"

echo "judge phase done"

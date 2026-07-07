#!/bin/bash
# Modal port of src/run_task.sh's evaluation phase (lines 243-363): the
# max-tokens retry ladder over the per-task evaluate.py. Differences:
#   - runs in its own fresh container (vllm_debug image), so the
#     `nvidia-smi ... kill -9` GPU-process cleanup between attempts is gone;
#   - the repo's src tree is copied to a writable /work (the /repo mount is
#     read-only and inspect_ai writes logs/ into the task cwd);
#   - final_model is at /work/final_model (copied from the results volume by
#     app.py:run_eval).
#
# Inputs via environment (set by app.py:run_eval):
#   EVALUATION_TASK LOCAL_EVAL_DIR REPO HF_RO plus API keys from the secret.

source "${REPO}/src/modal_adapter/hf_cache_lib.sh"

mkdir -p "${LOCAL_EVAL_DIR}"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

# set openai api keys appropriately (run_task.sh:64-69): the eval container
# only receives OPENAI_API_KEY for the model-graded tasks.
CODEX_API_KEY="${OPENAI_API_KEY:-}"
unset OPENAI_API_KEY
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

mkdir -p /work
cp -r "${REPO}/src" /work/src

export TMP_HF_CACHE="/tmp/hf_cache_90afd0"
hf_symlink_farm "${HF_RO}" "${TMP_HF_CACHE}"

export EVAL_COUNTER=0
export LOCAL_EVAL_DIR
export EVALUATION_TASK

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    sleep 5
    (
        cd "/work/src/eval/tasks/${EVALUATION_TASK}" && \
        env $(modal_env_unsets) \
            HF_HOME="${TMP_HF_CACHE}" \
            OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
            VLLM_API_KEY="inspectai" \
            PYTHONNOUSERSITE="1" \
            python evaluate.py \
                --model-path /work/final_model \
                --templates-dir ../../../../src/eval/templates \
                --limit -1 \
                ${max_tokens_arg} \
                --json-output-file "${LOCAL_EVAL_DIR}/metrics.json"
    ) > "${LOCAL_EVAL_DIR}/final_eval_${eval_num}.txt" 2>&1
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${LOCAL_EVAL_DIR}/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        export EVAL_COUNTER
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        timeout --signal=TERM --kill-after=60s 28800s bash -c "$(declare -f run_evaluation modal_env_unsets); run_evaluation \"$max_tokens_arg\" \"$EVAL_COUNTER\""

        if [ -f "${LOCAL_EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# First evaluation: up to 4 attempts
run_evaluation_with_retry 4 ""

# Second evaluation with adjusted max tokens: up to 3 attempts
case "${EVALUATION_TASK}" in
    aime2025)
        MAX_TOKENS_ARG="--max-tokens 12000"
        ;;
    arenahardwriting)
        MAX_TOKENS_ARG="--max-new-tokens 12288"
        ;;
    bfcl)
        MAX_TOKENS_ARG="--max-tokens 12000"
        ;;
    gpqamain)
        MAX_TOKENS_ARG="--max-tokens 12000"
        ;;
    gsm8k)
        MAX_TOKENS_ARG="--max-tokens 3000"
        ;;
    healthbench)
        MAX_TOKENS_ARG="--max-new-tokens 12288"
        ;;
    humaneval)
        MAX_TOKENS_ARG="--max-tokens 3000"
        ;;
    *)
        MAX_TOKENS_ARG=""
        ;;
esac

run_evaluation_with_retry 3 "$MAX_TOKENS_ARG"

# Third evaluation with further adjusted max tokens: up to 2 attempts
case "${EVALUATION_TASK}" in
    aime2025)
        MAX_TOKENS_ARG="--max-tokens 8000"
        ;;
    arenahardwriting)
        MAX_TOKENS_ARG="--max-new-tokens 8192"
        ;;
    bfcl)
        MAX_TOKENS_ARG="--max-tokens 8000"
        ;;
    gpqamain)
        MAX_TOKENS_ARG="--max-tokens 8000"
        ;;
    gsm8k)
        MAX_TOKENS_ARG="--max-tokens 2000"
        ;;
    healthbench)
        MAX_TOKENS_ARG="--max-new-tokens 8192"
        ;;
    humaneval)
        MAX_TOKENS_ARG="--max-tokens 2000"
        ;;
    *)
        MAX_TOKENS_ARG=""
        ;;
esac

run_evaluation_with_retry 2 "$MAX_TOKENS_ARG"

if [ "${EVAL_COUNTER}" -gt 0 ] && [ -f "${LOCAL_EVAL_DIR}/final_eval_${EVAL_COUNTER}.txt" ]; then
    echo $(cat "${LOCAL_EVAL_DIR}/final_eval_${EVAL_COUNTER}.txt")
fi

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

[ -f "${LOCAL_EVAL_DIR}/metrics.json" ]

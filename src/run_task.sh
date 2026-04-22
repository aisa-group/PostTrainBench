#!/bin/bash

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
NUM_GPUS="${7:-1}"

source src/commit_utils/set_env_vars.sh

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:[]' '____')

AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:[]' '____')

RANDOM_UUID=$(uuidgen)

GPU_SUFFIX=""
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    GPU_SUFFIX="_${NUM_GPUS}gpu"
fi

export EVAL_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${GPU_SUFFIX}${POST_TRAIN_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p ${EVAL_DIR}

exec 1>${EVAL_DIR}/output.log
exec 2>${EVAL_DIR}/error.log

echo "$@"

export TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

echo "Preparing job directory..." 
mkdir -p "${JOB_DIR}"

mkdir "${JOB_DIR}/task"

cp "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task"
if [ -d "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task"
fi
cp -r src/eval/templates "${JOB_DIR}/task/"

if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r src/eval/tasks/${EVALUATION_TASK}/task_context/* "${JOB_DIR}/task"
fi
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

BENCHMARK=$(cat src/eval/tasks/${EVALUATION_TASK}/benchmark.txt)
PROMPT=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL_TO_TRAIN" --benchmark-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --num-gpus "$NUM_GPUS" --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh

# set openai api keys appropriately
export CODEX_API_KEY="${OPENAI_API_KEY}"
unset OPENAI_API_KEY
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

# Copy scripts needed inside the container
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp src/utils/system_monitor.sh "${JOB_DIR}/system_monitor.sh"
cp src/utils/timestamp_lines.py "${JOB_DIR}/timestamp_lines.py"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

# Copy agent-specific auth if present (e.g. for non-API agents)
if [ -f "agents/${AGENT}/auth.json" ]; then
    cp "agents/${AGENT}/auth.json" "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${JOB_DIR}/oauth_token"
fi

# Utils
with_huggingface_overlay() {
    mkdir -p "$TMP_SUBDIR/merged_huggingface"
    mkdir -p "$TMP_SUBDIR/upper_huggingface"
    mkdir -p "$TMP_SUBDIR/fuse_workdir"
    fuse-overlayfs -o "lowerdir=$HF_HOME,upperdir=$TMP_SUBDIR/upper_huggingface,workdir=$TMP_SUBDIR/fuse_workdir" "$TMP_SUBDIR/merged_huggingface"
    
    "$@"
    local exit_code=$?
    
    fusermount -u "$TMP_SUBDIR/merged_huggingface"
    rm -r "$TMP_SUBDIR/merged_huggingface"
    rm -r "$TMP_SUBDIR/upper_huggingface"
    rm -r "$TMP_SUBDIR/fuse_workdir"
    
    return $exit_code
}

with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)
    
    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"
    
    return $exit_code
}

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

solve_task() {
    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    apptainer exec \
        --nv \
        -c \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env HF_HOME="${HF_HOME_NEW}" \
        --env ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
        --env CODEX_API_KEY="${CODEX_API_KEY}" \
        --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
        --env OPENCODE_API_KEY="${OPENCODE_API_KEY}" \
        --env DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY}" \
        --env ZAI_API_KEY="${ZAI_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --env NUM_GPUS="${NUM_GPUS}" \
        --env PROMPT="${PROMPT}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_MERGED}:${HF_HOME_NEW}" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
        bash -c "{ python /home/ben/check_cuda.py && python /home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & MONITOR_PID=\$!; bash /home/ben/agent_solve.sh; kill \$MONITOR_PID 2>/dev/null; } 2>&1 | python /home/ben/timestamp_lines.py" > "${SOLVE_OUT}" 2>&1
}

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

with_huggingface_overlay with_record_the_time solve_task
SOLVE_EXIT=$?

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
elif [ $SOLVE_EXIT -gt 128 ]; then
    echo "status: killed by signal $((SOLVE_EXIT - 128)) ($(kill -l $((SOLVE_EXIT - 128)) 2>/dev/null || echo unknown))"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "final_model_files: $(ls "${JOB_DIR}/task/final_model/" 2>/dev/null | wc -l)"
echo "hostname: $(hostname)"
echo "fuse_overlayfs_alive: $(ps aux 2>/dev/null | grep fuse-overlay | grep -v grep | wc -l)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "disk_tmp: $(du -sh "${JOB_TMP}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# Parse agent trace into human-readable format
TRACE_PARSER="agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python "$TRACE_PARSER" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
    cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"
else
    echo "Warning: No trace parser found at $TRACE_PARSER, using raw output"
    cp "${SOLVE_OUT}" "${JOB_DIR}/solve_parsed.txt"
fi

echo "========================================="
echo "=== RUNNING CONTAMINATION JUDGE ==="
echo "========================================="

# Make judge helper tooling and benchmark metadata available inside the judge sandbox.
cp "src/disallowed_usage_judge/judge_tools/contamination_check.py" "${JOB_DIR}/contamination_check.py"
cp "src/disallowed_usage_judge/judge_tools/model_identity_check.py" "${JOB_DIR}/model_identity_check.py"
cp -r "src/disallowed_usage_judge/judge_tools/reference_configs" "${JOB_DIR}/reference_configs"

# Expose final_model/config.json to the judge as ../final_model_config.json so
# model_identity_check.py can compare it against the reference. Only the
# config.json is needed for the architecture-identity check, not the weights.
if [ -f "${JOB_DIR}/task/final_model/config.json" ]; then
    cp "${JOB_DIR}/task/final_model/config.json" "${JOB_DIR}/final_model_config.json"
fi

if [ -f "src/eval/tasks/${EVALUATION_TASK}/test_data.json" ]; then
    cp "src/eval/tasks/${EVALUATION_TASK}/test_data.json" "${JOB_DIR}/test_data.json"
fi

JUDGE_TASK=$(python src/disallowed_usage_judge/get_judge_prompt.py --benchmark-id "${EVALUATION_TASK}" --model "${MODEL_TO_TRAIN}")

# Reset codex config to prevent agent-specific settings (e.g. model_reasoning_effort)
# from leaking into the judge, which uses a different model
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

# ---- Judge 1: GPT-5.2 via codex CLI (subscription) ----
echo "=== Judge 1: GPT-5.2 (codex CLI, subscription) ==="

# Set up ChatGPT Pro subscription auth for codex judge
cp "agents/codex_non_api/auth.json" "${JOB_DIR}/.codex/auth.json"
if ! grep -q "forced_login_method" "${JOB_DIR}/.codex/config.toml" 2>/dev/null; then
    printf '\nforced_login_method = "chatgpt"\n' >> "${JOB_DIR}/.codex/config.toml"
fi

with_huggingface_overlay apptainer exec \
    --nv \
    -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env CODEX_API_KEY="" \
    --env OPENAI_API_KEY="" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --bind "${JOB_TMP}:/tmp" \
    --bind "${HF_MERGED}:${HF_HOME_NEW}" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/opus_4_6_codex_5_3.sif codex --search -a never exec -c model_reasoning_summary=detailed -c model_reasoning_effort=high --skip-git-repo-check --yolo --model "gpt-5.2" "$JUDGE_TASK" 2>&1 | tee "${EVAL_DIR}/judge_output_gpt5_2.txt"

if [ -f "${JOB_DIR}/task/contamination_judgement.txt" ]; then
    cp "${JOB_DIR}/task/contamination_judgement.txt" "${EVAL_DIR}/contamination_judgement_gpt5_2.txt"
fi
if [ -f "${JOB_DIR}/task/disallowed_model_judgement.txt" ]; then
    cp "${JOB_DIR}/task/disallowed_model_judgement.txt" "${EVAL_DIR}/disallowed_model_judgement_gpt5_2.txt"
fi

# Clean judgement files so the next judge starts fresh
rm -f "${JOB_DIR}/task/contamination_judgement.txt"
rm -f "${JOB_DIR}/task/disallowed_model_judgement.txt"

# ---- Judge 2: Claude Sonnet 4.6 via claude CLI (subscription) ----
echo "=== Judge 2: Claude Sonnet 4.6 (subscription) ==="

# Set up Claude Max subscription auth for claude judge
JUDGE_OAUTH_TOKEN=""
if [ -f "agents/claude_non_api/oauth_token" ]; then
    JUDGE_OAUTH_TOKEN="$(cat agents/claude_non_api/oauth_token)"
elif [ -f "${JOB_DIR}/oauth_token" ]; then
    JUDGE_OAUTH_TOKEN="$(cat ${JOB_DIR}/oauth_token)"
fi

with_huggingface_overlay apptainer exec \
    --nv \
    -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env ANTHROPIC_API_KEY="" \
    --env CLAUDE_CODE_OAUTH_TOKEN="${JUDGE_OAUTH_TOKEN}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --env CLAUDE_CODE_EFFORT_LEVEL="high" \
    --bind "${JOB_TMP}:/tmp" \
    --bind "${HF_MERGED}:${HF_HOME_NEW}" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/opus_4_6_codex_5_3.sif claude --print --verbose --model claude-sonnet-4-6 --output-format stream-json --dangerously-skip-permissions "$JUDGE_TASK" 2>&1 | tee "${EVAL_DIR}/judge_output_sonnet4_6.json"

python agents/claude/human_readable_trace.py "${EVAL_DIR}/judge_output_sonnet4_6.json" -o "${EVAL_DIR}/judge_output_sonnet4_6.txt"

if [ -f "${JOB_DIR}/task/contamination_judgement.txt" ]; then
    cp "${JOB_DIR}/task/contamination_judgement.txt" "${EVAL_DIR}/contamination_judgement_sonnet4_6.txt"
fi
if [ -f "${JOB_DIR}/task/disallowed_model_judgement.txt" ]; then
    cp "${JOB_DIR}/task/disallowed_model_judgement.txt" "${EVAL_DIR}/disallowed_model_judgement_sonnet4_6.txt"
fi

# ---- Aggregate: flag if either judge flags ----
echo "=== Aggregating Judge Results ==="

CONTAM_GPT=$(cat "${EVAL_DIR}/contamination_judgement_gpt5_2.txt" 2>/dev/null || echo "")
CONTAM_SONNET=$(cat "${EVAL_DIR}/contamination_judgement_sonnet4_6.txt" 2>/dev/null || echo "")

if echo "$CONTAM_GPT" | grep -qix "contamination detected" || echo "$CONTAM_SONNET" | grep -qix "contamination detected"; then
    echo "contamination detected" > "${EVAL_DIR}/contamination_judgement.txt"
else
    echo "no contamination detected" > "${EVAL_DIR}/contamination_judgement.txt"
fi

MODEL_GPT=$(cat "${EVAL_DIR}/disallowed_model_judgement_gpt5_2.txt" 2>/dev/null || echo "")
MODEL_SONNET=$(cat "${EVAL_DIR}/disallowed_model_judgement_sonnet4_6.txt" 2>/dev/null || echo "")

if echo "$MODEL_GPT" | grep -qix "disallowed use detected" || echo "$MODEL_SONNET" | grep -qix "disallowed use detected"; then
    echo "disallowed use detected" > "${EVAL_DIR}/disallowed_model_judgement.txt"
else
    echo "only allowed use detected" > "${EVAL_DIR}/disallowed_model_judgement.txt"
fi

echo "  Aggregated Contamination: $(cat "${EVAL_DIR}/contamination_judgement.txt")"
echo "  Aggregated Model usage: $(cat "${EVAL_DIR}/disallowed_model_judgement.txt")"

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
tree ${JOB_DIR}/task
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
fi

if [ -f "${JOB_DIR}/task/system_monitor.log" ]; then
    cp "${JOB_DIR}/task/system_monitor.log" "$EVAL_DIR/system_monitor.log"
fi

python containers/delete_hf_models.py "${JOB_DIR}/task"

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"

rm -rf /tmp/posttrain_container

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export REPO_ROOT="$(pwd)"

export TMP_HF_CACHE="/tmp/hf_cache_90afd0"

export EVAL_COUNTER=0

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
    with_huggingface_overlay apptainer exec \
        --nv \
        --env "HF_HOME=${TMP_HF_CACHE}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python "evaluate.py" \
            --model-path "$EVAL_DIR/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${max_tokens_arg} \
            --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/final_eval_${eval_num}.txt"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        export EVAL_COUNTER
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        timeout --signal=TERM --kill-after=60s 28800s bash -c "$(declare -f run_evaluation with_huggingface_overlay); run_evaluation \"$max_tokens_arg\" \"$EVAL_COUNTER\""

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# First evaluation: up to 4 attempts
run_evaluation_with_retry 4 ""

# Second evaluation with adjusted max tokens: up to 2 attempts
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

echo $(cat "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt")

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

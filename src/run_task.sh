#!/bin/bash

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
NUM_GPUS="${7:-1}"

source src/commit_utils/set_env_vars.sh

# Select the judge backend for grader-based benchmarks (arenahardwriting / healthbench):
# default to the OpenAI-backed evaluate.py, but fall back to the OpenRouter variant when
# .env provides OPENROUTER_API_KEY but no OPENAI_API_KEY.
JUDGE_BACKEND="openai"
if { [ "$EVALUATION_TASK" = "arenahardwriting" ] || [ "$EVALUATION_TASK" = "healthbench" ]; } \
   && [ -z "${OPENAI_API_KEY}" ] && [ -n "${OPENROUTER_API_KEY}" ]; then
    JUDGE_BACKEND="openrouter"
fi

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    export EVAL_SCRIPT="evaluate_openrouter.py"
else
    export EVAL_SCRIPT="evaluate.py"
fi

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
echo "Judge backend: ${JUDGE_BACKEND} (eval script: ${EVAL_SCRIPT})"

export TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

echo "Preparing job directory..." 
mkdir -p "${JOB_DIR}"

mkdir "${JOB_DIR}/task"

cp "src/eval/tasks/${EVALUATION_TASK}/${EVAL_SCRIPT}" "${JOB_DIR}/task/evaluate.py"
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

# --- API key allowlist ----------------------------------------------------
# Each agent declares the third-party API keys it may receive in
# agents/<agent>/api_keys.json; each benchmark declares the keys its grading
# needs in src/eval/tasks/<task>/info.json ("required_api_keys", default none).
# The agent sandbox is launched with -c --cleanenv, so it inherits NOTHING from
# the host: it receives ONLY the union of those two sets via --env (built into
# API_KEY_ENV_ARGS below). Every other provider key is never passed in, hence
# unset inside the sandbox. OPENAI_API_KEY thus reaches the agent only for the
# arenahardwriting/healthbench benchmarks (which list it in required_api_keys).
if ! ALLOWED_API_KEYS_RAW="$(python3 -c '
import json, sys
agent_keys = json.load(open(sys.argv[1]))["allowed_api_keys"]
bench = json.load(open(sys.argv[2]))
seen, out = set(), []
for k in list(agent_keys) + bench.get("required_api_keys", []):
    if k not in seen:
        seen.add(k); out.append(k)
print("\n".join(out))
' "agents/${AGENT}/api_keys.json" "src/eval/tasks/${EVALUATION_TASK}/info.json")"; then
    echo "ERROR: failed to compute API key allowlist for agent=${AGENT} task=${EVALUATION_TASK}" >&2
    exit 1
fi

ALLOWED_API_KEYS=()
[ -n "$ALLOWED_API_KEYS_RAW" ] && mapfile -t ALLOWED_API_KEYS <<< "$ALLOWED_API_KEYS_RAW"

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    ALLOWED_API_KEYS+=("OPENROUTER_API_KEY")
fi

API_KEY_ENV_ARGS=()
for _k in "${ALLOWED_API_KEYS[@]}"; do
    API_KEY_ENV_ARGS+=(--env "${_k}=${!_k}")
done
echo "API keys provisioned for agent=${AGENT} task=${EVALUATION_TASK}: ${ALLOWED_API_KEYS[*]:-<none>}"

# Copy scripts needed inside the container
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp src/utils/system_monitor.sh "${JOB_DIR}/system_monitor.sh"
cp src/utils/timestamp_lines.py "${JOB_DIR}/timestamp_lines.py"
cp src/utils/update_agent_cli.sh "${JOB_DIR}/update_agent_cli.sh"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

# Self-decontamination tooling for the agent: the same n-gram checker and
# test-set copy the contamination judge gets, at the same paths (the judge
# phase re-copies both via prepare_judge_sandbox, so the judges never run
# agent-modified versions). The matching usage instructions are added to the
# agent prompt by get_prompt.py, conditional on the same test_data.json.
if [ -f "src/eval/tasks/${EVALUATION_TASK}/test_data.json" ]; then
    cp src/judges/judge_tools/contamination_check.py "${JOB_DIR}/contamination_check.py"
    cp "src/eval/tasks/${EVALUATION_TASK}/test_data.json" "${JOB_DIR}/test_data.json"
fi

# Agent-specific auth: auth.json is bind-mounted at apptainer exec time so the
# codex CLI can write the rotated refresh token back to the shared source file
# (single-use refresh tokens otherwise burn after one job).
AGENT_AUTH_SRC=""
if [ -f "agents/${AGENT}/auth.json" ]; then
    AGENT_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/auth.json")" && pwd)/auth.json"
    # Placeholder file inside the sandbox .codex dir for the bind mount to overlay.
    mkdir -p "${JOB_DIR}/.codex"
    : > "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${JOB_DIR}/oauth_token"
fi

# Cursor CLI persists its OAuth tokens at ~/.config/cursor/auth.json. Bind-mount
# the agent's copy so the CLI inside the sandbox reads and rotates against the
# shared credential file. Distinct filename in the agent dir avoids collision
# with the codex auth.json check above.
CURSOR_AUTH_SRC=""
if [ -f "agents/${AGENT}/cursor_auth.json" ]; then
    CURSOR_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/cursor_auth.json")" && pwd)/cursor_auth.json"
    mkdir -p "${JOB_DIR}/.config/cursor"
    : > "${JOB_DIR}/.config/cursor/auth.json"
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
    AGENT_AUTH_BIND=()
    [ -n "$AGENT_AUTH_SRC" ] && AGENT_AUTH_BIND=(--bind "${AGENT_AUTH_SRC}:/home/ben/.codex/auth.json")
    CURSOR_AUTH_BIND=()
    [ -n "$CURSOR_AUTH_SRC" ] && CURSOR_AUTH_BIND=(--bind "${CURSOR_AUTH_SRC}:/home/ben/.config/cursor/auth.json")
    # Forward the CLI-auto-update opt-out into the sandbox so update_agent_cli.sh
    # can honor it. Only set when the user opts in via .env.
    CLI_UPDATE_ENV=()
    [ -n "${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-}" ] && CLI_UPDATE_ENV+=(--env "POST_TRAIN_BENCH_SKIP_CLI_UPDATE=${POST_TRAIN_BENCH_SKIP_CLI_UPDATE}")
    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    apptainer exec \
        --nv \
        -c \
        --cleanenv \
        --pid \
        --no-init \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env HF_HOME="${HF_HOME_NEW}" \
        "${API_KEY_ENV_ARGS[@]}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --env NUM_GPUS="${NUM_GPUS}" \
        --env PROMPT="${PROMPT}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        "${CLI_UPDATE_ENV[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_MERGED}:${HF_HOME_NEW}" \
        "${AGENT_AUTH_BIND[@]}" \
        "${CURSOR_AUTH_BIND[@]}" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
        bash -c "{ python /home/ben/check_cuda.py && python /home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & MONITOR_PID=\$!; bash /home/ben/agent_solve.sh; kill \$MONITOR_PID 2>/dev/null; } 2>&1 | python /home/ben/timestamp_lines.py" > "${SOLVE_OUT}" 2>&1
}

# ---------- judge OAuth precheck ----------
# All judges run via the codex CLI with the subscription auth at
# agents/codex_non_api/auth.json (see src/judges/judge_lib.sh). If its ChatGPT
# session is invalidated, we'd waste the full agent run only to hard-error at
# the judge phase. One curl to a lightweight ChatGPT endpoint tells us the
# state: it uses the already-issued access token, no refresh path, so it
# doesn't rotate anything or race parallel job starts.

echo "================================"
echo "======= JUDGE AUTH CHECK ======="
echo "================================"
JUDGE_AUTH="agents/codex_non_api/auth.json"
if [ ! -f "$JUDGE_AUTH" ]; then
    echo "ERROR: judge auth file missing at $JUDGE_AUTH" >&2
    exit 1
fi
JUDGE_ACCESS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"]["access_token"])' "$JUDGE_AUTH") \
    || { echo "ERROR: could not extract tokens.access_token from $JUDGE_AUTH" >&2; exit 1; }
JUDGE_HTTP=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 \
    -H "Authorization: Bearer $JUDGE_ACCESS" \
    "https://chatgpt.com/backend-api/codex/models?client_version=0.124.0")
if [ "$JUDGE_HTTP" != "200" ]; then
    echo "ERROR: judge OAuth precheck failed (HTTP ${JUDGE_HTTP})." >&2
    echo "The ChatGPT session in $JUDGE_AUTH may be invalidated." >&2
    echo "Re-login on the head node:" >&2
    echo "  codex logout && codex login && cp ~/.codex/auth.json $JUDGE_AUTH && chmod 600 $JUDGE_AUTH" >&2
    exit 1
fi
echo "Judge OAuth OK (HTTP 200)"

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

# Record the (auto-updated) agent CLI version captured by update_agent_cli.sh.
if [ -f "${JOB_DIR}/cli_version.txt" ]; then
    cp "${JOB_DIR}/cli_version.txt" "${EVAL_DIR}/cli_version.txt"
    echo "--- AGENT CLI VERSION ---"
    cat "${EVAL_DIR}/cli_version.txt"
    echo "--- END AGENT CLI VERSION ---"
else
    echo "WARNING: ${JOB_DIR}/cli_version.txt not found (agent CLI version not recorded)" >&2
fi

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# Parse agent trace into human-readable format
python src/trace_parsing/parse_trace.py --agent "${AGENT}" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"

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

echo "========================================="
echo "=== RUNNING REWARD-HACKING JUDGES ==="
echo "========================================="

source src/judges/judge_lib.sh

# Make judge helper tooling and benchmark metadata available inside the judge
# sandbox. The final_model config comes from EVAL_DIR because delete_hf_models
# has already run on JOB_DIR/task during cleanup.
prepare_judge_sandbox "${JOB_DIR}" "${EVALUATION_TASK}" "${EVAL_DIR}/final_model/config.json"

# Reset codex config (so agent-specific settings like model_reasoning_effort
# can't leak into the judges) and set up the bind-mounted subscription auth
# (JUDGE_CODEX_AUTH_SRC), through which rotated refresh tokens persist back to
# the source instead of dying with the sandbox.
setup_judge_codex_auth "${JOB_DIR}" || exit 1

JUDGE_EXTRA_APPTAINER_ARGS=(
    --nv
    --env HF_HOME="${HF_HOME_NEW}"
    --env VLLM_API_KEY="inspectai"
    --bind "${HF_MERGED}:${HF_HOME_NEW}"
)

FIRST_JUDGE=1
for JUDGE_NAME in "${ALL_JUDGES[@]}"; do
    load_judge_conf "${JUDGE_NAME}" || exit 1

    echo "=== Judge: ${JUDGE_LABEL} ==="

    # Clean judgement file between judges so each one starts fresh
    [ "$FIRST_JUDGE" = "1" ] || rm -f "${JOB_DIR}/task/judgement.json"
    FIRST_JUDGE=0

    JUDGE_PROMPT=$(build_judge_prompt "${JUDGE_NAME}" "${EVALUATION_TASK}" "${MODEL_TO_TRAIN}" "${AGENT}" "${AGENT_CONFIG}")

    with_huggingface_overlay run_judge_exec "${JOB_DIR}" "${JOB_TMP}" "${EVAL_DIR}/judge_output_${JUDGE_OUTPUT_ID}.json" "${JUDGE_PROMPT}"

    # missing_fatal=0: a judge that produces no verdict warns and moves on. The
    # agent's 10h of work is already done, so it must still be evaluated; the
    # rerun pipeline can supply the missing verdict afterwards.
    collect_judge_output "${JOB_DIR}" "${EVAL_DIR}" "" 0
done

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
        --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python "${EVAL_SCRIPT}" \
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

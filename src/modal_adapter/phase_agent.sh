#!/bin/bash
# Modal port of src/run_task.sh phases 1-2: job setup, agent solve, trace
# parse, cleanup (run_task.sh lines 13-188 and 221-239). Differences:
#   - no apptainer: the Modal container IS the job container, so the solve
#     pipeline runs directly with an adjusted environment;
#   - fuse-overlayfs HF cache replaced by hf_symlink_farm (see hf_cache_lib.sh);
#   - artifacts are staged into LOCAL_EVAL_DIR; app.py copies them to the
#     results volume afterwards.
#
# Inputs via environment (set by app.py:run_agent):
#   EVALUATION_TASK AGENT AGENT_CONFIG MODEL_TO_TRAIN NUM_HOURS NUM_GPUS
#   LOCAL_EVAL_DIR REPO JOB_DIR HF_RO POST_TRAIN_BENCH_PROMPT
#   plus the API keys from the posttrainbench-keys secret.

source "${REPO}/src/modal_adapter/hf_cache_lib.sh"

mkdir -p "${LOCAL_EVAL_DIR}"
exec 1> >(tee "${LOCAL_EVAL_DIR}/output.log")
exec 2> >(tee "${LOCAL_EVAL_DIR}/error.log" >&2)

echo "agent phase: task=${EVALUATION_TASK} agent=${AGENT} config=${AGENT_CONFIG} model=${MODEL_TO_TRAIN} hours=${NUM_HOURS} gpus=${NUM_GPUS}"

cd "${REPO}"

export HF_HOME_NEW="${JOB_DIR}/hf_cache"
export PYTHONNOUSERSITE=1

echo "Preparing job directory..."
mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_DIR}/task"

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
echo "$PROMPT" > "${LOCAL_EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh

# set openai api keys appropriately (run_task.sh:64-69)
export CODEX_API_KEY="${OPENAI_API_KEY:-}"
unset OPENAI_API_KEY
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

# Copy scripts needed inside the job dir
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp src/utils/system_monitor.sh "${JOB_DIR}/system_monitor.sh"
cp src/utils/timestamp_lines.py "${JOB_DIR}/timestamp_lines.py"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

# Agent-specific auth for non-API agents. run_task.sh copies these from
# agents/<agent>/; on Modal they come from the posttrainbench-keys secret.
if [ -n "${CODEX_AUTH_JSON:-}" ]; then
    printf '%s' "${CODEX_AUTH_JSON}" > "${JOB_DIR}/.codex/auth.json"
fi
if [ -n "${CLAUDE_OAUTH_TOKEN:-}" ]; then
    printf '%s' "${CLAUDE_OAUTH_TOKEN}" > "${JOB_DIR}/oauth_token"
fi

with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)

    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${LOCAL_EVAL_DIR}/time_taken.txt"

    return $exit_code
}

SOLVE_OUT="${LOCAL_EVAL_DIR}/solve_out.txt"

# The cluster passes the host environment plus overrides into the container
# (apptainer -c contains the filesystem, not the environment). Mirror that:
# inherit everything except MODAL_* internals, with the same overrides that
# run_task.sh sets via --env (run_task.sh:122-147).
solve_task() {
    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    env $(modal_env_unsets) \
        HOME="${JOB_DIR}" \
        PATH="/root/.local/bin:${JOB_DIR}/.local/bin:${PATH}" \
        HF_HOME="${HF_HOME_NEW}" \
        VLLM_API_KEY="inspectai" \
        PYTHONNOUSERSITE="1" \
        NUM_GPUS="${NUM_GPUS}" \
        PROMPT="${PROMPT}" \
        AGENT_CONFIG="${AGENT_CONFIG}" \
        bash -c "cd '${JOB_DIR}/task' && { python '${JOB_DIR}/check_cuda.py' && python '${JOB_DIR}/check_cuda_writing.py' || exit 1; bash '${JOB_DIR}/system_monitor.sh' & MONITOR_PID=\$!; bash '${JOB_DIR}/agent_solve.sh'; kill \$MONITOR_PID 2>/dev/null; } 2>&1 | python '${JOB_DIR}/timestamp_lines.py'" > "${SOLVE_OUT}" 2>&1
}

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

hf_symlink_farm "${HF_RO}" "${HF_HOME_NEW}"

with_record_the_time solve_task
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
echo "modal_task: ${MODAL_TASK_ID:-unknown}"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "disk_tmp: $(du -sh /tmp 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

TRACE_PARSER="agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python "$TRACE_PARSER" "${SOLVE_OUT}" -o "${LOCAL_EVAL_DIR}/solve_parsed.txt"
else
    echo "Warning: No trace parser found at $TRACE_PARSER, using raw output"
    cp "${SOLVE_OUT}" "${LOCAL_EVAL_DIR}/solve_parsed.txt"
fi

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
tree "${JOB_DIR}/task"
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "${LOCAL_EVAL_DIR}/final_model"
fi

if [ -f "${JOB_DIR}/task/system_monitor.log" ]; then
    cp "${JOB_DIR}/task/system_monitor.log" "${LOCAL_EVAL_DIR}/system_monitor.log"
fi

python containers/delete_hf_models.py "${JOB_DIR}/task"

cp -r "${JOB_DIR}/task" "${LOCAL_EVAL_DIR}/task"

echo "agent phase done"
exit $SOLVE_EXIT

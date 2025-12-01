#!/bin/bash

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"

source src/commit_utils/set_env_vars.sh

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')

AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')

RANDOM_UUID=$(uuidgen)

EVAL_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_final_final/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p ${EVAL_DIR}

exec 1>${EVAL_DIR}/output.log
exec 2>${EVAL_DIR}/error.log

echo "$@"

TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

cp -r containers/download_hf_cache/hf_cache ${JOB_DIR}

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
PROMPT=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL_TO_TRAIN" --benchmark "$BENCHMARK" --num-hours "$NUM_HOURS" --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"
begin=$(date --iso-8601=seconds)

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

bash agents/${AGENT}/solve.sh "$NUM_HOURS" "$PROMPT" "$AGENT_CONFIG" "$JOB_DIR" "$JOB_TMP" "$EVAL_DIR" > "${SOLVE_OUT}" 2>&1

end=$(date --iso-8601=seconds)
time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
printf '%02d:%02d:%02d\n' \
  $(( time_taken / 3600 )) \
  $(( (time_taken % 3600) / 60 )) \
  $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"

echo "=================================================="
echo "=== TASK COMPLETE, RUNNING CONTAMINATION JUDGE ==="
echo "=================================================="

JUDGE_TASK=$(python src/disallowed_usage_judge/get_judge_prompt.py --benchmark "${BENCHMARK}" --model "${MODEL_TO_TRAIN}")

apptainer exec \
    --nv \
    -c \
    --env PATH="/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env CODEX_API_KEY="${OPENAI_API_KEY}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --bind "${JOB_TMP}:/tmp" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif codex --search -a never exec --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_TASK"

cp "${JOB_DIR}/task/contamination_judgement.txt" "${EVAL_DIR}/contamination_judgement.txt"
cp "${JOB_DIR}/task/disallowed_model_judgement.txt" "${EVAL_DIR}/disallowed_model_judgement.txt"

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
tree ${JOB_DIR}/task
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
fi

python containers/delete_hf_models.py "${JOB_DIR}/task"

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"

rm -rf /tmp/posttrain_container

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

mkdir -p /tmp/hf_cache

REPO_ROOT="$(pwd)"

apptainer exec \
    --nv \
    --env HF_HOME="/tmp/hf_cache" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "/tmp/hf_cache:/tmp/hf_cache" \
    --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif python "evaluate.py" \
        --model-path "$EVAL_DIR/final_model" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/final_eval.txt"

echo $(cat "$EVAL_DIR/final_eval.txt")
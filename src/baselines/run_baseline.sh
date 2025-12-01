#!/bin/bash

EVAL_NAME="$1"
MODEL_NAME="$2"
CLUSTER_ID="$3"

set -euo pipefail

REPO_ROOT="$(pwd)"
RESULT_PREFIX_SAFE=$(echo "${MODEL_NAME}" | tr '/:' '_')
RESULT_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/baseline/${EVAL_NAME}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

echo $RESULT_DIR

mkdir -p "${RESULT_DIR}"

exec 1>${RESULT_DIR}/output.log
exec 2>${RESULT_DIR}/error.log

echo "Eval: ${EVAL_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Cluster ID: ${CLUSTER_ID}"

source src/commit_utils/set_env_vars.sh

echo "================================"
echo "========= RUNNING EVAL ========="
echo "================================"

sleep 1

begin=$(date --iso-8601=seconds)
apptainer exec \
    --nv \
    --env HF_HOME="${HF_HOME}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --writable-tmpfs \
    --bind "${RESULT_DIR}:${RESULT_DIR}" \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --pwd "${REPO_ROOT}/src/eval/tasks/${EVAL_NAME}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif \
    python "evaluate.py" \
        --model-path "${MODEL_NAME}" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --json-output-file "${RESULT_DIR}/metrics.json" > "${RESULT_DIR}/final_eval.txt" 
end=$(date --iso-8601=seconds)

time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
printf '%02d:%02d:%02d\n' \
  $(( time_taken / 3600 )) \
  $(( (time_taken % 3600) / 60 )) \
  $(( time_taken % 60 )) > "${RESULT_DIR}/time_taken.txt"

echo "Time taken (seconds): $time_taken" >> "${RESULT_DIR}/final_eval.txt"

echo "${MODEL_NAME}" > "${RESULT_DIR}/model.txt"
echo "${EVAL_NAME}" > "${RESULT_DIR}/eval.txt"
date --iso-8601=seconds > "${RESULT_DIR}/timestamp.txt"

echo "================================"
echo "========= BASELINE DONE ========"
echo "================================"
#!/bin/bash

EVAL_NAME="$1"
MODEL_NAME="$2"
CLUSTER_ID="$3"

set -euo pipefail

source src/commit_utils/set_env_vars.sh

REPO_ROOT="$(pwd)"
RESULT_PREFIX_SAFE=$(echo "${MODEL_NAME}" | tr '/:' '_')
RESULT_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/baseline/${EVAL_NAME}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

RANDOM_UUID=$(uuidgen)
TMP_SUBDIR="/tmp/posttrain_baseline_${EVAL_NAME}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"
HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
TMP_HF_CACHE="/tmp/hf_cache_baseline"

echo $RESULT_DIR

mkdir -p "${RESULT_DIR}"
mkdir -p "${TMP_SUBDIR}"

exec 1>${RESULT_DIR}/output.log
exec 2>${RESULT_DIR}/error.log

echo "Eval: ${EVAL_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Cluster ID: ${CLUSTER_ID}"

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
        $(( time_taken % 60 )) > "${RESULT_DIR}/time_taken.txt"

    echo "Time taken (seconds): $time_taken" >> "${RESULT_DIR}/final_eval.txt"

    return $exit_code
}

check_cuda() {
    apptainer exec \
        --nv \
        --env HF_HOME="${TMP_HF_CACHE}" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif \
        python src/utils/check_cuda_writing.py > "${RESULT_DIR}/cuda_check.txt"
}

run_eval() {
    apptainer exec \
        --nv \
        --env HF_HOME="${TMP_HF_CACHE}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env VLLM_LOGGING_LEVEL="DEBUG" \
        --writable-tmpfs \
        --bind "${RESULT_DIR}:${RESULT_DIR}" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "${REPO_ROOT}/src/eval/tasks/${EVAL_NAME}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif \
        python "evaluate.py" \
            --model-path "${MODEL_NAME}" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            --json-output-file "${RESULT_DIR}/metrics.json" > "${RESULT_DIR}/final_eval.txt"
}

with_huggingface_overlay check_cuda

echo "================================"
echo "========= RUNNING EVAL ========="
echo "================================"

sleep 1

with_huggingface_overlay with_record_the_time run_eval

echo "${MODEL_NAME}" > "${RESULT_DIR}/model.txt"
echo "${EVAL_NAME}" > "${RESULT_DIR}/eval.txt"
date --iso-8601=seconds > "${RESULT_DIR}/timestamp.txt"

# Cleanup
rm -rf "${TMP_SUBDIR}"

echo "================================"
echo "========= BASELINE DONE ========"
echo "================================"
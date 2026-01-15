#!/bin/bash
EVALUATION_TASK="$1"
EVAL_DIR="$2"
export HOME="$3"
CLUSTER="$4"

TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RANDOM_UUID}"
HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
mkdir -p "${TMP_SUBDIR}"
mkdir -p "${HF_MERGED}"

source src/commit_utils/set_env_vars.sh

exec 1>${EVAL_DIR}/z_new_${CLUSTER}_output.log
exec 2>${EVAL_DIR}/z_new_${CLUSTER}_error.log

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi

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

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

REPO_ROOT="$(pwd)"

TMP_HF_CACHE="/tmp/hf_cache_90afd1"
with_huggingface_overlay apptainer exec \
    --nv \
    --env "HF_HOME=${TMP_HF_CACHE}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY})" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --env VLLM_LOGGING_LEVEL="DEBUG" \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
    --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif python "evaluate.py" \
        --model-path "$EVAL_DIR/final_model" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/z_new_${CLUSTER}_final_eval.txt"

echo $(cat "$EVAL_DIR/z_new_${CLUSTER}_final_eval.txt")
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
# See run_task.sh: node-local /tmp is not guaranteed to be the big scratch that
# the HTCondor submit file reserves. Unset means the upstream /tmp.
TMP_SUBDIR="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}/posttrain_baseline_${EVAL_NAME}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"
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

# --env HF_HOME is not enough on its own. huggingface_hub resolves its cache as
# HF_HUB_CACHE, else HUGGINGFACE_HUB_CACHE, else "$HF_HOME/hub" -- HF_HOME is the
# lowest-priority of the three. These apptainer execs do not pass --cleanenv, so a
# host that exports HUGGINGFACE_HUB_CACHE (this cluster does, globally, pointing at
# the real shared cache) has that value win inside the container, and the overlay
# mounted at TMP_HF_CACHE is never read. The host path is not bound in, so the hub
# then materialises it on the container root -- which --writable-tmpfs caps at
# `sessiondir max size` (64 MiB here) -- and the 3.4 GB download dies as
# "OSError: [Errno 28] No space left on device" inside file_download.py, four
# frames below anything that mentions a cache. Naming all three leaves nothing for
# the ambient environment to decide.
HF_CACHE_ENV=(
    --env HF_HOME="${TMP_HF_CACHE}"
    --env HF_HUB_CACHE="${TMP_HF_CACHE}/hub"
    --env HUGGINGFACE_HUB_CACHE="${TMP_HF_CACHE}/hub"
)

# The hub is not the only cache this host redirects. It points seven variables at
# one root -- UV_CACHE_DIR, PIP_CACHE_DIR, TRITON_CACHE_DIR, XDG_CACHE_HOME,
# HF_HOME, HUGGINGFACE_HUB_CACHE, TORCH_HOME -- and XDG_CACHE_HOME is the one that
# matters most, because it is where anything without a variable of its own lands:
# vLLM's VLLM_CACHE_ROOT defaults to "$XDG_CACHE_HOME/vllm", which is how
# vllm/modelinfos/ ends up there. None of that root is bound in, so each write goes
# to the 64 MiB container root and fails -- first as "Error saving model info cache"
# (survivable), then inside triton's kernel cache as
# torch._inductor.exc.InductorError: OSError: [Errno 28], which kills EngineCore
# and takes the vLLM server with it.
#
# One bind covers all seven, and the next library's variable too, which naming them
# individually would not. Binding it through rather than redirecting it also keeps
# the compiled triton kernels across runs; the alternative costs a recompile per
# evaluation, and there are up to nine per job. The HF cache is deliberately not
# reached this way -- the three variables above still point at the overlay, so the
# scorer cannot write into the shared hub.
CACHE_BIND=()
if [ -n "${XDG_CACHE_HOME:-}" ] && [ -d "${XDG_CACHE_HOME}" ]; then
    CACHE_BIND=(--bind "${XDG_CACHE_HOME}:${XDG_CACHE_HOME}")
fi

check_cuda() {
    apptainer exec \
        --nv \
        "${HF_CACHE_ENV[@]}" \
        "${CACHE_BIND[@]}" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif \
        python src/utils/check_cuda_writing.py > "${RESULT_DIR}/cuda_check.txt"
}

run_eval() {
    apptainer exec \
        --nv \
        "${HF_CACHE_ENV[@]}" \
        "${CACHE_BIND[@]}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
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
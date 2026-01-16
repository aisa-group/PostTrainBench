#!/bin/bash
set -euo pipefail
source src/commit_utils/set_env_vars.sh
export POST_TRAIN_BENCH_CONTAINERS_DIR=${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}
if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi
REPO_ROOT="$(pwd)"
CONTAINER_NAME="${1:-$POST_TRAIN_BENCH_CONTAINER_NAME}"
apptainer shell \
    --nv \
    --env HF_HOME="${HF_HOME}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${REPO_ROOT}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${CONTAINER_NAME}.sif"
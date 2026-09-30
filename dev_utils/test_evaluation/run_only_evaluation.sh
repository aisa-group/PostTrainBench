#!/bin/bash
export EVALUATION_TASK="$1"
export EVAL_DIR="$2"
export HOME="$3"
export CLUSTER="$4"

source src/commit_utils/set_env_vars.sh

exec 1>${EVAL_DIR}/z_new_${CLUSTER}_output.log
exec 2>${EVAL_DIR}/z_new_${CLUSTER}_error.log

# On exit, give the users in POST_TRAIN_BENCH_USERS_ACCESS access to what this wrote (src/utils/grant_access.py).
GRANT_ACCESS="$(pwd)/src/utils/grant_access.py"
trap 'python3 "${GRANT_ACCESS}" "${EVAL_DIR}"' EXIT

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi

export REPO_ROOT="$(pwd)"

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${REPO_ROOT}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python src/utils/check_cuda_writing.py > "$EVAL_DIR/cuda_check.txt"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

# The same final evaluation as src/run_task.sh. The per-seed metrics and logs go to their own folder, next to the
# evaluation/ folder of the original run. run_final_eval.sh refuses to overwrite an existing metrics.json.
bash src/eval/run_final_eval.sh "${EVALUATION_TASK}" "${EVAL_DIR}/final_model" "${EVAL_DIR}/z_new_${CLUSTER}_evaluation" "${EVAL_DIR}/metrics.json" \
    || { echo "ERROR: the final evaluation failed" >&2; exit 1; }

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

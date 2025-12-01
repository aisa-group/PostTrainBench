#!/bin/bash
EVALUATION_TASK="$1"
EVAL_DIR="$2"
export HOME="$3"
CLUSTER="$4"

source src/commit_utils/set_env_vars.sh

exec 1>${EVAL_DIR}/z_new_${CLUSTER}_output.log
exec 2>${EVAL_DIR}/z_new_${CLUSTER}_error.log

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

REPO_ROOT="$(pwd)"

apptainer exec \
    --nv \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY})" \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${HF_HOME}:${HF_HOME_NEW}" \
    --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif python "evaluate.py" \
        --model-path "$EVAL_DIR/final_model" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --json-output-file "${EVAL_DIR}/z_new_${CLUSTER}_metrics.json" > "$EVAL_DIR/z_new_${CLUSTER}_final_eval.txt"

echo $(cat "$EVAL_DIR/z_new_${CLUSTER}_final_eval.txt")
#!/bin/bash
export EVALUATION_TASK="$1"
export EVAL_DIR="$2"
export HOME="$3"
export CLUSTER="$4"

# Like run_task.sh's final evaluation: run the task's evaluate_final_eval.py, which every task needs (the final
# evaluation passes --seed, which evaluate.py does not accept).
export FINAL_EVAL_SCRIPT="evaluate_final_eval.py"
if [ ! -f "src/eval/tasks/${EVALUATION_TASK}/${FINAL_EVAL_SCRIPT}" ]; then
    echo "ERROR: src/eval/tasks/${EVALUATION_TASK}/${FINAL_EVAL_SCRIPT} not found — required for the seeded final evaluation" >&2
    exit 1
fi

export TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RANDOM_UUID}"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
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

with_huggingface_overlay apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${REPO_ROOT}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python src/utils/check_cuda_writing.py > "$EVAL_DIR/cuda_check.txt"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export REPO_ROOT="$(pwd)"

export TMP_HF_CACHE="/tmp/hf_cache_90afd1"

# The aggregated result is written to metrics.json; refuse to overwrite one.
if [ -f "${EVAL_DIR}/metrics.json" ]; then
    echo "ERROR: ${EVAL_DIR}/metrics.json already exists; move it away before re-evaluating" >&2
    exit 1
fi

# Same seeds and aggregation as src/run_task.sh. The per-seed metrics and logs
# go to their own folder, next to the evaluation/ folder of the original run.
EVAL_SEEDS=(0 1 2 3 4)
export EVAL_OUTPUT_DIR="${EVAL_DIR}/z_new_${CLUSTER}_evaluation"
mkdir -p "${EVAL_OUTPUT_DIR}"

export EVAL_COUNTER=0

run_evaluation() {
    local max_tokens_arg="$1"
    local seed="$2"
    local eval_num="$3"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
    with_huggingface_overlay apptainer exec \
        --nv \
        --env "HF_HOME=${TMP_HF_CACHE}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --env VLLM_LOGGING_LEVEL="DEBUG" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python "${FINAL_EVAL_SCRIPT}" \
            --model-path "$EVAL_DIR/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            --seed "${seed}" \
            ${max_tokens_arg} \
            --json-output-file "${EVAL_OUTPUT_DIR}/metrics_seed${seed}.json" > "${EVAL_OUTPUT_DIR}/final_eval_seed${seed}_${eval_num}.txt"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"
    local seed="$3"
    local metrics_file="${EVAL_OUTPUT_DIR}/metrics_seed${seed}.json"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${metrics_file}" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        export EVAL_COUNTER
        echo "Seed ${seed}: evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        timeout --signal=TERM --kill-after=60s 28800s bash -c "$(declare -f run_evaluation with_huggingface_overlay); run_evaluation \"$max_tokens_arg\" \"$seed\" \"$EVAL_COUNTER\""

        if [ -f "${metrics_file}" ]; then
            return 0
        fi
    done

    return 1
}

# Retry cascade, one stage per max-tokens setting: stage 0 uses evaluate.py's
# default (up to 4 attempts), stages 1 and 2 lower it (up to 3 and 2 attempts).
STAGE_MAX_RETRIES=(4 3 2)
STAGE_MAX_TOKENS_ARGS=("")

# Second evaluation stage with adjusted max tokens
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

STAGE_MAX_TOKENS_ARGS+=("$MAX_TOKENS_ARG")

# Third evaluation stage with further adjusted max tokens
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

STAGE_MAX_TOKENS_ARGS+=("$MAX_TOKENS_ARG")

# Runs the retry cascade for one seed, from stage $2 onwards. On success, sets
# SEED_STAGE to the stage that produced the seed's metrics.
run_seed_evaluation() {
    local seed="$1"
    local start_stage="$2"
    local stage

    for ((stage=start_stage; stage<${#STAGE_MAX_RETRIES[@]}; stage++)); do
        if run_evaluation_with_retry "${STAGE_MAX_RETRIES[$stage]}" "${STAGE_MAX_TOKENS_ARGS[$stage]}" "$seed"; then
            SEED_STAGE=$stage
            return 0
        fi
    done

    return 1
}

# Same seed policy as src/run_task.sh: if the first seed fails at every stage,
# the other seeds are skipped and no metrics.json is written; the later seeds
# start at the stage where the first seed succeeded.
SEED_RESULT_ARGS=()
START_STAGE=0
for seed in "${EVAL_SEEDS[@]}"; do
    EVAL_COUNTER=0
    echo "=== Seed ${seed}: evaluating from stage ${START_STAGE} ==="
    if run_seed_evaluation "$seed" "$START_STAGE"; then
        echo "Seed ${seed}: succeeded at stage ${SEED_STAGE}"
        SEED_RESULT_ARGS+=(--seed-result "$seed" "$SEED_STAGE" "${EVAL_OUTPUT_DIR}/metrics_seed${seed}.json")
        if [ "$seed" = "${EVAL_SEEDS[0]}" ]; then
            START_STAGE=$SEED_STAGE
        fi
        echo $(cat "${EVAL_OUTPUT_DIR}/final_eval_seed${seed}_${EVAL_COUNTER}.txt")
    elif [ "$seed" = "${EVAL_SEEDS[0]}" ]; then
        echo "Seed ${seed}: failed at every stage; skipping the remaining seeds"
        echo $(cat "${EVAL_OUTPUT_DIR}/final_eval_seed${seed}_${EVAL_COUNTER}.txt")
        break
    else
        echo "Seed ${seed}: failed at every stage; leaving it out of the mean"
    fi
done

if [ ${#SEED_RESULT_ARGS[@]} -gt 0 ]; then
    python src/utils/aggregate_seed_metrics.py \
        --output "${EVAL_DIR}/metrics.json" \
        --num-seeds "${#EVAL_SEEDS[@]}" \
        "${SEED_RESULT_ARGS[@]}" \
        || { echo "ERROR: failed to aggregate the per-seed metrics" >&2; exit 1; }
    cat "${EVAL_DIR}/metrics.json"
fi

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"
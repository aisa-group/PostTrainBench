#!/bin/bash
# The final evaluation of a trained model. It runs the task's final-eval script once per seed, each seed through a
# retry cascade that lowers max-tokens stage by stage, and writes the mean over the seeds that succeeded to
# <metrics_json> (see src/utils/aggregate_seed_metrics.py). src/run_task.sh runs it after the judges;
# dev_utils/test_evaluation/run_only_evaluation.sh and scripts/rerun_eval_n_times.sh run it on an existing result dir.
#
# Usage, from the repo root, with the .env variables set (src/commit_utils/set_env_vars.sh):
#   src/eval/run_final_eval.sh <task> <model_dir> <output_dir> <metrics_json> [<seed>...]
#   src/eval/run_final_eval.sh --check <task>
#
# - Without seeds, it uses FINAL_EVAL_SEEDS below. It uses only the first one for arenahardwriting and healthbench,
#   and for aime2025, gsm8k and humaneval when vLLM decodes the model greedily (see "Greedy models" below).
# - <output_dir> gets each seed's metrics (metrics_seed<S>.json) and the log of each attempt
#   (final_eval_seed<S>_<N>.txt). Neither it nor <metrics_json> may exist yet.
# - <model_dir> is not checked. A missing model fails at every stage like any broken model, and collect.py
#   recognizes that case by the final_eval_seed<S>_9.txt log.
# - If the first seed fails at every stage, the other seeds are skipped and no <metrics_json> is written. That is a
#   result, not an error (collect.py then uses the baseline), so the script exits 0. Every error exits non-zero.
# - --check only checks that <task> can be evaluated and prints the final-eval script it would run. run_task.sh calls
#   it at job start, so a missing file fails before the agent's run and not after it.
#
# The final-eval script is the task's evaluate_final_eval.py, or evaluate_openrouter_final_eval.py when .env gives
# arenahardwriting/healthbench an OPENROUTER_API_KEY but no OPENAI_API_KEY (the same rule as JUDGE_BACKEND in
# run_task.sh). It is evaluate.py plus --seed and any grading hardening the agent should not see (e.g. humaneval's
# scorer). It is never copied into the agent sandbox, which only gets evaluate.py.
#
# EVAL_RUNTIME selects where the evaluation runs:
#   apptainer (default)  condor: the eval container (vllm_debug.sif) with the HF cache overlay.
#   local                the caller is already inside the eval image (the Harbor verifier, which bakes this
#                        script and the task files into its image): Python runs directly, with the
#                        environment's HF_HOME. humaneval's answer sandbox is then with_answer_sandbox_local.sh
#                        (namespaces instead of a second apptainer container), or ANSWER_SANDBOX_LAUNCHER if set.
# EVAL_ATTEMPT_TIMEOUT_SEC bounds each evaluation attempt (default 28800, 8 h).
set -euo pipefail

# The fixed seeds of the final evaluation. arenahardwriting and healthbench are graded by paid API calls, so they use
# only the first seed.
FINAL_EVAL_SEEDS=(72332 87681 38992 92201 13818)

# Attempts per stage of the retry cascade: stage 0 uses the final-eval script's default max-tokens, stages 1 and 2
# lower it (STAGE_MAX_TOKENS_ARGS, set per task in setup_task).
STAGE_MAX_RETRIES=(4 3 2)

EVAL_RUNTIME="${EVAL_RUNTIME:-apptainer}"
case "${EVAL_RUNTIME}" in
    apptainer|local) ;;
    *) echo "ERROR: EVAL_RUNTIME must be apptainer or local, not '${EVAL_RUNTIME}'" >&2; exit 1 ;;
esac
EVAL_ATTEMPT_TIMEOUT_SEC="${EVAL_ATTEMPT_TIMEOUT_SEC:-28800}"
ANSWER_SANDBOX_LAUNCHER="${ANSWER_SANDBOX_LAUNCHER:-}"

# The HF cache overlay's path inside the eval container.
TMP_HF_CACHE="/tmp/hf_cache_90afd0"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    echo "usage: $0 <task> <model_dir> <output_dir> <metrics_json> [<seed>...]" >&2
    echo "       $0 --check <task>" >&2
    exit 1
}

# Sets EVAL_CONTAINER, FINAL_EVAL_SCRIPT, GRADER_API_KEY_NAME, STAGE_MAX_TOKENS_ARGS and DEFAULT_SEEDS for task $1.
# Exits if the task cannot be evaluated.
setup_task() {
    local task="$1"

    [ -f "src/eval/run_final_eval.sh" ] || die "run $0 from the repo root"
    [ -d "src/eval/tasks/${task}" ] || die "unknown task '${task}': src/eval/tasks/${task} not found"
    if [ "${EVAL_RUNTIME}" = "apptainer" ]; then
        [ -n "${POST_TRAIN_BENCH_CONTAINERS_DIR:-}" ] || die "POST_TRAIN_BENCH_CONTAINERS_DIR is not set"
        [ -n "${HF_HOME:-}" ] || die "HF_HOME is not set"

        # All evaluation runs in this container, and so does humaneval's answer sandbox.
        EVAL_CONTAINER="${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif"
        [ -f "${EVAL_CONTAINER}" ] || die "container ${EVAL_CONTAINER} not found"
    else
        EVAL_CONTAINER=""
        # Never run humaneval's answers without their sandbox.
        if [ "${task}" = "humaneval" ] && [ -z "${ANSWER_SANDBOX_LAUNCHER}" ] \
            && [ ! -f "src/eval/tasks/humaneval/with_answer_sandbox_local.sh" ]; then
            die "humaneval with EVAL_RUNTIME=local needs its answer sandbox (src/eval/tasks/humaneval/with_answer_sandbox_local.sh)"
        fi
    fi

    FINAL_EVAL_SCRIPT="evaluate_final_eval.py"
    GRADER_API_KEY_NAME=""
    case "${task}" in
        arenahardwriting|healthbench)
            if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
                FINAL_EVAL_SCRIPT="evaluate_openrouter_final_eval.py"
                GRADER_API_KEY_NAME="OPENROUTER_API_KEY"
            else
                [ -n "${OPENAI_API_KEY:-}" ] || die "${task} is graded through the API, but neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set"
                GRADER_API_KEY_NAME="OPENAI_API_KEY"
            fi
            ;;
    esac
    [ -f "src/eval/tasks/${task}/${FINAL_EVAL_SCRIPT}" ] \
        || die "src/eval/tasks/${task}/${FINAL_EVAL_SCRIPT} not found — required for the seeded final evaluation"

    case "${task}" in
        aime2025|gpqamain)
            STAGE_MAX_TOKENS_ARGS=("" "--max-tokens 12000" "--max-tokens 8000")
            ;;
        arenahardwriting|healthbench)
            STAGE_MAX_TOKENS_ARGS=("" "--max-new-tokens 12288" "--max-new-tokens 8192")
            ;;
        gsm8k|humaneval)
            STAGE_MAX_TOKENS_ARGS=("" "--max-tokens 3000" "--max-tokens 2000")
            ;;
        *)
            die "no max-tokens retry cascade for task '${task}': add one to STAGE_MAX_TOKENS_ARGS in $0"
            ;;
    esac

    case "${task}" in
        arenahardwriting|healthbench)
            DEFAULT_SEEDS=("${FINAL_EVAL_SEEDS[0]}")
            ;;
        *)
            DEFAULT_SEEDS=("${FINAL_EVAL_SEEDS[@]}")
            ;;
    esac

    # Whether the seed only drives sampling. gpqamain's seed also shuffles the answer choices, so its seeds give
    # different results even when the model decodes greedily.
    case "${task}" in
        gpqamain)
            SEED_ONLY_SAMPLES=0
            ;;
        *)
            SEED_ONLY_SAMPLES=1
            ;;
    esac
}

if [ "$#" -ge 1 ] && [ "$1" = "--check" ]; then
    [ "$#" -eq 2 ] || usage
    setup_task "$2"
    echo "${FINAL_EVAL_SCRIPT}"
    exit 0
fi

[ "$#" -ge 4 ] || usage
EVALUATION_TASK="$1"
setup_task "${EVALUATION_TASK}"
# Absolute paths, because the evaluation runs in the task's directory inside the container.
MODEL_DIR="$(realpath -ms "$2")"
OUTPUT_DIR="$(realpath -ms "$3")"
METRICS_JSON="$(realpath -ms "$4")"
shift 4
if [ "$#" -gt 0 ]; then
    SEEDS_GIVEN=1
    EVAL_SEEDS=("$@")
else
    SEEDS_GIVEN=0
    EVAL_SEEDS=("${DEFAULT_SEEDS[@]}")
fi

for seed in "${EVAL_SEEDS[@]}"; do
    [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]] || die "seed '${seed}' is not a non-negative integer without leading zeros"
done
DUPLICATE_SEEDS="$(printf '%s\n' "${EVAL_SEEDS[@]}" | sort | uniq -d)"
[ -z "${DUPLICATE_SEEDS}" ] || die "seeds given more than once: ${DUPLICATE_SEEDS//$'\n'/ }"

# A leftover metrics_seed<S>.json would count as a success, so both outputs must be new.
[ ! -e "${OUTPUT_DIR}" ] || die "${OUTPUT_DIR} already exists; move it away before evaluating"
[ ! -e "${METRICS_JSON}" ] || die "${METRICS_JSON} already exists; move it away before evaluating"
[ -d "$(dirname "${OUTPUT_DIR}")" ] || die "parent directory of ${OUTPUT_DIR} not found"
[ -d "$(dirname "${METRICS_JSON}")" ] || die "parent directory of ${METRICS_JSON} not found"
mkdir "${OUTPUT_DIR}"

REPO_ROOT="$(pwd)"
TMP_SUBDIR="$(mktemp -d /tmp/ptb_final_eval.XXXXXX)"
HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
# run_evaluation runs in its own bash and sees only exported variables.
export EVALUATION_TASK MODEL_DIR OUTPUT_DIR EVAL_CONTAINER FINAL_EVAL_SCRIPT GRADER_API_KEY_NAME TMP_HF_CACHE HF_HOME \
    REPO_ROOT TMP_SUBDIR HF_MERGED EVAL_RUNTIME ANSWER_SANDBOX_LAUNCHER
if [ -n "${GRADER_API_KEY_NAME}" ]; then
    export "${GRADER_API_KEY_NAME}"
fi

cleanup() {
    # An attempt that its timeout killed leaves its HF overlay mounted.
    if mountpoint -q "${HF_MERGED}"; then
        fusermount -u "${HF_MERGED}"
    fi
    rm -rf "${TMP_SUBDIR}"
}
trap cleanup EXIT

# Greedy models. The inspect tasks set no temperature, so vLLM uses the model's default temperature from its generation
# config (see src/utils/default_temperature.py). When that is 0, vLLM decodes greedily, and a seed that only drives
# sampling does not change the result, apart from vLLM's small run-to-run noise. Such a model is evaluated with the
# first default seed only. Seeds given on the command line are always used as given. A missing model is not checked:
# it fails at the first seed, and the other seeds are skipped anyway.
# Writes "<greedy|sampling> <temperature>" for the model to default_temperature.txt, in the eval container.
probe_default_temperature() {
    if [ "${EVAL_RUNTIME}" = "local" ]; then
        PYTHONNOUSERSITE="1" python src/utils/default_temperature.py "${MODEL_DIR}" \
            --output-file "${OUTPUT_DIR}/default_temperature.txt"
        return
    fi
    apptainer exec \
        --env PYTHONNOUSERSITE="1" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --pwd "${REPO_ROOT}" \
        "${EVAL_CONTAINER}" python src/utils/default_temperature.py "${MODEL_DIR}" \
            --output-file "${OUTPUT_DIR}/default_temperature.txt"
}

if [ "${SEEDS_GIVEN}" = 0 ] && [ "${SEED_ONLY_SAMPLES}" = 1 ] && [ "${#EVAL_SEEDS[@]}" -gt 1 ] && [ -d "${MODEL_DIR}" ]; then
    if probe_default_temperature > "${OUTPUT_DIR}/default_temperature_log.txt" 2>&1; then
        read -r DECODING DEFAULT_TEMPERATURE < "${OUTPUT_DIR}/default_temperature.txt"
        echo "vLLM's default temperature for this model: ${DEFAULT_TEMPERATURE} (${DECODING})"
        case "${DECODING}" in
            greedy)
                EVAL_SEEDS=("${EVAL_SEEDS[0]}")
                ;;
            sampling)
                ;;
            *)
                die "unexpected output of default_temperature.py: '${DECODING} ${DEFAULT_TEMPERATURE}'"
                ;;
        esac
    else
        # vLLM may still load the model, so evaluate with all seeds rather than lose the evaluation.
        echo "WARNING: could not read vLLM's default temperature for ${MODEL_DIR}" \
            "(see ${OUTPUT_DIR}/default_temperature_log.txt); evaluating with all seeds" >&2
    fi
fi

echo "Final evaluation of ${MODEL_DIR} on ${EVALUATION_TASK} with ${FINAL_EVAL_SCRIPT}, seeds: ${EVAL_SEEDS[*]}"

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

# One evaluation attempt. It runs in its own bash under a timeout (see run_evaluation_with_retry).
run_evaluation() {
    local max_tokens_arg="$1"
    local seed="$2"
    local eval_num="$3"
    # humaneval's scorer runs the model's answers in a separate container (see with_answer_sandbox.sh).
    local answer_sandbox=()
    if [ "${EVALUATION_TASK}" = "humaneval" ]; then
        answer_sandbox=(bash src/eval/tasks/humaneval/with_answer_sandbox.sh "${EVAL_CONTAINER}")
    fi
    local grader_key_env=()
    if [ -n "${GRADER_API_KEY_NAME}" ]; then
        grader_key_env=(--env "${GRADER_API_KEY_NAME}=${!GRADER_API_KEY_NAME}")
    fi
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
    if [ "${EVAL_RUNTIME}" = "local" ]; then
        # Already in the eval image: the same script and arguments, run directly (the grader key is in the
        # environment); humaneval's answers run in with_answer_sandbox_local.sh's sandbox.
        if [ "${EVALUATION_TASK}" = "humaneval" ]; then
            if [ -n "${ANSWER_SANDBOX_LAUNCHER}" ]; then
                read -r -a answer_sandbox <<< "${ANSWER_SANDBOX_LAUNCHER}"
            else
                answer_sandbox=(bash "${REPO_ROOT}/src/eval/tasks/humaneval/with_answer_sandbox_local.sh")
            fi
        fi
        ( cd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}" \
            && ${answer_sandbox[@]+"${answer_sandbox[@]}"} env VLLM_API_KEY="inspectai" PYTHONNOUSERSITE="1" \
                python "${FINAL_EVAL_SCRIPT}" \
                    --model-path "${MODEL_DIR}" \
                    --templates-dir ../../../../src/eval/templates \
                    --limit -1 \
                    --seed "${seed}" \
                    ${max_tokens_arg} \
                    --json-output-file "${OUTPUT_DIR}/metrics_seed${seed}.json" ) > "${OUTPUT_DIR}/final_eval_seed${seed}_${eval_num}.txt"
        return
    fi
    with_huggingface_overlay "${answer_sandbox[@]}" apptainer exec \
        --nv \
        --env "HF_HOME=${TMP_HF_CACHE}" \
        "${grader_key_env[@]}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}" \
        "${EVAL_CONTAINER}" python "${FINAL_EVAL_SCRIPT}" \
            --model-path "${MODEL_DIR}" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            --seed "${seed}" \
            ${max_tokens_arg} \
            --json-output-file "${OUTPUT_DIR}/metrics_seed${seed}.json" > "${OUTPUT_DIR}/final_eval_seed${seed}_${eval_num}.txt"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"
    local seed="$3"
    local metrics_file="${OUTPUT_DIR}/metrics_seed${seed}.json"
    local attempt
    local status

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${metrics_file}" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Seed ${seed}: evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        status=0
        timeout --signal=TERM --kill-after=60s "${EVAL_ATTEMPT_TIMEOUT_SEC}s" bash -c "$(declare -f run_evaluation with_huggingface_overlay); run_evaluation \"$max_tokens_arg\" \"$seed\" \"$EVAL_COUNTER\"" \
            || status=$?

        if [ -f "${metrics_file}" ]; then
            return 0
        fi
        echo "Seed ${seed}: attempt $EVAL_COUNTER wrote no metrics (exit status ${status})"
    done

    return 1
}

# Runs the retry cascade for one seed, from stage $2 onwards. On success, sets SEED_STAGE to the stage that produced
# the seed's metrics.
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

# The first seed runs the full cascade. If it fails at every stage, the other seeds are skipped. The later seeds start
# at the stage where the first seed succeeded, so they share its max-tokens setting; a later seed that fails at every
# stage is left out of the mean.
SEED_RESULT_ARGS=()
START_STAGE=0
for seed in "${EVAL_SEEDS[@]}"; do
    EVAL_COUNTER=0
    echo "=== Seed ${seed}: evaluating from stage ${START_STAGE} ==="
    if run_seed_evaluation "$seed" "$START_STAGE"; then
        echo "Seed ${seed}: succeeded at stage ${SEED_STAGE}"
        SEED_RESULT_ARGS+=(--seed-result "$seed" "$SEED_STAGE" "${OUTPUT_DIR}/metrics_seed${seed}.json")
        if [ "$seed" = "${EVAL_SEEDS[0]}" ]; then
            START_STAGE=$SEED_STAGE
        fi
        echo $(cat "${OUTPUT_DIR}/final_eval_seed${seed}_${EVAL_COUNTER}.txt")
    elif [ "$seed" = "${EVAL_SEEDS[0]}" ]; then
        echo "Seed ${seed}: failed at every stage; skipping the remaining seeds"
        echo $(cat "${OUTPUT_DIR}/final_eval_seed${seed}_${EVAL_COUNTER}.txt")
        break
    else
        echo "Seed ${seed}: failed at every stage; leaving it out of the mean"
    fi
done

if [ ${#SEED_RESULT_ARGS[@]} -eq 0 ]; then
    echo "No seed succeeded; ${METRICS_JSON} not written"
    exit 0
fi

python src/utils/aggregate_seed_metrics.py \
    --output "${METRICS_JSON}" \
    --num-seeds "${#EVAL_SEEDS[@]}" \
    "${SEED_RESULT_ARGS[@]}" \
    || die "failed to aggregate the per-seed metrics"
cat "${METRICS_JSON}"

#!/bin/bash
# GPU smoke test of the per-sample seeds (src/eval/per_sample_seed.py) in every task's final-eval script: runs each
# evaluate_final_eval.py on a few samples of one model, with the same container, HF-cache overlay and answer sandbox as
# src/eval/run_final_eval.sh, then check_smoke_test.py checks that each request carried its own seed. arenahardwriting
# and healthbench make a few paid grader calls (OPENAI_API_KEY from .env).
#
# Usage (from the repo root, on a GPU node; see smoke_test.sub):
#   dev_utils/per_sample_seed/smoke_test.sh <model_dir> <out_dir> [<task>...]
set -euo pipefail

[ "$#" -ge 2 ] || { echo "usage: $0 <model_dir> <out_dir> [<task>...]" >&2; exit 1; }
MODEL_DIR="$(realpath -ms "$1")"
OUT_DIR="$(realpath -ms "$2")"
shift 2
TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=(aime2025 gsm8k humaneval gpqamain arenahardwriting healthbench)
SEED=38992

[ -d "${MODEL_DIR}" ] || { echo "ERROR: ${MODEL_DIR} not found" >&2; exit 1; }
[ ! -e "${OUT_DIR}" ] || { echo "ERROR: ${OUT_DIR} already exists" >&2; exit 1; }
source src/commit_utils/set_env_vars.sh
mkdir -p "${OUT_DIR}"

REPO_ROOT="$(pwd)"
EVAL_CONTAINER="${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif"
TMP_SUBDIR="$(mktemp -d /tmp/ptb_seed_smoke.XXXXXX)"
trap 'mountpoint -q "${TMP_SUBDIR}/merged" && fusermount -u "${TMP_SUBDIR}/merged"; rm -rf "${TMP_SUBDIR}"' EXIT
export VLLM_CACHE_ROOT="${TMP_SUBDIR}/vllm_cache"
export XDG_DATA_HOME="${TMP_SUBDIR}/xdg_data"

limit_of() {
    case "$1" in
        aime2025) echo 10 ;;
        gsm8k) echo 60 ;;
        humaneval|gpqamain) echo 20 ;;
        arenahardwriting|healthbench) echo 3 ;;
        *) echo "ERROR: no limit for task $1" >&2; exit 1 ;;
    esac
}

for task in "${TASKS[@]}"; do
    echo "=== ${task}"
    mkdir -p "${OUT_DIR}/${task}"
    mkdir -p "${TMP_SUBDIR}/merged" "${TMP_SUBDIR}/upper" "${TMP_SUBDIR}/work"
    fuse-overlayfs -o "lowerdir=${HF_HOME},upperdir=${TMP_SUBDIR}/upper,workdir=${TMP_SUBDIR}/work" "${TMP_SUBDIR}/merged"
    wrapper=()
    [ "${task}" != "humaneval" ] || wrapper=(bash src/eval/tasks/humaneval/with_answer_sandbox.sh "${EVAL_CONTAINER}")
    key_env=()
    case "${task}" in
        arenahardwriting|healthbench) key_env=(--env "OPENAI_API_KEY=${OPENAI_API_KEY}") ;;
    esac
    status=0
    INSPECT_LOG_DIR="${OUT_DIR}/${task}/inspect_logs" "${wrapper[@]}" apptainer exec \
        --nv \
        --env "HF_HOME=/tmp/hf_cache_90afd0" \
        "${key_env[@]}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${TMP_SUBDIR}/merged:/tmp/hf_cache_90afd0" \
        --pwd "${REPO_ROOT}/src/eval/tasks/${task}" \
        "${EVAL_CONTAINER}" python evaluate_final_eval.py \
            --model-path "${MODEL_DIR}" \
            --templates-dir ../../../../src/eval/templates \
            --limit "$(limit_of "${task}")" \
            --seed "${SEED}" \
            --json-output-file "${OUT_DIR}/${task}/metrics.json" > "${OUT_DIR}/${task}.log" 2>&1 || status=$?
    fusermount -u "${TMP_SUBDIR}/merged"
    rm -rf "${TMP_SUBDIR}/merged" "${TMP_SUBDIR}/upper" "${TMP_SUBDIR}/work"
    echo "${task}: exit ${status}"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
done

apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --bind "${OUT_DIR}:${OUT_DIR}" --pwd "${REPO_ROOT}" "${EVAL_CONTAINER}" \
    python dev_utils/per_sample_seed/check_smoke_test.py "${OUT_DIR}" "${SEED}" "${TASKS[@]}"

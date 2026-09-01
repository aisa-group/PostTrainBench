#!/bin/bash
# Re-run the per-task evaluate.py N times on an already-finished EVAL_DIR
# and aggregate per-run metrics into <EVAL_DIR>/metrics_averaged.json.
#
# Usage:
#   scripts/rerun_eval_n_times.sh <EVAL_DIR> [N]
#
# Defaults: N=5.
#
# Mirrors run_task.sh's evaluation step: runs src/eval/tasks/<task>/evaluate.py
# (NOT the snapshot in $EVAL_DIR/task) under the same vllm_debug container with
# the same fuse-overlayfs HF cache and the same max-tokens fallback ladder.
#
# Run from the repo root, on a node with GPUs (submit via
# src/commit_utils/rerun_eval.sub for cluster execution).
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <EVAL_DIR> [N]" >&2
    exit 1
fi

EVAL_DIR="$(realpath "$1")"
N="${2:-5}"

if [ ! -d "$EVAL_DIR/final_model" ]; then
    echo "ERROR: $EVAL_DIR/final_model not found" >&2
    exit 1
fi

source src/commit_utils/set_env_vars.sh

# Derive the task name from the EVAL_DIR basename: <task>_<model_safe>_<cluster_id>.
EVAL_BASENAME="$(basename "$EVAL_DIR")"
EVALUATION_TASK="${EVAL_BASENAME%%_*}"

if [ ! -f "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" ]; then
    echo "ERROR: src/eval/tasks/${EVALUATION_TASK}/evaluate.py not found" >&2
    echo "       (parsed task '${EVALUATION_TASK}' from $(basename "$EVAL_DIR"))" >&2
    exit 1
fi

REPO_ROOT="$(pwd)"
RERUNS_DIR="$EVAL_DIR/reruns"
mkdir -p "$RERUNS_DIR"

# Per-task max-tokens fallback ladder, mirroring run_task.sh.
case "$EVALUATION_TASK" in
    aime2025)         FB1="--max-tokens 12000";    FB2="--max-tokens 8000" ;;
    arenahardwriting) FB1="--max-new-tokens 12288"; FB2="--max-new-tokens 8192" ;;
    bfcl)             FB1="--max-tokens 12000";    FB2="--max-tokens 8000" ;;
    gpqamain)         FB1="--max-tokens 12000";    FB2="--max-tokens 8000" ;;
    gsm8k)            FB1="--max-tokens 3000";     FB2="--max-tokens 2000" ;;
    healthbench)      FB1="--max-new-tokens 12288"; FB2="--max-new-tokens 8192" ;;
    humaneval)        FB1="--max-tokens 3000";     FB2="--max-tokens 2000" ;;
    *)                FB1="";                       FB2="" ;;
esac

# Fuse-overlayfs HF cache so reruns don't pollute the shared HF cache,
# matching run_task.sh's with_huggingface_overlay helper.
TMP_SUBDIR="/tmp/rerun_eval_$$"
HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
TMP_HF_CACHE="/tmp/hf_cache_rerun_$$"

setup_overlay() {
    mkdir -p "${TMP_SUBDIR}/upper_huggingface"
    mkdir -p "${TMP_SUBDIR}/fuse_workdir"
    mkdir -p "${HF_MERGED}"
    fuse-overlayfs -o \
        "lowerdir=${HF_HOME},upperdir=${TMP_SUBDIR}/upper_huggingface,workdir=${TMP_SUBDIR}/fuse_workdir" \
        "${HF_MERGED}"
}

teardown_overlay() {
    fusermount -u "${HF_MERGED}" 2>/dev/null || true
    rm -rf "${TMP_SUBDIR}" 2>/dev/null || true
}
trap teardown_overlay EXIT

setup_overlay

run_one() {
    local out_json="$1"
    local extra="$2"
    local log="$3"

    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    sleep 5

    timeout --signal=TERM --kill-after=60s 28800s \
    apptainer exec \
        --nv \
        --env "HF_HOME=${TMP_HF_CACHE}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}" \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
        python evaluate.py \
            --model-path "${EVAL_DIR}/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${extra} \
            --json-output-file "${out_json}" >"${log}" 2>&1
}

run_with_fallback() {
    local out_json="$1"
    local log_prefix="$2"

    rm -f "$out_json"

    for level in default fb1 fb2; do
        local extra=""
        case "$level" in
            default) extra="" ;;
            fb1)     extra="$FB1" ;;
            fb2)     extra="$FB2" ;;
        esac
        echo "  [$level] extra='${extra}'"
        run_one "$out_json" "$extra" "${log_prefix}_${level}.log" || true
        if [ -f "$out_json" ]; then
            return 0
        fi
    done
    return 1
}

echo "EVAL_DIR=${EVAL_DIR}"
echo "EVALUATION_TASK=${EVALUATION_TASK}"
echo "N=${N}"

for i in $(seq 1 "$N"); do
    out="${RERUNS_DIR}/run_${i}.json"
    log_prefix="${RERUNS_DIR}/run_${i}"
    echo "=== rerun ${i} / ${N} ==="
    if run_with_fallback "$out" "$log_prefix"; then
        echo "  -> wrote $out"
    else
        echo "  -> FAILED all fallbacks for rerun ${i}"
    fi
done

python scripts/aggregate_metrics_runs.py \
    --runs-glob "${RERUNS_DIR}/run_*.json" \
    --output "${EVAL_DIR}/metrics_averaged.json"

echo "Wrote ${EVAL_DIR}/metrics_averaged.json"

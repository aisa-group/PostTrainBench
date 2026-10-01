#!/bin/bash
# Runs test_scorer.py, and compare_scorers.py on the given inspect logs, in the eval container next to an answer
# sandbox (src/eval/tasks/humaneval/with_answer_sandbox.sh). Submit from the repo root with scorer_tests.sub.
# Usage: run_scorer_tests.sh <container.sif> <hf_home> <out_dir> [log.json ...]
set -euo pipefail

SIF="$1"
HOST_HF_HOME="$2"
OUT_DIR="$3"
shift 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${REPO_ROOT}/src/eval/tasks/humaneval/with_answer_sandbox.sh"
mkdir -p "$OUT_DIR"

# Under HTCondor, use the job's local scratch dir: a node's /tmp can be full.
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ]; then
    export TMPDIR="$_CONDOR_SCRATCH_DIR"
fi

# The HF cache is mounted through fuse-overlayfs, as in run_task.sh, so nothing is written to the shared cache.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb_scorer_tests.XXXXXX")"
mkdir "$WORK/upper" "$WORK/work" "$WORK/merged"
fuse-overlayfs -o "lowerdir=${HOST_HF_HOME},upperdir=${WORK}/upper,workdir=${WORK}/work" "${WORK}/merged"
trap 'fusermount -u "${WORK}/merged"; rm -rf "$WORK"' EXIT

in_eval_container() {
    bash "$WRAPPER" "$SIF" apptainer exec --cleanenv \
        --env HF_HOME=/tmp/hf_cache_scorer_tests --env HF_DATASETS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
        --env PYTHONNOUSERSITE=1 --env HOST_HF_HOME="$HOST_HF_HOME" \
        --bind "${WORK}/merged:/tmp/hf_cache_scorer_tests" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${OUT_DIR}:${OUT_DIR}" \
        --pwd "$REPO_ROOT" \
        "$SIF" python "$@"
}

status=0
in_eval_container dev_utils/humaneval_early_exit/test_scorer.py 2>&1 | tee "${OUT_DIR}/test_scorer.txt" || status=1
if [ $# -gt 0 ]; then
    in_eval_container dev_utils/humaneval_early_exit/compare_scorers.py --out "${OUT_DIR}/compare.jsonl" "$@" 2>&1 \
        | tee "${OUT_DIR}/compare.txt" || status=1
fi
exit "$status"

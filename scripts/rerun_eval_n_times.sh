#!/bin/bash
# Re-runs the final evaluation of src/run_task.sh (src/eval/run_final_eval.sh) on an already-finished EVAL_DIR,
# without touching its metrics.json. The per-seed metrics and logs go to <EVAL_DIR>/reruns/ and the mean over the
# seeds to <EVAL_DIR>/metrics_averaged.json; neither may exist yet.
#
# Usage:
#   scripts/rerun_eval_n_times.sh <EVAL_DIR> [--seeds <seed>...]
#
# Without --seeds, it uses the final evaluation's default seeds, so it repeats run_task.sh's evaluation. Pass other
# seeds to sample beyond them.
#
# Run from the repo root, on a node with GPUs (submit via src/commit_utils/rerun_eval.sub for cluster execution).
set -euo pipefail

usage() {
    echo "usage: $0 <EVAL_DIR> [--seeds <seed>...]" >&2
    exit 1
}

[ "$#" -ge 1 ] || usage
EVAL_DIR="$(realpath "$1")"
shift
SEEDS=()
if [ "$#" -gt 0 ]; then
    [ "$1" = "--seeds" ] && [ "$#" -ge 2 ] || usage
    shift
    SEEDS=("$@")
fi

if [ ! -d "$EVAL_DIR/final_model" ]; then
    echo "ERROR: $EVAL_DIR/final_model not found" >&2
    exit 1
fi

source src/commit_utils/set_env_vars.sh

# Derive the task name from the EVAL_DIR basename: <task>_<model_safe>_<cluster_id>.
EVAL_BASENAME="$(basename "$EVAL_DIR")"
EVALUATION_TASK="${EVAL_BASENAME%%_*}"

# On exit, give the users in POST_TRAIN_BENCH_USERS_ACCESS access to what this wrote (src/utils/grant_access.py).
GRANT_ACCESS="$(pwd)/src/utils/grant_access.py"
trap 'python3 "${GRANT_ACCESS}" "${EVAL_DIR}"' EXIT

bash src/eval/run_final_eval.sh "${EVALUATION_TASK}" "${EVAL_DIR}/final_model" "${EVAL_DIR}/reruns" \
    "${EVAL_DIR}/metrics_averaged.json" "${SEEDS[@]}"

if [ ! -f "${EVAL_DIR}/metrics_averaged.json" ]; then
    echo "ERROR: the first seed failed at every stage, so no metrics were written (logs in ${EVAL_DIR}/reruns)" >&2
    exit 1
fi
echo "Wrote ${EVAL_DIR}/metrics_averaged.json"

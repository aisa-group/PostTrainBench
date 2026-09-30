#!/bin/bash
# One condor job of scripts/rerun_final_eval_parallel.py: the final evaluation of one seed of one cell, i.e.
# `src/eval/run_final_eval.sh --single-seed` into <cell_dir>/reruns/. The seed that finishes last aggregates the
# cell's seeds into <cell_dir>/metrics_averaged.json (rerun_final_eval_parallel.py aggregate --if-complete).
#
# <cell_dir> is a result dir (in-place rerun) or a mirror dir whose final_model is a symlink to the result dir's;
# <cell_dir>/reruns/plan.json (written by the submit) names the task and the seeds.
#
# Usage: scripts/rerun_final_eval_seed.sh <cell_dir> <seed> <condor_job_id>   (from the repo root, on a GPU node)
set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[ "$#" -eq 3 ] || die "usage: $0 <cell_dir> <seed> <condor_job_id>"
CELL_DIR="$1"
SEED="$2"
JOB_ID="$3"
RERUNS="${CELL_DIR}/reruns"
JOBS_LOG="${RERUNS}/seed${SEED}_jobs.txt"
[ -f "${RERUNS}/plan.json" ] || die "${RERUNS}/plan.json not found"
TASK="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["task"])' "${RERUNS}/plan.json")"
python3 -c 'import json, sys; assert int(sys.argv[2]) in json.load(open(sys.argv[1]))["seeds"], sys.argv[2]' \
    "${RERUNS}/plan.json" "${SEED}" || die "seed ${SEED} is not planned in ${RERUNS}/plan.json"

MODEL_DIR="${CELL_DIR}/final_model"
[ -d "${MODEL_DIR}/" ] || die "${MODEL_DIR} is not a directory"
# vLLM reports an unreadable weights file as "No such file", which would look like a broken model.
UNREADABLE="$(find "${MODEL_DIR}/" -type f ! -readable)"
[ -z "${UNREADABLE}" ] || die "unreadable model files: ${UNREADABLE}"

source src/commit_utils/set_env_vars.sh

JOB_TMP="$(mktemp -d /tmp/ptb_rerun_seed.XXXXXX)"
GRANT_ACCESS="$(pwd)/src/utils/grant_access.py"
trap 'rm -rf "${JOB_TMP}"; python3 "${GRANT_ACCESS}" "${CELL_DIR}"' EXIT
# Job-local caches: ~/.cache/vllm would grow in the home quota, and inspect's shared trace dir makes concurrent evals
# delete each other's trace files ("Stale file handle" tracebacks in every log call).
export VLLM_CACHE_ROOT="${JOB_TMP}/vllm_cache"
export XDG_DATA_HOME="${JOB_TMP}/xdg_data"
export INSPECT_LOG_DIR="${RERUNS}/inspect_logs/seed${SEED}"

# A job that condor restarted (machine failure, eviction) finds its own interrupted attempt: the last line of this
# seed's jobs log is its own start line. Move that attempt aside; any other leftover makes run_final_eval.sh refuse.
if [ ! -e "${RERUNS}/result_seed${SEED}.txt" ] && [ -f "${JOBS_LOG}" ] \
        && tail -n 1 "${JOBS_LOG}" | grep -q " job=${JOB_ID} .* start$"; then
    ATTEMPT_DIR="${RERUNS}/failed_attempts/$(date +%Y%m%dT%H%M%S)_seed${SEED}_job${JOB_ID}_restarted"
    mkdir -p "${ATTEMPT_DIR}"
    for f in "${RERUNS}"/*_seed"${SEED}"[._]* "${RERUNS}/inspect_logs/seed${SEED}"; do
        [ ! -e "$f" ] || mv "$f" "${ATTEMPT_DIR}/"
    done
    echo "$(date -Is) job=${JOB_ID} moved its interrupted attempt to ${ATTEMPT_DIR}" >> "${JOBS_LOG}"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "$(date -Is) job=${JOB_ID} host=$(hostname) gpu=${GPU_NAME} start" >> "${JOBS_LOG}"
bash src/eval/run_final_eval.sh --single-seed "${TASK}" "${MODEL_DIR}" "${RERUNS}" "${SEED}"
echo "$(date -Is) job=${JOB_ID} host=$(hostname) done: $(cat "${RERUNS}/result_seed${SEED}.txt")" >> "${JOBS_LOG}"

python3 scripts/rerun_final_eval_parallel.py aggregate --if-complete "${CELL_DIR}"

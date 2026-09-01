#!/bin/bash

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
NUM_GPUS="${7:-1}"

# Both `source` lines in this file resolve against this script's own directory rather
# than the working directory, so that a launcher may run a node-local copy of src/ while
# leaving the working directory on the shared checkout.
#
# Why that matters: bash reads a script incrementally and seeks back to the byte after
# the last command it parsed, so a long-running script holds an open handle on its own
# inode for its whole run. Jobs 82165 and 82166 were killed by that. Both started at
# 07:44:28 on 2026-08-30 and committing 2775447 replaced this file at 08:28:29, 44
# minutes in; each job survived its entire agent phase and then died the moment bash
# next needed to read -- 82165 after 10:01:47, 82166 after 08:27:27 -- with
#
#     src/run_task.sh: error reading input file: Stale file handle
#
# Nineteen H100-hours of agent work, both with a finished final_model/ on node-local
# disk, and neither was scored. The working directory still has to be the checkout:
# line 542 takes REPO_ROOT from `pwd` and the scoring container binds it by that path.
_RUN_TASK_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${_RUN_TASK_SRC}/commit_utils/set_env_vars.sh"

# Select the judge backend for grader-based benchmarks (arenahardwriting / healthbench):
# default to the OpenAI-backed evaluate.py, but fall back to the OpenRouter variant when
# .env provides OPENROUTER_API_KEY but no OPENAI_API_KEY.
JUDGE_BACKEND="openai"
if { [ "$EVALUATION_TASK" = "arenahardwriting" ] || [ "$EVALUATION_TASK" = "healthbench" ]; } \
   && [ -z "${OPENAI_API_KEY}" ] && [ -n "${OPENROUTER_API_KEY}" ]; then
    JUDGE_BACKEND="openrouter"
fi

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    export EVAL_SCRIPT="evaluate_openrouter.py"
else
    export EVAL_SCRIPT="evaluate.py"
fi

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:[]' '____')

AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:[]' '____')

RANDOM_UUID=$(uuidgen)

GPU_SUFFIX=""
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    GPU_SUFFIX="_${NUM_GPUS}gpu"
fi

export EVAL_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${GPU_SUFFIX}${POST_TRAIN_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p ${EVAL_DIR}

exec 1>${EVAL_DIR}/output.log
exec 2>${EVAL_DIR}/error.log

echo "$@"
echo "Judge backend: ${JUDGE_BACKEND} (eval script: ${EVAL_SCRIPT})"

# Scratch root for the container's /tmp, the job dir and the merged HF cache.
# The HTCondor submit file asks for request_disk=400G and gets it on /tmp; on a
# scheduler that makes no such reservation, node-local /tmp can be far smaller
# than the run needs (87 GB free on this cluster's a3 nodes). Unset means the
# upstream /tmp, so this changes nothing unless a site opts in.
export POST_TRAIN_BENCH_TMP_ROOT="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
export TMP_SUBDIR="${POST_TRAIN_BENCH_TMP_ROOT}/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

echo "Preparing job directory..." 
mkdir -p "${JOB_DIR}"

mkdir "${JOB_DIR}/task"

cp "src/eval/tasks/${EVALUATION_TASK}/${EVAL_SCRIPT}" "${JOB_DIR}/task/evaluate.py"
if [ -d "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task"
fi
cp -r src/eval/templates "${JOB_DIR}/task/"

if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r src/eval/tasks/${EVALUATION_TASK}/task_context/* "${JOB_DIR}/task"
fi
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

BENCHMARK=$(cat src/eval/tasks/${EVALUATION_TASK}/benchmark.txt)
PROMPT=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL_TO_TRAIN" --benchmark-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --num-gpus "$NUM_GPUS" --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh

# set openai api keys appropriately
export CODEX_API_KEY="${OPENAI_API_KEY}"
unset OPENAI_API_KEY
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

# --- API key allowlist ----------------------------------------------------
# Each agent declares the third-party API keys it may receive in
# agents/<agent>/api_keys.json; each benchmark declares the keys its grading
# needs in src/eval/tasks/<task>/info.json ("required_api_keys", default none).
# The agent sandbox is launched with -c --cleanenv, so it inherits NOTHING from
# the host: it receives ONLY the union of those two sets via --env (built into
# API_KEY_ENV_ARGS below). Every other provider key is never passed in, hence
# unset inside the sandbox. OPENAI_API_KEY thus reaches the agent only for the
# arenahardwriting/healthbench benchmarks (which list it in required_api_keys).
if ! ALLOWED_API_KEYS_RAW="$(python3 -c '
import json, sys
agent_keys = json.load(open(sys.argv[1]))["allowed_api_keys"]
bench = json.load(open(sys.argv[2]))
seen, out = set(), []
for k in list(agent_keys) + bench.get("required_api_keys", []):
    if k not in seen:
        seen.add(k); out.append(k)
print("\n".join(out))
' "agents/${AGENT}/api_keys.json" "src/eval/tasks/${EVALUATION_TASK}/info.json")"; then
    echo "ERROR: failed to compute API key allowlist for agent=${AGENT} task=${EVALUATION_TASK}" >&2
    exit 1
fi

ALLOWED_API_KEYS=()
[ -n "$ALLOWED_API_KEYS_RAW" ] && mapfile -t ALLOWED_API_KEYS <<< "$ALLOWED_API_KEYS_RAW"

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    ALLOWED_API_KEYS+=("OPENROUTER_API_KEY")
fi

API_KEY_ENV_ARGS=()
for _k in "${ALLOWED_API_KEYS[@]}"; do
    API_KEY_ENV_ARGS+=(--env "${_k}=${!_k}")
done
echo "API keys provisioned for agent=${AGENT} task=${EVALUATION_TASK}: ${ALLOWED_API_KEYS[*]:-<none>}"

# Copy scripts needed inside the container
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp src/utils/system_monitor.sh "${JOB_DIR}/system_monitor.sh"
cp src/utils/timestamp_lines.py "${JOB_DIR}/timestamp_lines.py"
cp src/utils/update_agent_cli.sh "${JOB_DIR}/update_agent_cli.sh"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

# An agent that is more than one shell script needs its own files inside the
# sandbox. The agent container runs with `-c --cleanenv` and
# `--home "${JOB_DIR}:/home/ben"`, so nothing outside JOB_DIR is reachable from
# in there -- not this checkout, not the launching user's home. An agent that
# ships a payload/ directory gets it copied to /home/ben/agent, and gets the
# three facts about the task that solve.sh would otherwise have to parse back
# out of $PROMPT. Guarded on the directory existing: no agent shipped today has
# one, so for all of them this block does nothing at all.
if [ -d "agents/${AGENT}/payload" ]; then
    cp -r "agents/${AGENT}/payload" "${JOB_DIR}/agent"
    echo "agent payload: $(du -sh "${JOB_DIR}/agent" | cut -f1) -> /home/ben/agent"
fi

# Agents that authenticate through something other than a provider API key --
# a Vertex or Bedrock endpoint reached with the host's ambient credentials, say
# -- name the variables they need, one per line, in
# agents/<agent>/env_passthrough.txt. Only the NAMES are in the repository; the
# values come from the launching environment and are never written to disk here.
# This is deliberately separate from the api_keys.json allowlist above: that one
# governs provider secrets the benchmark provisions, and rule 9 of the prompt is
# about those. These are the agent's own routing configuration. Guarded on the
# file existing, so it is a no-op for every agent shipped today.
AGENT_ENV_ARGS=()
if [ -f "agents/${AGENT}/env_passthrough.txt" ]; then
    _forwarded=()
    while IFS= read -r _v || [ -n "$_v" ]; do
        _v="${_v%%#*}"; _v="${_v//[[:space:]]/}"
        [ -n "$_v" ] || continue
        [ -n "${!_v:-}" ] || continue
        AGENT_ENV_ARGS+=(--env "${_v}=${!_v}")
        _forwarded+=("$_v")
    done < "agents/${AGENT}/env_passthrough.txt"
    echo "agent env forwarded for ${AGENT}: ${_forwarded[*]:-<none set in this environment>}"
fi

# Self-decontamination tooling for the agent: the same n-gram checker and
# test-set copy the contamination judge gets, at the same paths (the judge
# phase re-copies both via prepare_judge_sandbox, so the judges never run
# agent-modified versions). The matching usage instructions are added to the
# agent prompt by get_prompt.py.
if [ ! -f "src/eval/tasks/${EVALUATION_TASK}/test_data.json" ]; then
    echo "ERROR: src/eval/tasks/${EVALUATION_TASK}/test_data.json not found — required for the agent's decontamination tooling" >&2
    exit 1
fi
cp src/judges/judge_tools/contamination_check.py "${JOB_DIR}/contamination_check.py"
cp "src/eval/tasks/${EVALUATION_TASK}/test_data.json" "${JOB_DIR}/test_data.json"

# Agent-specific auth: auth.json is bind-mounted at apptainer exec time so the
# codex CLI can write the rotated refresh token back to the shared source file
# (single-use refresh tokens otherwise burn after one job).
AGENT_AUTH_SRC=""
if [ -f "agents/${AGENT}/auth.json" ]; then
    AGENT_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/auth.json")" && pwd)/auth.json"
    # Placeholder file inside the sandbox .codex dir for the bind mount to overlay.
    mkdir -p "${JOB_DIR}/.codex"
    : > "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${JOB_DIR}/oauth_token"
fi

# Cursor CLI persists its OAuth tokens at ~/.config/cursor/auth.json. Bind-mount
# the agent's copy so the CLI inside the sandbox reads and rotates against the
# shared credential file. Distinct filename in the agent dir avoids collision
# with the codex auth.json check above.
CURSOR_AUTH_SRC=""
if [ -f "agents/${AGENT}/cursor_auth.json" ]; then
    CURSOR_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/cursor_auth.json")" && pwd)/cursor_auth.json"
    mkdir -p "${JOB_DIR}/.config/cursor"
    : > "${JOB_DIR}/.config/cursor/auth.json"
fi

# xAI Grok Build CLI persists its OAuth session at ~/.grok/auth.json. Bind-mount
# the agent's copy so the CLI inside the sandbox reads and rotates against the
# shared credential file. Distinct filename in the agent dir (grok_auth.json)
# avoids collision with the codex auth.json check above.
GROK_AUTH_SRC=""
if [ -f "agents/${AGENT}/grok_auth.json" ]; then
    GROK_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/grok_auth.json")" && pwd)/grok_auth.json"
    mkdir -p "${JOB_DIR}/.grok"
    : > "${JOB_DIR}/.grok/auth.json"
fi

# Utils
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

with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)
    
    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"
    
    return $exit_code
}

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

solve_task() {
    AGENT_AUTH_BIND=()
    [ -n "$AGENT_AUTH_SRC" ] && AGENT_AUTH_BIND=(--bind "${AGENT_AUTH_SRC}:/home/ben/.codex/auth.json")
    CURSOR_AUTH_BIND=()
    [ -n "$CURSOR_AUTH_SRC" ] && CURSOR_AUTH_BIND=(--bind "${CURSOR_AUTH_SRC}:/home/ben/.config/cursor/auth.json")
    GROK_AUTH_BIND=()
    [ -n "$GROK_AUTH_SRC" ] && GROK_AUTH_BIND=(--bind "${GROK_AUTH_SRC}:/home/ben/.grok/auth.json")
    # Forward the CLI-auto-update opt-out into the sandbox so update_agent_cli.sh
    # can honor it. Only set when the user opts in via .env.
    CLI_UPDATE_ENV=()
    [ -n "${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-}" ] && CLI_UPDATE_ENV+=(--env "POST_TRAIN_BENCH_SKIP_CLI_UPDATE=${POST_TRAIN_BENCH_SKIP_CLI_UPDATE}")
    # check_cuda.py fails the run unless torch.cuda.device_count() == NUM_GPUS,
    # and --cleanenv drops the host's CUDA_VISIBLE_DEVICES. A scheduler that
    # hands out whole nodes therefore shows all 8 devices to a NUM_GPUS=1 run.
    # Raising NUM_GPUS instead would be wrong: it appends a _8gpu suffix to
    # EVAL_DIR and so renames the method that collect.py reads.
    #
    # The value is the host index only when nothing renumbers the devices.
    # nvidia-container-cli injects *only* the listed cards and renumbers them from
    # zero, so under POST_TRAIN_BENCH_ISOLATE_GPUS=1 the sandbox holds one card at
    # index 0 whichever card it is: POST_TRAIN_BENCH_VISIBLE_GPUS=3 with
    # CUDA_VISIBLE_DEVICES=3 selects nothing, torch reports no CUDA at all, and
    # check_cuda.py ends the run two minutes in with an empty final_model. GPU 0 is
    # the single index where the two numberings agree, and a scheduler that hands
    # out whole nodes always starts at 0 -- so every one-cell run on this cluster
    # passed and this stayed invisible until six cells shared a node and five of
    # them died the same way.
    VISIBLE_GPUS_ENV=()
    if [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ]; then
        AGENT_CUDA_VISIBLE="${POST_TRAIN_BENCH_VISIBLE_GPUS}"
        if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-}" = "1" ]; then
            AGENT_CUDA_VISIBLE=$(seq -s, 0 "$(( $(awk -F, '{print NF}' <<<"${POST_TRAIN_BENCH_VISIBLE_GPUS}") - 1 ))")
        fi
        VISIBLE_GPUS_ENV+=(--env "CUDA_VISIBLE_DEVICES=${AGENT_CUDA_VISIBLE}")
        echo "agent gpu: host [${POST_TRAIN_BENCH_VISIBLE_GPUS}] -> container CUDA_VISIBLE_DEVICES=[${AGENT_CUDA_VISIBLE}] (isolate=${POST_TRAIN_BENCH_ISOLATE_GPUS:-0})"
    fi
    # CUDA_VISIBLE_DEVICES is an environment variable, not a device cgroup. It
    # satisfies check_cuda.py and it is what torch reads, but --nv binds every
    # /dev/nvidia* on the node, so the agent's own Bash still sees eight cards
    # in nvidia-smi and one `export` away from using them. HTCondor handed the
    # published runs a one-GPU cgroup; on an OverSubscribe=EXCLUSIVE partition
    # nothing does. nvidia-container-cli binds only the listed devices, which
    # restores the one-H100 contract the task prompt states. Off by default:
    # it needs nvidia-container-cli on the host, and a cluster whose scheduler
    # already isolates GPUs does not want a second mechanism doing it.
    #
    # NVIDIA_VISIBLE_DEVICES is read by apptainer itself out of the host
    # environment, so it is exported rather than passed with --env. --nvccli
    # also requires --writable-tmpfs AND -c, both of which this exec already
    # passes -- the second is load-bearing and does not look it: stock
    # apptainer.conf has `mount dev = yes`, which bind-mounts the host /dev back
    # over the device list nvidia-container-cli just built. Measured on an
    # 8-H100 node with NVIDIA_VISIBLE_DEVICES=0: --nvccli alone leaves all eight
    # /dev/nvidia* in the sandbox, -c --nvccli leaves one. Both exit 0 and
    # neither warns, so dropping -c here would silently un-isolate the run
    # while every other symptom stayed identical.
    NVCCLI_ARGS=()
    if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-}" = "1" ] && [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ]; then
        export NVIDIA_VISIBLE_DEVICES="${POST_TRAIN_BENCH_VISIBLE_GPUS}"
        NVCCLI_ARGS=(--nvccli)
    fi
    # The three facts a payload agent's entry point needs and cannot otherwise
    # get: --cleanenv drops them, and the only copies inside the sandbox are
    # prose inside $PROMPT and a countdown in timer.sh. Same guard as the copy
    # above, so an agent without a payload sees exactly the environment it saw
    # before this block existed.
    AGENT_CONTEXT_ENV=()
    [ -d "${JOB_DIR}/agent" ] && AGENT_CONTEXT_ENV=(
        --env "BENCHMARK_ID=${EVALUATION_TASK}"
        --env "MODEL_TO_TRAIN=${MODEL_TO_TRAIN}"
        --env "NUM_HOURS=${NUM_HOURS}"
    )
    # SOLVE_EXIT below is meant to say whether the agent worked. It did not: the
    # brace group ended with `kill $MONITOR_PID`, so its status was the kill's, and
    # the group was piped into timestamp_lines.py, so the pipeline's status was
    # python's. Both are 0 essentially always. A cell whose check_cuda.py refused to
    # start the agent at all therefore recorded `exit_code: 0 / status: exited
    # normally` beside `final_model_files: 0`, and the only field that disagreed was
    # the one nobody reads first. pipefail plus an explicit exit makes the number
    # mean what its label says; nothing downstream branches on it, so an honest
    # nonzero costs nothing and a dishonest zero cost five cells.
    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    apptainer exec \
        --nv \
        "${NVCCLI_ARGS[@]}" \
        -c \
        --cleanenv \
        --pid \
        --no-init \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env HF_HOME="${HF_HOME_NEW}" \
        "${API_KEY_ENV_ARGS[@]}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --env NUM_GPUS="${NUM_GPUS}" \
        --env PROMPT="${PROMPT}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        "${CLI_UPDATE_ENV[@]}" \
        "${VISIBLE_GPUS_ENV[@]}" \
        "${AGENT_CONTEXT_ENV[@]}" \
        "${AGENT_ENV_ARGS[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_MERGED}:${HF_HOME_NEW}" \
        "${AGENT_AUTH_BIND[@]}" \
        "${CURSOR_AUTH_BIND[@]}" \
        "${GROK_AUTH_BIND[@]}" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
        bash -c "set -o pipefail; { python /home/ben/check_cuda.py && python /home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & MONITOR_PID=\$!; bash /home/ben/agent_solve.sh; SOLVE_RC=\$?; kill \$MONITOR_PID 2>/dev/null; exit \$SOLVE_RC; } 2>&1 | python /home/ben/timestamp_lines.py" > "${SOLVE_OUT}" 2>&1
}

# ---------- judge OAuth precheck ----------
# All judges run via the codex CLI with the subscription auth at
# agents/codex_non_api/auth.json (see src/judges/judge_lib.sh). If its ChatGPT
# session is invalidated, we'd waste the full agent run only to hard-error at
# the judge phase. One curl to a lightweight ChatGPT endpoint tells us the
# state: it uses the already-issued access token, no refresh path, so it
# doesn't rotate anything or race parallel job starts.

echo "================================"
echo "======= JUDGE AUTH CHECK ======="
echo "================================"
JUDGE_AUTH="agents/codex_non_api/auth.json"
if [ ! -f "$JUDGE_AUTH" ]; then
    echo "ERROR: judge auth file missing at $JUDGE_AUTH" >&2
    exit 1
fi
JUDGE_ACCESS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"]["access_token"])' "$JUDGE_AUTH") \
    || { echo "ERROR: could not extract tokens.access_token from $JUDGE_AUTH" >&2; exit 1; }
JUDGE_HTTP=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 \
    -H "Authorization: Bearer $JUDGE_ACCESS" \
    "https://chatgpt.com/backend-api/codex/models?client_version=0.124.0")
if [ "$JUDGE_HTTP" != "200" ]; then
    echo "ERROR: judge OAuth precheck failed (HTTP ${JUDGE_HTTP})." >&2
    echo "The ChatGPT session in $JUDGE_AUTH may be invalidated." >&2
    echo "Re-login on the head node:" >&2
    echo "  codex logout && codex login && cp ~/.codex/auth.json $JUDGE_AUTH && chmod 600 $JUDGE_AUTH" >&2
    exit 1
fi
echo "Judge OAuth OK (HTTP 200)"

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

with_huggingface_overlay with_record_the_time solve_task
SOLVE_EXIT=$?

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
elif [ $SOLVE_EXIT -gt 128 ]; then
    echo "status: killed by signal $((SOLVE_EXIT - 128)) ($(kill -l $((SOLVE_EXIT - 128)) 2>/dev/null || echo unknown))"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "final_model_files: $(ls "${JOB_DIR}/task/final_model/" 2>/dev/null | wc -l)"
echo "hostname: $(hostname)"
echo "fuse_overlayfs_alive: $(ps aux 2>/dev/null | grep fuse-overlay | grep -v grep | wc -l)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "disk_tmp: $(du -sh "${JOB_TMP}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

# Record the (auto-updated) agent CLI version captured by update_agent_cli.sh.
if [ -f "${JOB_DIR}/cli_version.txt" ]; then
    cp "${JOB_DIR}/cli_version.txt" "${EVAL_DIR}/cli_version.txt"
    echo "--- AGENT CLI VERSION ---"
    cat "${EVAL_DIR}/cli_version.txt"
    echo "--- END AGENT CLI VERSION ---"
else
    echo "WARNING: ${JOB_DIR}/cli_version.txt not found (agent CLI version not recorded)" >&2
fi

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# Parse agent trace into human-readable format
#
# --raw-only for an agent that ships a payload/: parse_trace.py dispatches on a
# substring of the agent name, so "claude_autor" selects the claude parser, which
# reads the Claude CLI's stream-json. A payload agent writes its own log format
# and gets a stub plus one "NOT PARSABLE" line per input line on stderr -- into
# error.log, the file that is supposed to hold this harness's own errors. Raw-only
# takes the path upstream already has for an unrecognised agent (copy the trace
# verbatim, still sanitize it). Keyed on the payload directory, not on a name, so
# it stays true for the next agent that is a repository.
PARSE_TRACE_ARGS=()
[ -d "agents/${AGENT}/payload" ] && PARSE_TRACE_ARGS=(--raw-only)
python src/trace_parsing/parse_trace.py --agent "${AGENT}" "${PARSE_TRACE_ARGS[@]}" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
tree ${JOB_DIR}/task
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
fi

if [ -f "${JOB_DIR}/task/system_monitor.log" ]; then
    cp "${JOB_DIR}/task/system_monitor.log" "$EVAL_DIR/system_monitor.log"
fi

python containers/delete_hf_models.py "${JOB_DIR}/task"

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"

rm -rf /tmp/posttrain_container

echo "========================================="
echo "=== RUNNING REWARD-HACKING JUDGES ==="
echo "========================================="

source "${_RUN_TASK_SRC}/judges/judge_lib.sh"

# Make judge helper tooling and benchmark metadata available inside the judge
# sandbox. The final_model config comes from EVAL_DIR because delete_hf_models
# has already run on JOB_DIR/task during cleanup.
prepare_judge_sandbox "${JOB_DIR}" "${EVALUATION_TASK}" "${EVAL_DIR}/final_model/config.json"

# Reset codex config (so agent-specific settings like model_reasoning_effort
# can't leak into the judges) and set up the bind-mounted subscription auth
# (JUDGE_CODEX_AUTH_SRC), through which rotated refresh tokens persist back to
# the source instead of dying with the sandbox.
setup_judge_codex_auth "${JOB_DIR}" || exit 1

JUDGE_EXTRA_APPTAINER_ARGS=(
    --nv
    --env HF_HOME="${HF_HOME_NEW}"
    --env VLLM_API_KEY="inspectai"
    --bind "${HF_MERGED}:${HF_HOME_NEW}"
)

FIRST_JUDGE=1
for JUDGE_NAME in "${ALL_JUDGES[@]}"; do
    load_judge_conf "${JUDGE_NAME}" || exit 1

    echo "=== Judge: ${JUDGE_LABEL} ==="

    # Clean judgement file between judges so each one starts fresh
    [ "$FIRST_JUDGE" = "1" ] || rm -f "${JOB_DIR}/task/judgement.json"
    FIRST_JUDGE=0

    JUDGE_PROMPT=$(build_judge_prompt "${JUDGE_NAME}" "${EVALUATION_TASK}" "${MODEL_TO_TRAIN}" "${AGENT}" "${AGENT_CONFIG}")

    with_huggingface_overlay run_judge_exec "${JOB_DIR}" "${JOB_TMP}" "${EVAL_DIR}/judge_output_${JUDGE_OUTPUT_ID}.json" "${JUDGE_PROMPT}"

    # missing_fatal=0: a judge that produces no verdict warns and moves on. The
    # agent's 10h of work is already done, so it must still be evaluated; the
    # rerun pipeline can supply the missing verdict afterwards.
    collect_judge_output "${JOB_DIR}" "${EVAL_DIR}" "" 0
done

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export REPO_ROOT="$(pwd)"

export TMP_HF_CACHE="/tmp/hf_cache_90afd0"

# The scorer's container does not run --cleanenv, so the host's cache variables
# reach inside it and outrank the --env below. huggingface_hub ranks
# HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > "$HF_HOME/hub", and this cluster exports
# the middle one globally; XDG_CACHE_HOME is exported too, and it is where
# anything without a variable of its own lands -- vLLM's VLLM_CACHE_ROOT defaults
# to "$XDG_CACHE_HOME/vllm", torch inductor's triton kernels go there as well.
# None of that root is bound in, so each write is created on the container root,
# which --writable-tmpfs caps at `sessiondir max size` (64 MiB here).
#
# Left unfixed this costs a whole job and looks like nothing: the agent finishes,
# final_model is written, and then all nine evaluation attempts die at vLLM
# startup with
#
#     torch._inductor.exc.InductorError: OSError: [Errno 28] No space left
#
# on a node with terabytes free. Job 81521 spent an hour of opus and produced
# final_eval_1..9.txt and no metrics.json. Naming the three HF variables keeps the
# scorer reading the per-invocation overlay rather than the shared hub; binding
# the cache root covers everything else, including the next library to invent a
# variable, and keeps compiled triton kernels between the nine attempts.
#
# The agent sandbox needs neither: it launches with -c --cleanenv, so none of
# these ever reached it, which is why only the scorer failed.
#
# Both lists are built inside run_evaluation rather than out here, for the reason
# the comment above run_evaluation_with_retry's `declare -f` already gives once:
# that line starts a fresh `bash -c` which receives exported variables and the
# named function bodies, and nothing else. Bash cannot export an array. Defined at
# this level they would arrive empty, "${CACHE_BIND[@]}" would expand to nothing,
# and the exec would silently go back to the form that fails -- the same shape of
# bug, in the same file, one scope over. Inside the function `declare -f` carries
# them and they cannot drift out of the list.

export EVAL_COUNTER=0

# Free the GPUs before vLLM starts. Upstream kills every compute process on
# every visible device, unfiltered by owner or by device -- correct under an
# HTCondor whole-node claim, where nothing else can be running. Under a
# scheduler where processes can reach the node outside the allocation (an
# interactive ssh session, say), that same command reaches other people's work,
# and run_evaluation is called up to nine times per job. The default is
# upstream's behaviour; "own" restricts the sweep to this user's processes and
# "none" disables it.
#
# Owner is not a fine enough filter here. On an EXCLUSIVE whole-node partition the only
# economical shape is several cells sharing one node, one GPU each, all of them the same
# POSIX user -- and several humans share this account besides. "own" then sweeps every
# H100 on the box and kills the sibling cells' vLLM servers, hours in, leaving nothing in
# either log but a scorer that restarted. So the query is scoped to the device this cell
# was given: `nvidia-smi -i "$VISIBLE"` takes the same index CUDA_VISIBLE_DEVICES does and
# exits 0. When the variable is unset the behaviour is exactly what it was -- all devices
# -- because a job that did not say which GPU is its own has not claimed one.
reap_gpu_processes() {
    local mode="${POST_TRAIN_BENCH_EVAL_GPU_REAP:-all}"
    local pids
    local device_arg=()
    [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ] && \
        device_arg=(-i "${POST_TRAIN_BENCH_VISIBLE_GPUS}")
    case "$mode" in
        none)
            echo "reap_gpu_processes: disabled (POST_TRAIN_BENCH_EVAL_GPU_REAP=none)"
            return 0
            ;;
        own)
            pids=$(nvidia-smi "${device_arg[@]}" --query-compute-apps=pid --format=csv,noheader \
                   | tr -d ' ' \
                   | while read -r p; do
                         [ -n "$p" ] || continue
                         [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ] && echo "$p"
                     done)
            echo "reap_gpu_processes: own-user on gpu [${POST_TRAIN_BENCH_VISIBLE_GPUS:-all}], killing [${pids//$'\n'/ }]"
            ;;
        *)
            pids=$(nvidia-smi "${device_arg[@]}" --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
            echo "reap_gpu_processes: all users on gpu [${POST_TRAIN_BENCH_VISIBLE_GPUS:-all}], killing [${pids//$'\n'/ }]"
            ;;
    esac
    [ -n "$pids" ] && echo "$pids" | xargs -r kill -9
    return 0
}

run_evaluation() {
    # EVAL_DIR has to be bound. Both paths this function hands the scorer --
    # --model-path "$EVAL_DIR/final_model" and --json-output-file
    # "$EVAL_DIR/metrics.json" -- are under POST_TRAIN_BENCH_RESULTS_DIR, and
    # only REPO_ROOT and the HF cache are bound here, so with a results dir
    # outside the checkout neither exists inside the container and evaluate.py
    # cannot load the model it was asked to score. Four attempts, then two more,
    # and the run records no metrics.json while every other artifact looks
    # healthy -- including final_eval_N.txt, because that redirect is the host
    # shell's and lands on the host regardless. Upstream never meets this:
    # example.env's results dir is the relative "results", which lands inside the
    # REPO_ROOT bind. src/baselines/run_baseline.sh already binds its own
    # RESULT_DIR -- this is the same bind, in the path that scores an agent
    # rather than the base model.
    local max_tokens_arg="$1"
    local eval_num="$2"
    reap_gpu_processes
    sleep 5
    local hf_cache_env=(
        --env "HF_HOME=${TMP_HF_CACHE}"
        --env "HF_HUB_CACHE=${TMP_HF_CACHE}/hub"
        --env "HUGGINGFACE_HUB_CACHE=${TMP_HF_CACHE}/hub"
    )
    local cache_bind=()
    [ -n "${XDG_CACHE_HOME:-}" ] && [ -d "${XDG_CACHE_HOME}" ] && \
        cache_bind=(--bind "${XDG_CACHE_HOME}:${XDG_CACHE_HOME}")
    local visible_gpus_env=()
    [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ] && \
        visible_gpus_env+=(--env "CUDA_VISIBLE_DEVICES=${POST_TRAIN_BENCH_VISIBLE_GPUS}")
    with_huggingface_overlay apptainer exec \
        --nv \
        "${visible_gpus_env[@]}" \
        "${hf_cache_env[@]}" \
        "${cache_bind[@]}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "${EVAL_DIR}:${EVAL_DIR}" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python "${EVAL_SCRIPT}" \
            --model-path "$EVAL_DIR/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${max_tokens_arg} \
            --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/final_eval_${eval_num}.txt"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        export EVAL_COUNTER
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        # reap_gpu_processes has to be in this list. run_evaluation calls it, and
        # this subshell gets only the functions named here -- upstream had the kill
        # inline, so `declare -f run_evaluation` carried it and this list did not
        # have to know about it.
        timeout --signal=TERM --kill-after=60s 28800s bash -c "$(declare -f run_evaluation with_huggingface_overlay reap_gpu_processes); run_evaluation \"$max_tokens_arg\" \"$EVAL_COUNTER\""

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# First evaluation: up to 4 attempts
run_evaluation_with_retry 4 ""

# Second evaluation with adjusted max tokens: up to 2 attempts
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

run_evaluation_with_retry 3 "$MAX_TOKENS_ARG"

# Third evaluation with further adjusted max tokens: up to 2 attempts
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

run_evaluation_with_retry 2 "$MAX_TOKENS_ARG"

echo $(cat "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt")

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

# Six evaluation attempts can all fail and this script still ends on an echo, so
# it exits 0 and the scheduler records COMPLETED over an empty result. Upstream
# runs under HTCondor with a human reading the directory afterwards; a Slurm
# queue is read by looking at the state column, and "COMPLETED, no score" is the
# one outcome that must not look like the good one. Neither retry ladder's return
# value is checked above -- deliberately, because the first ladder failing and the
# second succeeding is a normal run -- so the check is on the artifact, not on a
# status: metrics.json is the whole deliverable of this script.
if [ ! -f "${EVAL_DIR}/metrics.json" ]; then
    echo "FATAL: every evaluation attempt failed; ${EVAL_DIR}/metrics.json was never written" >&2
    exit 1
fi

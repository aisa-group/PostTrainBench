#!/bin/bash
# Verifier — a port of everything src/run_task.sh does after the agent exits
# (lines 290-518), plus the scoring rules scripts/collect.py applies later.
#
# Upstream splits this work across a host script and two apptainer images;
# Harbor gives it one container that runs after the agent phase. The section
# banners below are upstream's, and each block cites the run_task.sh lines it
# came from. Read them side by side.
#
# Two path substitutions, applied consistently:
#
#   run_task.sh          here            what it is
#   -------------------  --------------  ---------------------------------
#   $JOB_DIR             /home/ben       the agent sandbox home. Upstream it
#                                        is a host dir mounted into the
#                                        sandbox; here Harbor transfers it in
#                                        as the task's [[artifacts]] entry,
#                                        re-materialised at the same path.
#   $EVAL_DIR            /logs/verifier  the result dir. Harbor bind-mounts
#                                        it from the host.
#   $(pwd) / REPO_ROOT   /tests/repo     the repo. Upstream bind-mounts the
#                                        real checkout; build_tasks.py stages
#                                        the subtree these phases read, at
#                                        the same relative paths, so every
#                                        upstream script and every relative
#                                        path in them works unmodified.
#
# Deliberate differences are marked `DIVERGENCE:` and explained inline.

set -u

JOB_DIR="/home/ben"
EVAL_DIR="/logs/verifier"
REPO_ROOT="/tests/repo"
TESTS="/tests"

mkdir -p "$EVAL_DIR"

cd "$REPO_ROOT" || exit 1

# --- task identity ----------------------------------------------------------
# Read from /tests, never from the workspace: the agent can write to its own
# home, so anything that steers the verifier has to come from the baked-in
# image.
EVALUATION_TASK=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])")
MODEL_TO_TRAIN=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json'))['model_to_train'])")
AGENT=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json'))['agent'])")
AGENT_CONFIG=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json'))['agent_config'])")
# Prefer the model the platform actually scheduled. metadata.json's
# agent_config is baked at build time, and one task serves every candidate
# (D228), so it is correct only for whichever candidate it was built for.
# The judges' harness-identity clause is built from this value.
if [ -n "${KAGGLE_AGENT_LLM:-}" ]; then
    [ "${KAGGLE_AGENT_LLM}" != "${AGENT_CONFIG}" ] && \
        echo "agent_config: metadata says ${AGENT_CONFIG}, platform scheduled ${KAGGLE_AGENT_LLM} — using the latter"
    AGENT_CONFIG="${KAGGLE_AGENT_LLM}"
fi
export EVALUATION_TASK

echo "benchmark_id=${EVALUATION_TASK} model_to_train=${MODEL_TO_TRAIN}"
echo "agent=${AGENT} agent_config=${AGENT_CONFIG}"

# --- judge backend selection ------------------------------------------------
# run_task.sh:13-26, verbatim. Default to the OpenAI-backed evaluate.py; fall
# back to the OpenRouter variant when only OPENROUTER_API_KEY is available.
JUDGE_BACKEND="openai"
if { [ "$EVALUATION_TASK" = "arenahardwriting" ] || [ "$EVALUATION_TASK" = "healthbench" ]; } \
   && [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
    JUDGE_BACKEND="openrouter"
fi

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    export EVAL_SCRIPT="evaluate_openrouter.py"
else
    export EVAL_SCRIPT="evaluate.py"
fi
echo "Judge backend: ${JUDGE_BACKEND} (eval script: ${EVAL_SCRIPT})"


echo "--- SOLVE DIAGNOSTICS ---"
# run_task.sh:290-307. The exit-code branch is gone: harbor owns the agent
# process and its status is in the trial result, not reachable from here.
# Everything else is the same postmortem snapshot.
echo "final_model_files: $(ls "${JOB_DIR}/task/final_model/" 2>/dev/null | wc -l)"
echo "hostname: $(hostname)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
# run_task.sh:303 reports fuse_overlayfs_alive; there is no overlay here (the
# HF cache is a read-only mount, see the task README), so it is omitted.
echo "--- END SOLVE DIAGNOSTICS ---"

# run_task.sh:310-317. update_agent_cli.sh does not run in this port — harbor
# installs and pins the agent CLI itself via `--ak version=` — so
# cli_version.txt is absent by construction rather than by failure. The
# version harbor applied is recorded in the trial's config.json.
if [ -f "${JOB_DIR}/cli_version.txt" ]; then
    cp "${JOB_DIR}/cli_version.txt" "${EVAL_DIR}/cli_version.txt"
    echo "--- AGENT CLI VERSION ---"
    cat "${EVAL_DIR}/cli_version.txt"
    echo "--- END AGENT CLI VERSION ---"
fi

# --- time_taken.txt ---------------------------------------------------------
# run_task.sh:197-210, with_record_the_time, wraps solve_task and writes
# HH:MM:SS to $EVAL_DIR/time_taken.txt. Harbor owns the agent process, so the
# duration is measured the only way available from here: create_timer.sh
# stamps CREATION_DATE into the generated timer.sh at the moment preflight
# runs, i.e. immediately before the agent starts, so now-minus-CREATION_DATE
# is the agent phase plus harbor's teardown.
#
# This is not cosmetic. collect.py calls load_time_taken inside the same
# try/except as the metrics load (utils.py:627), so a run with no
# time_taken.txt is treated as broken and scored as the baseline.
if [ -f "${JOB_DIR}/task/timer.sh" ]; then
    CREATION_DATE=$(grep -oP '^CREATION_DATE=\K[0-9]+' "${JOB_DIR}/task/timer.sh" | head -1)
    if [ -n "${CREATION_DATE:-}" ]; then
        TIME_TAKEN=$(( $(date +%s) - CREATION_DATE ))
        printf '%02d:%02d:%02d\n' \
            $(( TIME_TAKEN / 3600 )) \
            $(( (TIME_TAKEN % 3600) / 60 )) \
            $(( TIME_TAKEN % 60 )) > "${EVAL_DIR}/time_taken.txt"
        echo "time_taken: $(cat "${EVAL_DIR}/time_taken.txt")"
    fi
fi
if [ ! -f "${EVAL_DIR}/time_taken.txt" ]; then
    echo "WARNING: could not derive time_taken.txt (no CREATION_DATE in ${JOB_DIR}/task/timer.sh)" >&2
fi

# --- the raw agent trace ------------------------------------------------------
# run_task.sh:212.
SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

# run_task.sh:250 fills it by redirecting the whole sandbox session
# (stdout+stderr, piped through timestamp_lines.py) into $SOLVE_OUT. Harbor
# owns the agent process instead and tees the same raw stream to
# /logs/agent/<file>, one name per adapter:
#
#   claude-code.txt   harbor/agents/installed/claude_code.py:1782-1783
#   codex.txt         harbor/agents/installed/codex.py:52 + :1448-1449
#   cursor-cli.txt    harbor/agents/installed/cursor_cli.py:209 + :883
#   gemini-cli.txt    harbor/agents/installed/gemini_cli.py:854
#   opencode.txt      harbor/agents/installed/opencode.py:519
#
# Four of those five are the structured stream upstream's solve.sh captures:
# claude_code.py:1778 passes --output-format=stream-json, codex.py:1443
# --json, cursor_cli.py:882 --output-format=stream-json, opencode.py:516
# --format=json. gemini is the exception — gemini_cli.py:852-854 passes no
# output-format flag at all, unlike agents/gemini/solve.sh:7's
# `--output-format stream-json`, so its tee is human-readable chatter (three
# lines in a real trial) and not a trace. The equivalent structured record is
# the CLI's own session log, which harbor copies to
# /logs/agent/gemini-cli.trajectory.jsonl at gemini_cli.py:862-871, so that is
# preferred for gemini with the tee as fallback.
#
# task.toml declares /logs/agent as an artifact, so the directory is here.
# The name is selected from ${AGENT} by the same substring test
# parse_trace.py:37 uses -- but see the discovery fallback below: in this port
# ${AGENT} is a placeholder, so the two DID disagree and the mapping missed.
#
# DIVERGENCE: harbor does not run timestamp_lines.py, so the lines carry no
# `[2026-...Z] ` prefix. Every parser treats it as optional — claude_parser.py
# :206-209, codex_parser.py:29-31 and :513-516, gemini_parser.py:336-338,
# opencode_parser.py:215-217, and cursor_parser.py via
# claude_parser.load_events — so the trace still parses; the judges lose
# wall-clock times.
#
# DIVERGENCE: upstream's stream also carries check_cuda and system_monitor.sh
# output (run_task.sh:250 runs all three in one pipeline). Here those run in
# preflight.sh, before harbor starts the agent, so this trace holds the agent
# alone. system_monitor.log is still copied out below.
case "${AGENT}" in
    *claude*)   HARBOR_TRACE="claude-code.txt" ;;
    *codex*)    HARBOR_TRACE="codex.txt" ;;
    *cursor*)   HARBOR_TRACE="cursor-cli.txt" ;;
    *gemini*)   HARBOR_TRACE="gemini-cli.trajectory.jsonl"
                [ -f "/logs/agent/${HARBOR_TRACE}" ] || HARBOR_TRACE="gemini-cli.txt" ;;
    *opencode*) HARBOR_TRACE="opencode.txt" ;;
    *)          HARBOR_TRACE="" ;;
esac
# ${AGENT} is metadata.json's, and for this port that is the PROMPT_AGENT
# placeholder ("ptb") rather than a harness name: one task is deliberately
# shared by every candidate (D228), so metadata cannot name the harness that
# actually ran. The case above therefore matches nothing and HARBOR_TRACE
# stays empty, which silently skipped the copy -- solve_out.txt and
# solve_parsed.txt were absent from all 20 runs of the first v5 cycle, and
# every judge fell back to reconstructing from secondary artifacts.
#
# Harbor's own output is the reliable signal: it writes exactly one trace,
# named for the adapter that ran. Probe the known names in a fixed order
# rather than globbing, so an unexpected file in /logs/agent cannot be
# mistaken for a trace.
if [ -z "${HARBOR_TRACE}" ] || [ ! -f "/logs/agent/${HARBOR_TRACE}" ]; then
    for _ptb_cand in claude-code.txt codex.txt opencode.txt \
                     gemini-cli.trajectory.jsonl gemini-cli.txt cursor-cli.txt; do
        if [ -f "/logs/agent/${_ptb_cand}" ]; then
            HARBOR_TRACE="${_ptb_cand}"
            echo "agent trace: resolved by discovery (metadata agent=${AGENT} names no harness)"
            break
        fi
    done
fi

# The harness the trace came from is the real upstream agent name, and it is
# what run_task.sh's ${AGENT} carries. parse_trace.py dispatches on it
# (select_parser returns None for an unknown name) and get_judge_prompt.py
# builds the "ignore the harness identity" clause from it -- with "ptb" the
# judges got the generic branch instead of the claude/codex/gemini-specific
# one, which is what tells them a `claude-haiku-*` helper call is the harness
# and not an agent-made API call.
case "${HARBOR_TRACE}" in
    claude-code*) AGENT_REAL="claude" ;;
    codex*)       AGENT_REAL="codex" ;;
    cursor-cli*)  AGENT_REAL="cursor" ;;
    gemini-cli*)  AGENT_REAL="gemini" ;;
    opencode*)    AGENT_REAL="opencode" ;;
    *)            AGENT_REAL="${AGENT}" ;;
esac

if [ -n "${HARBOR_TRACE}" ] && [ -f "/logs/agent/${HARBOR_TRACE}" ]; then
    cp "/logs/agent/${HARBOR_TRACE}" "${SOLVE_OUT}"
    echo "agent trace: /logs/agent/${HARBOR_TRACE} -> ${SOLVE_OUT} ($(wc -l < "${SOLVE_OUT}") lines)"
    echo "agent (real, from trace): ${AGENT_REAL}"
else
    echo "WARNING: no harbor agent trace found in /logs/agent (metadata agent=${AGENT})" >&2
fi

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# run_task.sh:323-325, verbatim. cwd is $REPO_ROOT (set at the top of this
# file), which is what puts src/trace_parsing on sys.path for the parsers'
# flat sibling imports — exactly as upstream's invocation does.
# Parse agent trace into human-readable format
python src/trace_parsing/parse_trace.py --agent "${AGENT_REAL}" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

# run_task.sh:331-333
echo "Task directory contents:"
if command -v tree > /dev/null 2>&1; then
    tree "${JOB_DIR}/task"
else
    find "${JOB_DIR}/task" -maxdepth 3 | sort
fi
echo "================================"

# run_task.sh:335-341. Order is load-bearing: final_model and the monitor log
# are copied out BEFORE delete_hf_models.py runs, which is why the judges
# below see a task/ dir with the weights already stripped.
if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
fi

if [ -f "${JOB_DIR}/task/system_monitor.log" ]; then
    cp "${JOB_DIR}/task/system_monitor.log" "$EVAL_DIR/system_monitor.log"
fi

# run_task.sh:343. Deletes every HuggingFace model folder under the agent's
# task dir — including final_model, which was just copied out. The judges run
# after this, so they inspect the agent's code with all weights gone; that is
# why prepare_judge_sandbox hands them final_model/config.json separately.
python "$REPO_ROOT/containers/delete_hf_models.py" "${JOB_DIR}/task"

# run_task.sh:345
cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"

echo "========================================="
echo "=== RUNNING REWARD-HACKING JUDGES ==="
echo "========================================="

# run_task.sh:253-281 runs a judge OAuth precheck against chatgpt.com before
# the agent even starts, because upstream's judges authenticate with a
# ChatGPT subscription (agents/codex_non_api/auth.json, bind-mounted so the
# rotated refresh token survives the job).
#
# DIVERGENCE: there is no subscription auth on Kaggle. Every LLM call goes
# through the model proxy, so the judges run codex in API mode against
# OPENAI_BASE_URL / OPENAI_API_KEY, which task.toml wires from the
# executor's proxy credentials. Consequences:
#   - setup_judge_codex_auth's auth.json bind mount has no analogue; the
#     .codex config reset it also does is kept below.
#   - the OAuth precheck is replaced by a reachability check on the proxy.
#   - upstream's judge model names are not proxy slugs, so judge_model_map.json
#     (written by build_tasks.py) remaps them. An identity map reproduces
#     upstream's strings exactly.

# judge_lib.sh:35-36
JUDGE_DEFAULT_MODEL="gpt-5.4"
JUDGE_DEFAULT_REASONING_EFFORT="xhigh"

# judge_lib.sh:31 — full set, in execution order.
ALL_JUDGES=(data_contamination_judge api_usage_judge ptb_lookup_judge general_judge)

# judge_lib.sh:42-66
load_judge_conf() {
    local judge_name="$1"
    local conf="$REPO_ROOT/src/judges/$judge_name/judge.conf"
    if [ ! -f "$conf" ]; then
        echo "ERROR: unknown judge '$judge_name' (no $conf)" >&2
        return 1
    fi
    JUDGE_LABEL=""
    JUDGE_OUTPUT_ID=""
    JUDGE_PROMPT_FILE=""
    JUDGE_MODEL="$JUDGE_DEFAULT_MODEL"
    JUDGE_REASONING_EFFORT="$JUDGE_DEFAULT_REASONING_EFFORT"
    JUDGE_CODEX_VERSION=""
    # shellcheck disable=SC1090
    source "$conf"
    if [ -z "$JUDGE_LABEL" ] || [ -z "$JUDGE_OUTPUT_ID" ] || [ -z "$JUDGE_PROMPT_FILE" ]; then
        echo "ERROR: $conf must set JUDGE_LABEL, JUDGE_OUTPUT_ID and JUDGE_PROMPT_FILE" >&2
        return 1
    fi
    # Proxy remap, applied after the conf so the conf stays byte-identical
    # to upstream.
    JUDGE_MODEL_RESOLVED=$(python3 -c "
import json, sys
m = json.load(open('$TESTS/judge_model_map.json'))
print(m.get(sys.argv[1], sys.argv[1]))
" "$JUDGE_MODEL")
}

# judge_lib.sh:68-92
prepare_judge_sandbox() {
    local job_dir="$1" benchmark_id="$2" final_model_config_src="$3"

    cp "$REPO_ROOT/src/judges/judge_tools/contamination_check.py" "$job_dir/contamination_check.py"
    cp "$REPO_ROOT/src/judges/judge_tools/model_identity_check.py" "$job_dir/model_identity_check.py"
    cp -r "$REPO_ROOT/src/judges/judge_tools/reference_configs" "$job_dir/reference_configs"

    if [ -f "$final_model_config_src" ]; then
        cp "$final_model_config_src" "$job_dir/final_model_config.json"
    fi

    if [ -f "$REPO_ROOT/src/eval/tasks/$benchmark_id/test_data.json" ]; then
        cp "$REPO_ROOT/src/eval/tasks/$benchmark_id/test_data.json" "$job_dir/test_data.json"
    fi
}

# judge_lib.sh:94-112, minus the auth.json handling (see DIVERGENCE above).
# The config reset is kept: it is what stops agent-specific codex settings
# such as model_reasoning_effort leaking into the judges.
setup_judge_codex_config() {
    local job_dir="$1"
    rm -rf "$job_dir/.codex"
    cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$job_dir/"
}

# judge_lib.sh:114-128
build_judge_prompt() {
    local judge_name="$1" benchmark_id="$2" model_hf="$3" agent="$4" agent_config="$5"
    local args=(--judge "$judge_name" --benchmark-id "$benchmark_id" --model "$model_hf")
    [ -n "$agent" ] && args+=(--agent "$agent")
    [ -n "$agent_config" ] && args+=(--agent-config "$agent_config")
    python "$REPO_ROOT/src/judges/get_judge_prompt.py" "${args[@]}"
}

# judge_lib.sh:130-179. The apptainer invocation becomes a direct call — this
# already is the judge's container. Flags with an analogue are kept:
#   --env CODEX_API_KEY / OPENAI_API_KEY  upstream blanks both to force
#                                         subscription auth; here they are
#                                         the proxy credentials and must stay.
#   --pwd /home/ben/task                  reproduced with cd.
#   --home /home/ben                      reproduced with HOME.
run_judge_exec() {
    local job_dir="$1" output_json="$2" prompt="$3"

    local codex_bin="codex"
    if [ -n "$JUDGE_CODEX_VERSION" ]; then
        local pin_prefix=".codex-cli-${JUDGE_CODEX_VERSION}"
        if [ ! -x "$job_dir/$pin_prefix/bin/codex" ]; then
            echo "  installing pinned codex CLI @openai/codex@${JUDGE_CODEX_VERSION} for ${JUDGE_LABEL} ..."
            npm install -g --prefix "${job_dir}/${pin_prefix}" --no-fund --no-audit \
                "@openai/codex@${JUDGE_CODEX_VERSION}"
        fi
        if [ ! -x "$job_dir/$pin_prefix/bin/codex" ]; then
            echo "ERROR: install of pinned @openai/codex@${JUDGE_CODEX_VERSION} failed (no ${pin_prefix}/bin/codex in the sandbox home) — ${JUDGE_LABEL} cannot run" >&2
            return 1
        fi
        codex_bin="${job_dir}/${pin_prefix}/bin/codex"
    fi

    # Flags upstream does not pass, because upstream does not need them: its
    # judges authenticate against the real OpenAI endpoint, so codex's default
    # provider is correct. Here they have to reach the model proxy, and the
    # codex CLI ignores OPENAI_BASE_URL — harbor's own adapter says so at
    # agents/installed/codex.py:1403 ("codex 0.118.0 only honors
    # openai_base_url from config.toml, not the env var").
    #
    # `-c openai_base_url=` was the first attempt and it is NOT enough. It
    # redirects the built-in `openai` provider, and that provider prefers a
    # WEBSOCKET transport: codex dials wss://$OPENAI_BASE_URL/responses. The
    # proxy registers that route only WITH a trailing slash, so:
    #
    #     wss://…/models/openapi/responses    -> 307 Temporary Redirect
    #     wss://…/models/openapi/responses/   -> 401 (route exists, wants auth)
    #
    # and codex's websocket client does not follow redirects. It burns five
    # reconnects and fails the turn. Measured in prod, verifier stdout of run
    # 865446, and reproduced locally against the same host with codex 0.142.5.
    # This is why every judge produced no verdict and run 768637 died with
    # RewardFileNotFoundError after a 7-minute verifier phase.
    #
    # A CUSTOM provider takes the plain-HTTPS path instead — confirmed in prod
    # by run 866264, whose custom-provider attempts have no websocket error at
    # all. The provider has to be a new id: codex rejects `model_providers.openai`
    # outright ("reserved built-in provider IDs … cannot be overridden"), which
    # is also why this cannot be expressed as a smaller edit. Four `-c` flags
    # replace one; `-c` is still the same mechanism upstream already uses for
    # model_reasoning_effort, so this adds flags rather than changing any.
    # codex 0.146.1 (D227) needs no provider surgery: it falls back to HTTPS
    # when the proxy 307s the websocket upgrade on `wss://$BASE/responses`.
    # 0.124.0, which gpt_5_5.def pins, has no such fallback -- five reconnects
    # then turn.failed -- which is why this used to inject a whole custom
    # `model_providers.kproxy.*` block and set model_provider=kproxy.
    #
    # The stock `openai` provider is now used, redirected with the TOP-LEVEL
    # `openai_base_url` key -- the same one harbor's adapter sets
    # (codex.py:1403) -- and NOT `model_providers.openai.base_url`.
    #
    # ⚠️ Those two are not interchangeable and the difference is fatal. Writing
    # under `model_providers.openai` tries to override a built-in provider id,
    # which codex refuses before it does anything else:
    #
    #     Error loading config.toml: model_providers contains reserved
    #     built-in provider IDs: `openai`. Built-in providers cannot be
    #     overridden. Rename your custom provider (for example,
    #     `openai-custom`).
    #
    # That is a config-load failure, so it kills the judge before a single
    # request is made. Measured against codex 0.142.5 locally, and it is what
    # made all four judges produce no verdict on every run of the 2026-09-04
    # sweep -- the paragraph above already said codex rejects
    # `model_providers.openai`, and the code below it did exactly that anyway.
    #
    # `-c openai_base_url=` was rejected earlier in this comment as "NOT
    # enough", which was true of 0.124.0: it takes the websocket path and the
    # proxy 307s the upgrade. 0.146.1 (D227) falls back to HTTPS, so the
    # objection no longer applies and the smaller edit is the right one.
    BASE_URL_ARG=()
    if [ -n "${JUDGE_BASE_URL:-}" ]; then
        BASE_URL_ARG=(-c "openai_base_url=\"${JUDGE_BASE_URL}\"")
    fi

    # NOTE on the second blocker, which is NOT fixed by anything in this
    # function. With the transport fixed, every judge call fails identically:
    #     403 Forbidden: Google Search Grounding is not enabled for your account
    # Raw HTTP from inside the verifier isolates it exactly (run 866861):
    #     POST /responses  no tools / function tool / tools:[]   -> 200
    #     POST /responses  tools:[{"type":"web_search"}]         -> 403
    # so ordinary function calling is fine and only the provider-NATIVE tool is
    # refused. codex sends web_search on every request and NO codex config
    # removes it — `--search`/no `--search`, tools.web_search=false,
    # tools.web_search_request=false, tools.enabled_tools=[] and
    # experimental_supported_tools=[] all yield the same 11-tool list (verified
    # against a recording HTTP server locally, and in prod runs 866646/866861).
    #
    # The entitlement is a token capability, and the token is minted per run by
    # the platform, not by us: TokenUtil.cs:151-154 sets
    # allow_model_native_tools from the feature flag
    # MODEL_PROXY_NATIVE_TOOL_ACCESS (4831), evaluated against the account that
    # launches the run. A token minted at admin.kaggle.com/admin/benchmarks
    # carries it, which is why --search verified fine from a workstation on
    # 2026-08-26 and 403s here. Granting that flag is the real fix; the shim
    # below the judge loop is only a diagnostic stand-in.
    (
        cd "$job_dir/task" || exit 1
        HOME="$job_dir" \
        PYTHONNOUSERSITE=1 \
        "$codex_bin" --search -a never exec --json \
            -c model_reasoning_summary=detailed \
            -c model_reasoning_effort="${JUDGE_REASONING_EFFORT}" \
            "${BASE_URL_ARG[@]}" \
            --skip-git-repo-check --yolo --model "${JUDGE_MODEL_RESOLVED}" "$prompt"
    ) 2>&1 | tee "$output_json"
}

# judge_lib.sh:181-197. missing_fatal is 0 here for the same reason
# run_task.sh passes 0: a judge that produces no verdict must never cost a
# finished agent run its evaluation. Unlike upstream there is no rerun
# pipeline to supply the verdict later, so the consequence surfaces at
# scoring time instead — see the reward section.
collect_judge_output() {
    local job_dir="$1" out_dir="$2"
    local out_base="judge_output_${JUDGE_OUTPUT_ID}"
    local judgement="$out_dir/judgement_${JUDGE_OUTPUT_ID}.json"

    # judge_lib.sh:186 — render the judge's own raw codex stream into a
    # readable transcript. Nothing downstream reads it (collect.py ignores it
    # entirely), but it is a documented result-dir artefact (AGENTS.md,
    # "Results Structure": judge_output_gpt5_4.{json,txt}) and it is what both
    # the WARNING below and the rerun pipeline point a human at. Unguarded,
    # exactly as upstream: this function has no `set -e` above it and a failed
    # parse must not stop the judge loop.
    python "$REPO_ROOT/src/trace_parsing/parse_trace.py" --agent codex \
        "$out_dir/${out_base}.json" -o "$out_dir/${out_base}.txt"

    if [ -f "$job_dir/task/judgement.json" ]; then
        cp "$job_dir/task/judgement.json" "$judgement"
        echo "  ${JUDGE_LABEL} judgement: $(cat "$judgement")"
    else
        echo "WARNING: judgement.json not created by ${JUDGE_LABEL} (see $out_dir/${out_base}.txt); continuing — a missing inline verdict never aborts the task run" >&2
    fi
}

echo "================================"
echo "======= JUDGE AUTH CHECK ======="
echo "================================"
# Stands in for run_task.sh:261-281. Same purpose — fail fast on broken judge
# credentials rather than after the fact — against the proxy instead of
# chatgpt.com.
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY is empty; the judges cannot authenticate." >&2
else
    JUDGE_HTTP=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 \
        -H "Authorization: Bearer ${OPENAI_API_KEY}" \
        "${OPENAI_BASE_URL:-https://api.openai.com/v1}/models" 2>/dev/null)
    echo "Judge endpoint check: HTTP ${JUDGE_HTTP:-000}"
fi

# No trace guard here, deliberately. src/judges/run_judges.sh:82-92 has one,
# but run_judges.sh is the standalone re-run tool (src/judges/rerun/) — the
# sweep path is run_task.sh, which sources judge_lib.sh and runs the judge
# loop inline at :374-391 with no guard of any kind, and no `set -e`. With no
# trace, upstream's parse_trace.py fails nonfatally, the judges find neither
# `../solve_parsed.txt` nor `../solve_out.txt`, and collect_judge_output runs
# with missing_fatal=0 (run_task.sh:390), which warns and returns 0
# (judge_lib.sh:188-196). The run still evaluates and still scores.

# DIAGNOSTIC ESCAPE HATCH, part 2 of 2 — part 1 is the PTB_DISABLE_WEB_SEARCH
# shim (D137b), which this pairs with; both are off by default. (An earlier
# revision called part 1 "the NO_SEARCH_ARG block"; no such block has existed
# since the shim replaced it. Reference corrected 2026-09-04.) Started once for
# all four judges; killed on exit. JUDGE_BASE_URL is what the codex provider
# points at.
JUDGE_BASE_URL="${OPENAI_BASE_URL:-}"
if [ "${PTB_DISABLE_WEB_SEARCH:-0}" = "1" ] && [ -n "${OPENAI_BASE_URL:-}" ]; then
    PTB_SHIM_UPSTREAM="${OPENAI_BASE_URL}" PTB_SHIM_PORT=8931 \
        python3 "${TESTS}/strip_web_search.py" &
    PTB_SHIM_PID=$!
    trap 'kill "${PTB_SHIM_PID}" 2>/dev/null' EXIT
    PTB_SHIM_UP=0
    for _ in $(seq 1 40); do
        if curl -s -o /dev/null -m 2 "http://127.0.0.1:8931/models"; then
            PTB_SHIM_UP=1; break
        fi
        sleep 0.25
    done
    if [ "$PTB_SHIM_UP" = "1" ]; then
        JUDGE_BASE_URL="http://127.0.0.1:8931"
        echo "  web-search strip shim up on ${JUDGE_BASE_URL}"
    else
        # Fall through to the direct URL rather than silently judging against
        # a dead port: the 403 is at least a legible failure.
        echo "  WARNING: strip shim did not come up; judges go direct." >&2
    fi
fi

# run_task.sh:353-364
prepare_judge_sandbox "${JOB_DIR}" "${EVALUATION_TASK}" "${EVAL_DIR}/final_model/config.json"
setup_judge_codex_config "${JOB_DIR}"

# run_task.sh:373-391
FIRST_JUDGE=1
for JUDGE_NAME in "${ALL_JUDGES[@]}"; do
    load_judge_conf "${JUDGE_NAME}" || exit 1

    echo "=== Judge: ${JUDGE_LABEL} ==="

    # Clean judgement file between judges so each one starts fresh
    [ "$FIRST_JUDGE" = "1" ] || rm -f "${JOB_DIR}/task/judgement.json"
    FIRST_JUDGE=0

    JUDGE_PROMPT=$(build_judge_prompt "${JUDGE_NAME}" "${EVALUATION_TASK}" "${MODEL_TO_TRAIN}" "${AGENT_REAL}" "${AGENT_CONFIG}")

    run_judge_exec "${JOB_DIR}" "${EVAL_DIR}/judge_output_${JUDGE_OUTPUT_ID}.json" "${JUDGE_PROMPT}"

    collect_judge_output "${JOB_DIR}" "${EVAL_DIR}"
done

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

# run_task.sh:397-425. The apptainer exec becomes a direct call — this is the
# evaluation container. Every argument to evaluate.py is unchanged, including
# the `--templates-dir ../../../../src/eval/templates` relative path, which
# resolves identically because /tests/repo mirrors the upstream tree.
export EVAL_COUNTER=0

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    # run_task.sh:406 kills every process holding the GPU. Guarded against
    # PID 1 here: upstream's apptainer sandbox has no init to kill, but in a
    # container a reparented vLLM child can show up as PID 1 and killing it
    # would take the container down with it.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | while read -r pid; do
            [ -n "$pid" ] || continue
            if [ "$pid" -gt 1 ] 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
        done
    sleep 5
    (
        cd "$REPO_ROOT/src/eval/tasks/${EVALUATION_TASK}" || exit 1
        HF_HOME="${TMP_HF_CACHE}" \
        VLLM_API_KEY="inspectai" \
        PYTHONNOUSERSITE=1 \
        PTB_JUDGE_MODEL="openai/gpt-5-mini" \
        OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
        OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
        python "${EVAL_SCRIPT}" \
            --model-path "$EVAL_DIR/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${max_tokens_arg} \
            --json-output-file "${EVAL_DIR}/metrics.json"
    ) > "$EVAL_DIR/final_eval_${eval_num}.txt" 2>&1
}

# run_task.sh:399 — upstream's scratch HF_HOME, kept at its literal path.
export TMP_HF_CACHE="${PTB_HF_HOME:-/tmp/hf_cache_90afd0}"

# run_task.sh:410-411 binds the fuse-overlayfs *merged* tree here, so upstream's
# evaluation gets a HF_HOME that reads the whole cache and is still WRITABLE —
# huggingface_hub takes a lock under $HF_HOME/hub/.locks on every download, and
# inspect_evals resolves its dataset under the same root.
#
# Pointing HF_HOME straight at the read-only mount instead would make every
# eval attempt fail on PermissionError, all nine retries, and the task would
# score the zero-shot baseline while looking healthy. So reproduce the overlay
# the same way environment/preflight.sh does for the agent: a real writable
# directory whose per-repo entries are symlinks into the mount. New downloads
# land beside them; dotted entries (.locks, .no_exist) are not linked, so HF's
# bookkeeping stays writable. `for x in "$d"/*` skips those for us.
# PTB_HF_CACHE_MOUNT is a colon-separated list — the cache ships as several
# Kaggle datasets to stay under the 1,000-file-per-dataset cap, and every
# shard carries the same top-level layout, so linking them all into one
# directory merges them. Same routine as environment/preflight.sh.
# D233. Place one cache repo into the writable HF_HOME: either a whole-dir
# symlink (cheap, correct for anything only ever READ) or a mirrored tree of
# real directories whose leaves are symlinks (writable at every depth, still
# copies zero bytes).
#
# Symlinking the repo dir makes the repo itself read-only ground, and
# huggingface writes INSIDE a repo on paths that look like pure reads. Two
# such writes are known, and they are in different trees:
#
#   hub/datasets--*   `revision=` that the shipped cache lacks -> huggingface_hub
#                     mkdirs snapshots/<rev> inside the repo. Only bfcl pins a
#                     revision (inspect_evals bfcl.py:46), and bfcl scored its
#                     baseline on 8 of 8 runs.
#   datasets/<name>   the datasets library takes a FileLock at
#                     <name>/<config>/<version>/..._builder.lock before
#                     download_and_prepare. Hit by humaneval and aime2025 --
#                     `datasets/openai___openai_humaneval/openai_humaneval/
#                     0.0.0/..._builder.lock: [Errno 30]`.
#
# The second is intermittent in a way that misled us: if the AGENT happened to
# materialise the dataset during its own run, the verifier finds it prepared
# and never takes the lock. So the same task passes for one candidate and
# fails for another, which looks like a model difference and is not.
#
# `cp -rs` is the whole trick: directories become real, files become symlinks
# into the mount. Nothing is copied, so this stays affordable for the 346 GiB,
# and every directory is writable so any lock/mkdir lands in scratch. chmod
# u+w because the mount's mode bits are inherited by the created dirs.
#
# models--* stay whole-dir symlinks: they are the bulk, and nothing writes
# inside a model repo. The original needs none of this -- it bind-mounts a
# fuse-overlayfs merged tree (run_task.sh:410-411), writable at every depth.
_ptb_place() {
    local top="$1" repo="$2" link="$3"
    case "$top/$(basename "$repo")" in
    datasets/*|modules/*|hub/datasets--*)
        if cp -rs "$repo" "$link" 2>/dev/null; then
            chmod -R u+w "$link" 2>/dev/null || true
            # Drop inherited lock symlinks. The shipped cache already contains
            # `<version>_builder.lock` files; cp -rs turns each into a symlink
            # into the read-only mount, and filelock opens it O_CREAT|O_RDWR,
            # which fails ELOOP ("too many levels of symbolic links") instead
            # of the EROFS we just removed -- a different error, same silent
            # baseline fallback. Verified against the real 372 GiB mount by
            # the zz-hfcache probe. Removing them lets huggingface create real
            # locks in the now-writable directory.
            find "$link" -type l \( -name '*.lock' -o -name '*.incomplete' \) \
                -delete 2>/dev/null || true
            return 0
        fi
        rm -rf "$link" 2>/dev/null || true
        # cp -rs is GNU-only; fall back to linking rather than shipping an
        # unpopulated cache entry, and say so loudly -- a silent fallback here
        # reproduces exactly the bug this function exists to remove.
        echo "[hf-cache] WARN: cp -rs unavailable for $repo; linking instead," \
             "writes inside it will fail with EROFS" >&2
        ln -s "$repo" "$link"
        ;;
    *)
        ln -s "$repo" "$link"
        ;;
    esac
}

if [ -n "${PTB_HF_CACHE_MOUNT:-}" ]; then
    mkdir -p "${TMP_HF_CACHE}"
    IFS=':' read -r -a _ptb_mounts <<< "${PTB_HF_CACHE_MOUNT}"
    for _mount in "${_ptb_mounts[@]}"; do
        [ -n "$_mount" ] && [ -d "$_mount" ] || continue
        for _src in "$_mount"/*; do
            [ -e "$_src" ] || continue
            _dst="${TMP_HF_CACHE}/$(basename "$_src")"
            if [ -d "$_src" ]; then
                mkdir -p "$_dst"
                _top="$(basename "$_src")"
                for _repo in "$_src"/*; do
                    [ -e "$_repo" ] || continue
                    _link="$_dst/$(basename "$_repo")"
                    [ -e "$_link" ] || [ -L "$_link" ] || \
                        _ptb_place "$_top" "$_repo" "$_link"
                done
            else
                [ -e "$_dst" ] || [ -L "$_dst" ] || ln -s "$_src" "$_dst"
            fi
        done
    done
    echo "linked ${#_ptb_mounts[@]} cache shard(s) into writable HF_HOME ${TMP_HF_CACHE}"
fi

# run_task.sh:427-449, verbatim including the pre-loop `sleep 5` and the
# early return when metrics.json already exists.
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

        timeout --signal=TERM --kill-after=60s 28800s bash -c \
            "$(declare -f run_evaluation); run_evaluation \"$max_tokens_arg\" \"$EVAL_COUNTER\""

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}
export -f run_evaluation
export EVAL_DIR REPO_ROOT EVAL_SCRIPT

# First evaluation: up to 4 attempts
run_evaluation_with_retry 4 ""

# Second evaluation with adjusted max tokens: up to 2 attempts
# (run_task.sh:454 says "up to 2" in its comment but passes 3; kept as-is.)
case "${EVALUATION_TASK}" in
    aime2025)         MAX_TOKENS_ARG="--max-tokens 12000" ;;
    arenahardwriting) MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
    bfcl)             MAX_TOKENS_ARG="--max-tokens 12000" ;;
    gpqamain)         MAX_TOKENS_ARG="--max-tokens 12000" ;;
    gsm8k)            MAX_TOKENS_ARG="--max-tokens 3000" ;;
    healthbench)      MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
    humaneval)        MAX_TOKENS_ARG="--max-tokens 3000" ;;
    *)                MAX_TOKENS_ARG="" ;;
esac

run_evaluation_with_retry 3 "$MAX_TOKENS_ARG"

# Third evaluation with further adjusted max tokens: up to 2 attempts
case "${EVALUATION_TASK}" in
    aime2025)         MAX_TOKENS_ARG="--max-tokens 8000" ;;
    arenahardwriting) MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
    bfcl)             MAX_TOKENS_ARG="--max-tokens 8000" ;;
    gpqamain)         MAX_TOKENS_ARG="--max-tokens 8000" ;;
    gsm8k)            MAX_TOKENS_ARG="--max-tokens 2000" ;;
    healthbench)      MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
    humaneval)        MAX_TOKENS_ARG="--max-tokens 2000" ;;
    *)                MAX_TOKENS_ARG="" ;;
esac

run_evaluation_with_retry 2 "$MAX_TOKENS_ARG"

if [ -f "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt" ]; then
    cat "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt"
fi

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

# ---------------------------------------------------------------------------
# Reward
#
# Upstream writes no reward: run_task.sh stops at metrics.json, and
# scripts/collect.py turns a run directory into a score days later. Harbor's
# contract is one scalar per task, so collect.py's rules have to be applied
# here. score_run.py is that translation and nothing more — every branch in
# it cites the collect.py line it implements.
# ---------------------------------------------------------------------------
echo "=== SCORING ==="
python3 "$TESTS/score_run.py" \
    --result-dir "$EVAL_DIR" \
    --metadata "$TESTS/metadata.json" \
    --reward-file "$EVAL_DIR/reward.txt"
SCORE_EXIT=$?

echo "Results in $EVAL_DIR:"
ls -la "$EVAL_DIR"

exit $SCORE_EXIT

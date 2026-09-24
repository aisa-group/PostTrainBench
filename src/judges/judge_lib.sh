#!/bin/bash
#
# Shared library for the reward-hacking judges in src/judges/.
#
# Used by:
#   - src/run_task.sh           inline judges right after the agent run
#                               (output files without suffix)
#   - src/judges/run_judges.sh  standalone/rerun runs (output files with the
#                               _rerun suffix)
#
# Each judge lives in src/judges/<judge_name>/ with a judge.conf (see
# load_judge_conf) and a prompt template. ALL_JUDGES defines the full set and
# the execution order.
#
# Callers source this file and use:
#   load_judge_conf <judge_name>
#   prepare_judge_sandbox <job_dir> <benchmark_id> <final_model_config_src>
#   setup_judge_codex_auth <job_dir>
#   build_judge_prompt <judge_name> <benchmark_id> <model_hf> <agent> <agent_config>
#   run_judge_exec <job_dir> <job_tmp> <output_json> <prompt>
#   collect_judge_output <job_dir> <out_dir> <name_suffix> <missing_fatal>
#
# run_judge_exec additionally reads the caller-provided array
# JUDGE_EXTRA_APPTAINER_ARGS (e.g. --nv + HF cache binds during run_task.sh;
# empty for standalone reruns).

JUDGES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGES_REPO_ROOT="$(cd "$JUDGES_DIR/../.." && pwd)"

# All judges, in execution order.
ALL_JUDGES=(data_contamination_judge api_usage_judge ptb_lookup_judge general_judge)

# codex CLI defaults shared by the judges; a judge.conf may override
# JUDGE_MODEL / JUDGE_REASONING_EFFORT / JUDGE_CODEX_VERSION per judge.
#
# 2026-08-31: OpenAI retired gpt-5.4 from Codex on ChatGPT-signed-in sessions
# (docs.chatgpt.com/models). All ChatGPT-authed judges must now use
# gpt-5.6-terra (successor) with codex CLI >= 0.144.0 for the compact-endpoint
# migration. general_judge has been running this pairing successfully since
# introduction, so we align the shared defaults to match.
JUDGE_DEFAULT_MODEL="gpt-5.6-terra"
JUDGE_DEFAULT_REASONING_EFFORT="xhigh"
JUDGE_DEFAULT_CODEX_VERSION="0.144.5"
JUDGE_CONTAINER="gpt_5_5.sif"

# load_judge_conf <judge_name>
# Loads src/judges/<judge_name>/judge.conf into the JUDGE_* variables,
# resetting them first so nothing leaks between judges.
load_judge_conf() {
    local judge_name="$1"
    local conf="$JUDGES_DIR/$judge_name/judge.conf"
    if [ ! -f "$conf" ]; then
        echo "ERROR: unknown judge '$judge_name' (no $conf)" >&2
        return 1
    fi
    JUDGE_LABEL=""
    JUDGE_OUTPUT_ID=""
    JUDGE_PROMPT_FILE=""
    JUDGE_MODEL="$JUDGE_DEFAULT_MODEL"
    JUDGE_REASONING_EFFORT="$JUDGE_DEFAULT_REASONING_EFFORT"
    # A version (e.g. "0.144.5") makes run_judge_exec npm-install exactly that
    # @openai/codex release into the sandbox home and run it instead of the
    # container's pinned codex. Set an empty override in judge.conf to fall
    # back to whatever codex the container ships.
    JUDGE_CODEX_VERSION="$JUDGE_DEFAULT_CODEX_VERSION"
    source "$conf"
    if [ -z "$JUDGE_LABEL" ] || [ -z "$JUDGE_OUTPUT_ID" ] || [ -z "$JUDGE_PROMPT_FILE" ]; then
        echo "ERROR: $conf must set JUDGE_LABEL, JUDGE_OUTPUT_ID and JUDGE_PROMPT_FILE" >&2
        return 1
    fi
}

# prepare_judge_sandbox <job_dir> <benchmark_id> <final_model_config_src>
# Copies the judge helper tooling and benchmark metadata into the sandbox
# home (shared by all judges).
prepare_judge_sandbox() {
    local job_dir="$1" benchmark_id="$2" final_model_config_src="$3"

    cp "$JUDGES_DIR/judge_tools/contamination_check.py" "$job_dir/contamination_check.py"
    cp "$JUDGES_DIR/judge_tools/model_identity_check.py" "$job_dir/model_identity_check.py"
    cp -r "$JUDGES_DIR/judge_tools/reference_configs" "$job_dir/reference_configs"

    # Expose final_model/config.json to the judge as ../final_model_config.json
    # so model_identity_check.py can compare it against the reference. Only the
    # config.json is needed for the architecture-identity check, not the weights.
    if [ -f "$final_model_config_src" ]; then
        cp "$final_model_config_src" "$job_dir/final_model_config.json"
    fi

    if [ -f "$JUDGES_REPO_ROOT/src/eval/tasks/$benchmark_id/test_data.json" ]; then
        cp "$JUDGES_REPO_ROOT/src/eval/tasks/$benchmark_id/test_data.json" "$job_dir/test_data.json"
    fi
}

# setup_judge_codex_auth <job_dir>
# Resets the sandbox codex config so agent-specific settings (e.g.
# model_reasoning_effort) can't leak into the judge, and prepares the
# ChatGPT-subscription auth: auth.json itself is bind-mounted from the shared
# location at apptainer exec time so codex can write the rotated refresh token
# back to the source and the next job picks it up instead of reusing a stale
# single-use refresh token. Sets JUDGE_CODEX_AUTH_SRC for run_judge_exec.
setup_judge_codex_auth() {
    local job_dir="$1"

    JUDGE_CODEX_AUTH_SRC="$JUDGES_REPO_ROOT/agents/codex_non_api/auth.json"
    if [ ! -f "$JUDGE_CODEX_AUTH_SRC" ]; then
        echo "ERROR: agents/codex_non_api/auth.json not found — the judges need subscription auth" >&2
        return 1
    fi

    cp -r "$JUDGES_REPO_ROOT/containers/other_home_data/.codex" "$job_dir/"
    # Touch a placeholder so apptainer has something to bind onto inside .codex/.
    : > "$job_dir/.codex/auth.json"
    if ! grep -q "forced_login_method" "$job_dir/.codex/config.toml" 2>/dev/null; then
        printf '\nforced_login_method = "chatgpt"\n' >> "$job_dir/.codex/config.toml"
    fi
}

# build_judge_prompt <judge_name> <benchmark_id> <model_hf> <agent> <agent_config>
# Prints the judge's prompt. Agent identity may be empty; it is only used by
# prompts that reference the agent harness (e.g. the API usage judge).
build_judge_prompt() {
    local judge_name="$1" benchmark_id="$2" model_hf="$3" agent="$4" agent_config="$5"
    local args=(--judge "$judge_name" --benchmark-id "$benchmark_id" --model "$model_hf")
    [ -n "$agent" ] && args+=(--agent "$agent")
    [ -n "$agent_config" ] && args+=(--agent-config "$agent_config")
    python "$JUDGES_DIR/get_judge_prompt.py" "${args[@]}"
}

# run_judge_exec <job_dir> <job_tmp> <output_json> <prompt>
# Runs the loaded judge's codex CLI in the sandbox, teeing the raw JSON trace
# to <output_json>. Requires load_judge_conf and setup_judge_codex_auth to
# have run; extra apptainer flags come from JUDGE_EXTRA_APPTAINER_ARGS.
# When judge.conf pins JUDGE_CODEX_VERSION, that exact @openai/codex release
# is npm-installed into a version-specific prefix in the sandbox home
# (idempotent across judges sharing the sandbox) and used instead of the
# container's codex.
run_judge_exec() {
    local job_dir="$1" job_tmp="$2" output_json="$3" prompt="$4"

    local codex_bin="codex"
    if [ -n "$JUDGE_CODEX_VERSION" ]; then
        local pin_prefix=".codex-cli-${JUDGE_CODEX_VERSION}"
        if [ ! -x "$job_dir/$pin_prefix/bin/codex" ]; then
            echo "  installing pinned codex CLI @openai/codex@${JUDGE_CODEX_VERSION} for ${JUDGE_LABEL} ..."
            apptainer exec \
                --containall \
                --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
                --bind "${job_tmp}:/tmp" \
                --home "${job_dir}:/home/ben" \
                --writable-tmpfs \
                "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}" \
                npm install -g --prefix "/home/ben/${pin_prefix}" --no-fund --no-audit "@openai/codex@${JUDGE_CODEX_VERSION}"
        fi
        if [ ! -x "$job_dir/$pin_prefix/bin/codex" ]; then
            echo "ERROR: install of pinned @openai/codex@${JUDGE_CODEX_VERSION} failed (no ${pin_prefix}/bin/codex in the sandbox home) — ${JUDGE_LABEL} cannot run" >&2
            return 1
        fi
        codex_bin="/home/ben/${pin_prefix}/bin/codex"
    fi

    apptainer exec \
        --containall \
        "${JUDGE_EXTRA_APPTAINER_ARGS[@]}" \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env CODEX_API_KEY="" \
        --env OPENAI_API_KEY="" \
        --env PYTHONNOUSERSITE="1" \
        --bind "${job_tmp}:/tmp" \
        --bind "${JUDGE_CODEX_AUTH_SRC}:/home/ben/.codex/auth.json" \
        --home "${job_dir}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}" \
        "$codex_bin" --search -a never exec --json -c model_reasoning_summary=detailed -c model_reasoning_effort="${JUDGE_REASONING_EFFORT}" --skip-git-repo-check --yolo --model "${JUDGE_MODEL}" "$prompt" 2>&1 | tee "$output_json"
}

# collect_judge_output <job_dir> <out_dir> <name_suffix> <missing_fatal>
# Parses the raw codex trace into a human-readable report and copies the
# judgement produced in the sandbox to
# <out_dir>/judgement_<JUDGE_OUTPUT_ID><name_suffix>.json. Returns 1 on a
# missing judgement only when <missing_fatal> is 1.
#
# <missing_fatal> is a property of the caller, not of the judge: standalone
# reruns (run_judges.sh) pass 1, because producing the verdict is the whole
# point of the job. run_task.sh passes 0 — a judge that produces no verdict
# must never cost a finished 10h agent run its evaluation, and the rerun
# pipeline can supply the verdict later.
collect_judge_output() {
    local job_dir="$1" out_dir="$2" suffix="$3" missing_fatal="$4"
    local out_base="judge_output_${JUDGE_OUTPUT_ID}${suffix}"
    local judgement="$out_dir/judgement_${JUDGE_OUTPUT_ID}${suffix}.json"

    python "$JUDGES_REPO_ROOT/src/trace_parsing/parse_trace.py" --agent codex "$out_dir/${out_base}.json" -o "$out_dir/${out_base}.txt"

    if [ -f "$job_dir/task/judgement.json" ]; then
        cp "$job_dir/task/judgement.json" "$judgement"
        echo "  ${JUDGE_LABEL} judgement: $(cat "$judgement")"
    elif [ "$missing_fatal" = "1" ]; then
        echo "ERROR: judgement.json not created by ${JUDGE_LABEL} (see $out_dir/${out_base}.txt)" >&2
        return 1
    else
        echo "WARNING: judgement.json not created by ${JUDGE_LABEL} (see $out_dir/${out_base}.txt); continuing — a missing inline verdict never aborts the task run" >&2
    fi
}

# tag_judgement_meta <judgement_path> <slot>
# Injects a "_meta" block (judge_model, judge_codex_version, slot, timestamp)
# into an existing verdict JSON so downstream tooling can tell which judge
# model produced it (the JUDGE_OUTPUT_ID alone doesn't distinguish gpt-5.4 vs
# gpt-5.6-terra after the 2026-08-31 migration). Silently no-op if the file
# doesn't exist. Reads JUDGE_MODEL / JUDGE_CODEX_VERSION from the currently
# loaded judge.conf.
tag_judgement_meta() {
    local judgement_path="$1" slot="$2"
    [ -f "$judgement_path" ] || return 0
    local ts tmpf
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # Temp file next to the verdict (same filesystem, so mv is an atomic
    # rename) carrying the verdict's own mode: mktemp creates files 0600, and
    # mv-ing that over the verdict would make it unreadable to everyone but
    # the job's owner (collect.py run by anyone else then crashes on it).
    tmpf="$(mktemp "${judgement_path}.XXXXXX")"
    chmod --reference="$judgement_path" "$tmpf"
    if jq --arg model "$JUDGE_MODEL" \
          --arg cv "${JUDGE_CODEX_VERSION:-container-default}" \
          --arg slot "$slot" \
          --arg ts "$ts" \
          '. + {_meta: {judge_model: $model, judge_codex_version: $cv, slot: ($slot|tonumber), timestamp: $ts}}' \
          "$judgement_path" > "$tmpf" 2>/dev/null; then
        mv "$tmpf" "$judgement_path"
    else
        rm -f "$tmpf"
    fi
}

# run_contamination_judge_slot <job_dir> <job_tmp> <out_dir> <slot> <benchmark> <model_hf> <agent> <agent_config> <missing_fatal>
#
# Runs data_contamination_judge and writes its output for one slot of the
# multi-run best-of-3 experiment:
#   slot 1 -> <out_dir>/judgement_gpt5_4.json (top level, backwards compat)
#   slot N -> <out_dir>/judgement_multi_runs/judgement_gpt5_4_run<N>.json
#
# The emitted judgement is tagged with a _meta block via tag_judgement_meta.
# Cleans $job_dir/task/judgement.json before invoking the judge (SAFE: any
# previous slot's judgement has already been copied out to its final path by
# collect_judge_output).
#
# Caller must have already run load_judge_conf "data_contamination_judge",
# prepare_judge_sandbox, and setup_judge_codex_auth. May also set
# JUDGE_EXTRA_APPTAINER_ARGS.
run_contamination_judge_slot() {
    local job_dir="$1" job_tmp="$2" out_dir="$3" slot="$4"
    local benchmark="$5" model_hf="$6" agent="$7" agent_config="$8"
    local missing_fatal="$9"

    local slot_out_dir suffix
    if [ "$slot" = "1" ]; then
        slot_out_dir="$out_dir"
        suffix=""
    else
        slot_out_dir="$out_dir/${MULTI_RUN_DIRNAME:-judgement_multi_runs}"
        mkdir -p "$slot_out_dir"
        suffix="_run${slot}"
    fi

    # Clean any leftover judgement in the sandbox — the previous slot/judge
    # has already been safely copied to its final path by collect_judge_output.
    rm -f "$job_dir/task/judgement.json"

    local prompt
    prompt=$(build_judge_prompt "data_contamination_judge" "$benchmark" "$model_hf" "$agent" "$agent_config")

    local output_json="$slot_out_dir/judge_output_${JUDGE_OUTPUT_ID}${suffix}.json"
    run_judge_exec "$job_dir" "$job_tmp" "$output_json" "$prompt"
    collect_judge_output "$job_dir" "$slot_out_dir" "$suffix" "$missing_fatal"

    tag_judgement_meta "$slot_out_dir/judgement_${JUDGE_OUTPUT_ID}${suffix}.json" "$slot"
}

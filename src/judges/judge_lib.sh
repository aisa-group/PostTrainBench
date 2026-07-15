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
ALL_JUDGES=(data_contamination_judge api_usage_judge ptb_lookup_judge)

# codex CLI defaults shared by the judges; a judge.conf may override
# JUDGE_MODEL / JUDGE_REASONING_EFFORT per judge.
JUDGE_DEFAULT_MODEL="gpt-5.4"
JUDGE_DEFAULT_REASONING_EFFORT="xhigh"
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
    JUDGE_MISSING_JUDGEMENT_FATAL_INLINE=1
    JUDGE_CONDOR_WEIGHT=1250
    JUDGE_MODEL="$JUDGE_DEFAULT_MODEL"
    JUDGE_REASONING_EFFORT="$JUDGE_DEFAULT_REASONING_EFFORT"
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
run_judge_exec() {
    local job_dir="$1" job_tmp="$2" output_json="$3" prompt="$4"

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
        codex --search -a never exec --json -c model_reasoning_summary=detailed -c model_reasoning_effort="${JUDGE_REASONING_EFFORT}" --skip-git-repo-check --yolo --model "${JUDGE_MODEL}" "$prompt" 2>&1 | tee "$output_json"
}

# collect_judge_output <job_dir> <out_dir> <name_suffix> <missing_fatal>
# Parses the raw codex trace into a human-readable report and copies the
# judgement produced in the sandbox to
# <out_dir>/judgement_<JUDGE_OUTPUT_ID><name_suffix>.json. Returns 1 on a
# missing judgement only when <missing_fatal> is 1.
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
    fi
}

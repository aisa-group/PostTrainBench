#!/bin/bash
# Run PostTrainBench on Modal through Harbor: one task, or a whole sweep in one command.
#
# Task selection (exactly one):
#   --task DIR                     an already generated task (run_adapter.py)
#   --benchmark B --base-model M   generate the tasks first, into tasks/<job-name>/.
#                                  Comma-separated lists, or `all`: every benchmark
#                                  condor scores (scripts/utils.py HARDCODED_BENCHMARKS;
#                                  bfcl is retired) / every base model (adapter.py MODELS). --num-hours H sets the agent budget (default 10).
#
# Every task gets its own Modal volume, ptb-<job-name>-<task>, which hands the trained
# model from the agent sandbox to the separate verifier sandbox (see
# template/task.toml). Harbor does not manage volumes, so this wrapper creates them and
# KEEPS them after the run (they are the trained models: `modal volume get <volume> /
# ./final_model`); --delete-volume removes them.
#
# A sweep (--benchmark/--base-model) launches every task at once (--parallel N caps
# it), each as its own harbor job in jobs/<job-name>/<task>/ with its launcher log in
# jobs/<job-name>/<task>.log, auto-confirms harbor's host-env prompt (--yes) and ends
# with a summary table. A single --task runs in the foreground in jobs/<job-name>/ and
# keeps harbor's interactive prompt. tasks/ and jobs/ live next to this script.
#
# Usage:
#   bash run_modal_task.sh --benchmark all --base-model all \
#       --agent claude-code --model anthropic/claude-opus-4-8 --job-name sweep1
#   bash run_modal_task.sh --task tasks/posttrainbench-gsm8k-qwen3-1.7b \
#       --agent claude-code --model anthropic/claude-opus-4-8
#   options: [--job-name NAME] [--num-hours H] [--parallel N] [--cli-version latest|<x.y.z>]
#       [--effort high|...] [--thinking-display summarized|omitted|none]
#       [--codex-auth-json agents/codex_non_api/auth.json] [--agent-kwarg k=v]
#       [--delete-volume] [-- <extra harbor run args>]
#
# Agent CLI version, same precedence as condor's src/utils/update_agent_cli.sh
# (claude-code, codex, gemini-cli, opencode):
#   1. --cli-version latest|<x.y.z>
#   2. <BIN>_CLI_VERSION (CLAUDE_CLI_VERSION, CODEX_CLI_VERSION, GEMINI_CLI_VERSION,
#      OPENCODE_CLI_VERSION): exported, else from .env. A pin is strict: it must exist
#      on npm, and harbor fails the trial if it cannot install it.
#   3. POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1: the version baked into the image
#      (template/environment/Dockerfile).
#   4. otherwise latest.
# `latest` is resolved with `npm view` once, at launch, so every task of a sweep runs
# the same version; the result is always passed to harbor as --ak version=<x.y.z> and
# recorded in result.json (agent_info.version -> cli_version.txt on export). condor's
# per-model pins in agents/*/solve.sh (claude_non_api_max, glmx) are not mirrored.
#
# Agent launch parity with PostTrainBench v1.1 agents/claude*/solve.sh:
#   - effort: CLAUDE_CODE_EFFORT_LEVEL=high by default (harbor: --ak reasoning_effort)
#   - BASH_MAX_TIMEOUT_MS=36000000 (harbor: --ae)
#   - `--thinking-display summarized` (harbor: --ak thinking_display, upstream
#     harbor-framework/harbor#3030; needs harbor >= 0.23.0). Without it the
#     stream-json trace carries EMPTY thinking blocks. `--thinking-display
#     none` omits the flag (older harbor rejects the kwarg).
#
# Keys come from the repo-root .env (condor's canonical key store; PTB_ENV_FILE
# overrides the path); an exported variable wins over .env. Loaded, and nothing else:
#   - OPENAI_API_KEY: always (the judges in the verifier)
#   - the agent's condor allowlist, agents/<agent>/api_keys.json (claude-code:
#     ANTHROPIC_API_KEY, gemini-cli: GEMINI_API_KEY, opencode: its three keys)
#   - claude-code on a Claude Max subscription: CLAUDE_CODE_OAUTH_TOKEN (from
#     `claude setup-token`) and CLAUDE_FORCE_OAUTH=1
#   - HF_TOKEN (read-only; here .env beats the shell), see the HF_TOKEN block below
set -euo pipefail

TASK=""; BENCHMARKS=""; BASE_MODELS=""; NUM_HOURS=""; PARALLEL=0
AGENT="claude-code"; MODEL=""; JOB_NAME=""; DELETE_VOLUME=0
CLI_VERSION=""; EFFORT="high"; CODEX_AUTH_JSON=""; THINKING_DISPLAY="summarized"
AGENT_KWARGS=(); EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --benchmark) BENCHMARKS="$2"; shift 2 ;;
        --base-model) BASE_MODELS="$2"; shift 2 ;;
        --num-hours) NUM_HOURS="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --cli-version) CLI_VERSION="$2"; shift 2 ;;
        --codex-auth-json) CODEX_AUTH_JSON="$2"; shift 2 ;;
        --effort) EFFORT="$2"; shift 2 ;;
        --thinking-display) THINKING_DISPLAY="$2"; shift 2 ;;
        --agent-kwarg|--ak) AGENT_KWARGS+=(--ak "$2"); shift 2 ;;
        --delete-volume) DELETE_VOLUME=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) awk 'NR == 1 { next } /^set -euo pipefail/ { exit } { print }' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] || { echo "need --model (the agent's model; see --help)" >&2; exit 2; }
if [ -n "$TASK" ]; then
    [ -z "$BENCHMARKS$BASE_MODELS$NUM_HOURS" ] || { echo "--task cannot be combined with --benchmark/--base-model/--num-hours" >&2; exit 2; }
    [ -d "$TASK" ] || { echo "task dir not found: $TASK" >&2; exit 2; }
    TASK="$(cd "$TASK" && pwd)"
else
    [ -n "$BENCHMARKS" ] && [ -n "$BASE_MODELS" ] || { echo "need --task, or --benchmark and --base-model (see --help)" >&2; exit 2; }
fi
NUM_HOURS="${NUM_HOURS:-10}"
[[ "$NUM_HOURS" =~ ^[1-9][0-9]*$ ]] || { echo "--num-hours must be a positive integer" >&2; exit 2; }
[[ "$PARALLEL" =~ ^[0-9]+$ ]] || { echo "--parallel must be a non-negative integer (0 = all at once)" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# API keys: an exported var wins; otherwise fall back to the repo-root .env
# (the condor pipeline's canonical key store). Read directly, like
# src/judges/rerun/commit_rerun_judges.sh — never source set_env_vars.sh (its
# module-loading block fails off the head node). Only the keys this flow uses
# are loaded; the rest of .env never enters the process env.
ENV_FILE="${PTB_ENV_FILE:-$REPO_ROOT/.env}"
env_file_value() {   # value of KEY=$1 in .env, outer quotes stripped; empty if absent
    [ -f "$ENV_FILE" ] || return 0
    grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed "s/^[\"']//; s/[\"']\$//" || true
}

# Harbor agent -> PostTrainBench agent name: selects the key allowlist here, the
# agent-specific prompt clauses at task generation, and tells the verifier which
# harness ran (judges' harness clause, trace parser; condor: AGENT / AGENT_CONFIG).
case "$AGENT" in
    claude-code) PTB_AGENT="claude" ;;
    gemini-cli)  PTB_AGENT="gemini" ;;
    *)           PTB_AGENT="$AGENT" ;;
esac
KEYS=(OPENAI_API_KEY)
if [ -f "$REPO_ROOT/agents/$PTB_AGENT/api_keys.json" ]; then
    KEYS+=($(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["allowed_api_keys"]))' "$REPO_ROOT/agents/$PTB_AGENT/api_keys.json"))
fi
[ "$AGENT" = "claude-code" ] && KEYS+=(CLAUDE_CODE_OAUTH_TOKEN CLAUDE_FORCE_OAUTH)
for _k in "${KEYS[@]}"; do
    if [ -z "${!_k:-}" ]; then
        _v="$(env_file_value "$_k")"
        [ -n "$_v" ] && export "$_k=$_v"
    fi
done
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set and not found in $ENV_FILE (needed by the verifier's judge)" >&2; exit 2; }

if _t="$(env_file_value HF_TOKEN)" && [ -n "$_t" ]; then
    HF_TOKEN_SOURCE="HF_TOKEN in $ENV_FILE"
elif [ -n "${HF_TOKEN:-}" ]; then
    _t="$HF_TOKEN"; HF_TOKEN_SOURCE="exported HF_TOKEN"
else
    echo "no Hugging Face token: set HF_TOKEN in $ENV_FILE or export it (a read-only token; needed for the gated Hub assets, see check_hf_token.py)" >&2
    exit 2
fi
export HF_TOKEN="$_t"
echo "hf token: $HF_TOKEN_SOURCE"
python3 "$SCRIPT_DIR/check_hf_token.py"

# The `modal` CLI must come from the same environment as `harbor` (harbor's
# venv has the modal extra; on hosts with an HTTP proxy it also needs the
# `python-socks` package or every Modal call fails with a connection error).
HARBOR_BIN="$(command -v harbor)" || { echo "harbor not on PATH" >&2; exit 2; }
HARBOR_PY="$(dirname "$(readlink -f "$HARBOR_BIN")")/python"
MODAL=("$HARBOR_PY" -m modal)

# ---- agent CLI version (precedence: see the header) ----
case "$AGENT" in
    claude-code) CLI_PKG="@anthropic-ai/claude-code"; CLI_ENV="CLAUDE_CLI_VERSION" ;;
    codex)       CLI_PKG="@openai/codex";             CLI_ENV="CODEX_CLI_VERSION" ;;
    gemini-cli)  CLI_PKG="@google/gemini-cli";        CLI_ENV="GEMINI_CLI_VERSION" ;;
    opencode)    CLI_PKG="opencode-ai";               CLI_ENV="OPENCODE_CLI_VERSION" ;;
    *)           CLI_PKG="";                          CLI_ENV="" ;;
esac
if [ -n "$CLI_PKG" ]; then
    if [ -n "$CLI_VERSION" ]; then
        CLI_SOURCE="--cli-version"
    elif [ -n "${!CLI_ENV:-}" ]; then
        CLI_VERSION="${!CLI_ENV}"; CLI_SOURCE="exported $CLI_ENV"
    elif _v="$(env_file_value "$CLI_ENV")" && [ -n "$_v" ]; then
        CLI_VERSION="$_v"; CLI_SOURCE="$CLI_ENV in $ENV_FILE"
    else
        _skip="${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-$(env_file_value POST_TRAIN_BENCH_SKIP_CLI_UPDATE)}"
        case "${_skip,,}" in
            1|true|yes|on)
                CLI_VERSION="$(grep -oE "${CLI_PKG}@[0-9][^ \\\\]*" "$SCRIPT_DIR/template/environment/Dockerfile" | head -1 | sed 's/.*@//')"
                [ -n "$CLI_VERSION" ] || { echo "POST_TRAIN_BENCH_SKIP_CLI_UPDATE is set but template/environment/Dockerfile pins no $CLI_PKG version" >&2; exit 2; }
                CLI_SOURCE="image pin (POST_TRAIN_BENCH_SKIP_CLI_UPDATE)" ;;
            *)  CLI_VERSION="latest"; CLI_SOURCE="default" ;;
        esac
    fi
    if [ "$CLI_VERSION" = "latest" ]; then
        CLI_VERSION="$(npm view "$CLI_PKG" version 2>/dev/null)" && [ -n "$CLI_VERSION" ] \
            || { echo "could not resolve the latest $CLI_PKG with npm view" >&2; exit 2; }
        CLI_SOURCE="$CLI_SOURCE: latest, resolved now"
    else
        # A pin is strict: catch a typo here rather than in every sandbox. npm exits
        # 1 both for an unknown version (E404) and when the registry is unreachable.
        if ! _npm_err="$(npm view "$CLI_PKG@$CLI_VERSION" version 2>&1 >/dev/null)"; then
            if grep -q E404 <<< "$_npm_err"; then
                echo "$CLI_PKG@$CLI_VERSION (from $CLI_SOURCE) does not exist on npm" >&2
            else
                echo "could not check $CLI_PKG@$CLI_VERSION on npm: $(grep -m1 'npm error' <<< "$_npm_err")" >&2
            fi
            exit 2
        fi
    fi
    AGENT_KWARGS+=(--ak "version=$CLI_VERSION")
    echo "cli:     $CLI_PKG@$CLI_VERSION ($CLI_SOURCE)"
elif [ -n "$CLI_VERSION" ]; then
    echo "--cli-version is supported for claude-code, codex, gemini-cli and opencode only" >&2; exit 2
fi

# ---- agent launch knobs ----
AGENT_ENV=()
if [ "$AGENT" = "codex" ]; then
    # PostTrainBench agents/codex*/solve.sh:
    #   codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model M
    # harbor's codex agent: codex exec --json --dangerously-bypass-approvals-and-sandbox
    #   --skip-git-repo-check --model M [-c ...]; effort defaults to high.
    # `codex` / `codex_non_api` = CLI default effort (medium): pass --effort medium;
    # `_high` / `_xhigh` variants: --effort high|xhigh. Subscription auth
    # (codex_non_api*): --codex-auth-json agents/codex_non_api/auth.json.
    [ -n "$EFFORT" ] && [ "$EFFORT" != "none" ] && AGENT_KWARGS+=(--ak "reasoning_effort=$EFFORT")
    AGENT_KWARGS+=(--ak "reasoning_summary=detailed" --ak "web_search=live")
    [ -n "$CODEX_AUTH_JSON" ] && export CODEX_AUTH_JSON_PATH="$(readlink -f "$CODEX_AUTH_JSON")"
fi
if [ "$AGENT" = "opencode" ]; then
    # PostTrainBench agents/opencode/solve.sh writes an opencode.json that
    # registers the providers it uses (incl. `zai` as a custom OpenAI-compatible
    # provider with its baseURL, and `{env:...}` api keys). Harbor's opencode
    # agent only declares the model, so hand it PTB's block via the
    # `opencode_config` kwarg (deep-merged into the generated opencode.json) and
    # pass the keys PTB's api_keys.json allows for this agent (loaded above).
    PTB_OPENCODE_JSON="$(sed -n "/cat > opencode.json << 'EOF'/,/^EOF/p" "$REPO_ROOT/agents/opencode/solve.sh" | sed '1d;$d' | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)))')" \
        || { echo "could not extract opencode.json from agents/opencode/solve.sh" >&2; exit 2; }
    AGENT_KWARGS+=(--ak "opencode_config=$PTB_OPENCODE_JSON")
    for k in $(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["allowed_api_keys"]))' "$REPO_ROOT/agents/opencode/api_keys.json"); do
        [ -n "${!k:-}" ] && AGENT_ENV+=(--ae "$k=${!k}")
    done
fi
if [ "$AGENT" = "claude-code" ]; then
    case "$THINKING_DISPLAY" in
        summarized|omitted) AGENT_KWARGS+=(--ak "thinking_display=$THINKING_DISPLAY") ;;
        none) ;;   # trace will carry empty thinking blocks
        *) echo "--thinking-display must be summarized, omitted or none" >&2; exit 2 ;;
    esac
    [ -n "$EFFORT" ] && [ "$EFFORT" != "none" ] && AGENT_KWARGS+=(--ak "reasoning_effort=$EFFORT")
    AGENT_ENV+=(--ae "BASH_MAX_TIMEOUT_MS=36000000")
fi

# Model without the provider prefix (anthropic/claude-opus-4-8 -> claude-opus-4-8).
VERIFIER_ENV=(--ve "PTB_AGENT_NAME=$PTB_AGENT" --ve "PTB_AGENT_CONFIG=${MODEL#*/}")

JOB_NAME="${JOB_NAME:-$(date +%Y%m%d-%H%M%S)}"
[[ "$JOB_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--job-name may only contain letters, digits, '.', '_' and '-'" >&2; exit 2; }
JOBS_ROOT="$SCRIPT_DIR/jobs"

# ---- tasks ----
if [ -n "$TASK" ]; then
    TASKS=("$TASK")
else
    TASKS_DIR="$SCRIPT_DIR/tasks/$JOB_NAME"
    [ ! -e "$TASKS_DIR" ] && [ ! -e "$JOBS_ROOT/$JOB_NAME" ] \
        || { echo "tasks/$JOB_NAME or jobs/$JOB_NAME already exists; pick another --job-name" >&2; exit 2; }
    GENERATED="$(python3 - "$SCRIPT_DIR" "$TASKS_DIR" "$BENCHMARKS" "$BASE_MODELS" "$NUM_HOURS" "$PTB_AGENT" <<'PY'
import contextlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from adapter import BENCHMARKS, DEFAULT_BENCHMARKS, MODELS, PostTrainBenchAdapter

out, bench_spec, model_spec, hours, agent = Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]

def pick(spec, known, what, default):
    names = list(default) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in known]
    if unknown or not names:
        sys.exit(f"unknown {what}: {', '.join(unknown) or repr(spec)} (known: {', '.join(known)})")
    return names

# `all` = the scored benchmarks (DEFAULT_BENCHMARKS: bfcl is retired); any benchmark by name.
benches = pick(bench_spec, BENCHMARKS, "benchmark", DEFAULT_BENCHMARKS)
models = pick(model_spec, MODELS, "base model", MODELS)
adapter = PostTrainBenchAdapter(output_dir=out, num_hours=hours, agent_name=agent)
for b in benches:
    for m in models:
        with contextlib.redirect_stdout(sys.stderr):
            task_dir = adapter.generate_task(b, m)
        print(f"TASKDIR\t{task_dir}")
PY
)" || { echo "task generation failed" >&2; exit 2; }
    mapfile -t TASKS < <(printf '%s\n' "$GENERATED" | sed -n 's/^TASKDIR\t//p')
    [ "${#TASKS[@]}" -gt 0 ] || { echo "no tasks generated" >&2; exit 2; }
    echo "tasks:   ${#TASKS[@]} generated in tasks/$JOB_NAME/ (agent budget ${NUM_HOURS}h)"
fi

task_short() { basename "$1" | sed 's/^posttrainbench-//'; }
volume_name() {   # Modal volume names: [a-zA-Z0-9._-], max 64 chars
    printf 'ptb-%s-%s' "$JOB_NAME" "$(task_short "$1")" | sed 's/[^A-Za-z0-9._-]/-/g' | cut -c1-64
}
DUPES="$(for t in "${TASKS[@]}"; do volume_name "$t"; echo; done | sort | uniq -d)"
[ -z "$DUPES" ] || { echo "volume names collide after the 64-char cut (use a shorter --job-name): $DUPES" >&2; exit 2; }

# run_one <task dir> <harbor jobs dir> <harbor job name> [extra harbor args...]
run_one() {
    local task="$1" jobs_dir="$2" job="$3" volume rc
    shift 3
    volume="$(volume_name "$task")"
    echo "task:    $task"
    echo "job:     $jobs_dir/$job"
    echo "volume:  $volume  (mounted at /mnt/ptb_final_model in agent + verifier)"
    "${MODAL[@]}" volume create "$volume" || { echo "could not create volume $volume" >&2; return 1; }
    set +e
    harbor run \
        --path "$task" \
        --agent "$AGENT" \
        --model "$MODEL" \
        "${AGENT_KWARGS[@]}" \
        "${AGENT_ENV[@]}" \
        "${VERIFIER_ENV[@]}" \
        --env modal \
        --ek "volumes={\"/mnt/ptb_final_model\":\"$volume\"}" \
        -n 1 \
        --jobs-dir "$jobs_dir" \
        --job-name "$job" \
        "$@" \
        "${EXTRA[@]}"
    rc=$?
    set -e
    echo
    echo "harbor exit code: $rc"
    echo "results:  $jobs_dir/$job/"
    if [ "$DELETE_VOLUME" = 1 ]; then
        "${MODAL[@]}" volume delete "$volume" --yes && echo "volume $volume deleted"
    else
        echo "model:    $HARBOR_PY -m modal volume get $volume / ./final_model_$(task_short "$task")"
        echo "cleanup:  $HARBOR_PY -m modal volume delete $volume --yes"
    fi
    return $rc
}

# ---- single task: foreground, harbor's interactive host-env prompt kept ----
if [ -n "$TASK" ]; then
    run_one "$TASK" "$JOBS_ROOT" "$JOB_NAME"
    exit $?
fi

# ---- sweep: one harbor job per task, launched concurrently ----
SWEEP_DIR="$JOBS_ROOT/$JOB_NAME"
mkdir -p "$SWEEP_DIR"
[ "$PARALLEL" -gt 0 ] || PARALLEL="${#TASKS[@]}"
echo "sweep:   jobs/$JOB_NAME/ (${#TASKS[@]} tasks, up to $PARALLEL at once; per-task logs jobs/$JOB_NAME/<task>.log)"
for t in "${TASKS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n || true; done
    # --yes: nobody can answer harbor's host-env prompt for concurrent runs. The
    # variables it lists are the ones printed above (keys resolved by this script).
    run_one "$t" "$SWEEP_DIR" "$(task_short "$t")" --yes > "$SWEEP_DIR/$(task_short "$t").log" 2>&1 &
    echo "launched $(task_short "$t")"
done
wait

echo
python3 - "$SWEEP_DIR" "${TASKS[@]}" <<'PY'
import json, re, sys
from pathlib import Path

sweep, tasks = Path(sys.argv[1]), sys.argv[2:]
print(f"{'task':<34} {'harbor':>6}  {'reward':>8}  note")
failed = 0
for t in tasks:
    short = re.sub(r"^posttrainbench-", "", Path(t).name)
    log = sweep / f"{short}.log"
    m = re.search(r"^harbor exit code: (\d+)", log.read_text(errors="replace"), re.M) if log.is_file() else None
    rc = m.group(1) if m else "?"
    reward, note = "-", ""
    results = sorted((sweep / short).glob("*/result.json"))
    if results:
        r = json.loads(results[-1].read_text())
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        reward = f"{rew:.4f}" if isinstance(rew, (int, float)) else "-"
        exc = r.get("exception_info") or {}
        note = f"{exc.get('exception_type')}: {(exc.get('exception_message') or '')[:60]}" if exc else ""
    else:
        note = f"no trial result, see {log.name}"
    failed += rc != "0" or reward == "-"
    print(f"{short:<34} {rc:>6}  {reward:>8}  {note}")
print(f"\n{len(tasks) - failed}/{len(tasks)} tasks produced a reward")
PY
echo "export: python3 $SCRIPT_DIR/harbor_to_results.py $SWEEP_DIR"

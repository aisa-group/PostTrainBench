#!/bin/bash
# Run one generated PostTrainBench Harbor task on Modal, with the shared
# volume that hands the trained model from the agent sandbox to the separate
# verifier sandbox (see template/task.toml for why a volume is needed).
#
# Harbor does not manage volume lifecycle, so this wrapper does:
#   1. create a Modal volume named ptb-<job-name>-<task-short-name>
#   2. harbor run ... --ek 'volumes={"/mnt/ptb_final_model":"<volume>"}'
#   3. print where the results and the model live
#
# One volume per task run; the volume is KEPT after the run (it is the trained
# model — fetch it with `modal volume get <volume> / ./final_model`), delete it
# with `modal volume delete <volume> --yes` or pass --delete-volume.
#
# Usage:
#   bash run_modal_task.sh --task tasks/posttrainbench-gsm8k-qwen3-1.7b \
#       --agent claude-code --model anthropic/claude-opus-4-8 \
#       [--job-name NAME] [--cli-version latest|<x.y.z>] [--effort high|...] \
#       [--codex-auth-json agents/codex_non_api/auth.json] [--agent-kwarg k=v] \
#       [--delete-volume] [-- <extra harbor run args>]
#
# Agent launch parity with PostTrainBench v1.1 agents/claude*/solve.sh:
#   - effort: CLAUDE_CODE_EFFORT_LEVEL=high by default (harbor: --ak reasoning_effort)
#   - BASH_MAX_TIMEOUT_MS=36000000 (harbor: --ae)
#   - CLI version: the image pin by default (opus_5.def era); condor's
#     update_agent_cli.sh upgrades to @latest at run start instead — pass
#     --cli-version latest (resolved via `npm view` now, so it is recorded) or
#     an explicit version; harbor installs it in the sandbox at agent setup.
#   - NOT replicated: `--thinking-display summarized` (harbor's claude-code
#     agent has a fixed command line); see README "Known Gotchas".
#
# Auth: ANTHROPIC_API_KEY for claude-code, or a Claude Max subscription via
#   export CLAUDE_CODE_OAUTH_TOKEN="$(cat ../../agents/claude_non_api/oauth_token)" CLAUDE_FORCE_OAUTH=1
# OPENAI_API_KEY is always required (contamination judge in the verifier).
set -euo pipefail

TASK=""; AGENT="claude-code"; MODEL=""; JOB_NAME=""; DELETE_VOLUME=0
CLI_VERSION=""; EFFORT="high"; CODEX_AUTH_JSON=""
AGENT_KWARGS=(); EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --cli-version) CLI_VERSION="$2"; shift 2 ;;
        --codex-auth-json) CODEX_AUTH_JSON="$2"; shift 2 ;;
        --effort) EFFORT="$2"; shift 2 ;;
        --agent-kwarg|--ak) AGENT_KWARGS+=(--ak "$2"); shift 2 ;;
        --delete-volume) DELETE_VOLUME=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$TASK" ] && [ -n "$MODEL" ] || { echo "need --task and --model (see --help)" >&2; exit 2; }
[ -d "$TASK" ] || { echo "task dir not found: $TASK" >&2; exit 2; }
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set (needed by the verifier's judge)" >&2; exit 2; }

# The `modal` CLI must come from the same environment as `harbor` (harbor's
# venv has the modal extra; on hosts with an HTTP proxy it also needs the
# `python-socks` package or every Modal call fails with a connection error).
HARBOR_BIN="$(command -v harbor)" || { echo "harbor not on PATH" >&2; exit 2; }
HARBOR_PY="$(dirname "$(readlink -f "$HARBOR_BIN")")/python"
MODAL=("$HARBOR_PY" -m modal)

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
    if [ "$CLI_VERSION" = "latest" ]; then
        CLI_VERSION="$(npm view @openai/codex version 2>/dev/null)" \
            || { echo "could not resolve latest @openai/codex via npm" >&2; exit 2; }
        echo "cli:     @openai/codex@$CLI_VERSION (latest, resolved now)"
    fi
    [ -n "$CLI_VERSION" ] && AGENT_KWARGS+=(--ak "version=$CLI_VERSION")
fi
# opencode / gemini-cli: harbor's agents always `npm i -g <pkg>@latest` unless a
# version is given (they never reuse the image's binary), so pin the image
# versions (containers/opus_5.def) by default to keep the pin-by-default policy.
if [ "$AGENT" = "opencode" ] || [ "$AGENT" = "gemini-cli" ]; then
    case "$AGENT" in
        opencode)   PKG="opencode-ai";        IMAGE_PIN="1.17.18" ;;
        gemini-cli) PKG="@google/gemini-cli"; IMAGE_PIN="0.18.4" ;;
    esac
    if [ "$CLI_VERSION" = "latest" ]; then
        CLI_VERSION="$(npm view "$PKG" version 2>/dev/null)" \
            || { echo "could not resolve latest $PKG via npm" >&2; exit 2; }
        echo "cli:     $PKG@$CLI_VERSION (latest, resolved now)"
    fi
    AGENT_KWARGS+=(--ak "version=${CLI_VERSION:-$IMAGE_PIN}")
fi
if [ "$AGENT" = "claude-code" ]; then
    [ -n "$EFFORT" ] && [ "$EFFORT" != "none" ] && AGENT_KWARGS+=(--ak "reasoning_effort=$EFFORT")
    AGENT_ENV+=(--ae "BASH_MAX_TIMEOUT_MS=36000000")
    if [ "$CLI_VERSION" = "latest" ]; then
        CLI_VERSION="$(npm view @anthropic-ai/claude-code version 2>/dev/null)" \
            || { echo "could not resolve latest @anthropic-ai/claude-code via npm" >&2; exit 2; }
        echo "cli:     @anthropic-ai/claude-code@$CLI_VERSION (latest, resolved now)"
    fi
    [ -n "$CLI_VERSION" ] && AGENT_KWARGS+=(--ak "version=$CLI_VERSION")
fi

# Tell the verifier which harness ran, for the judges' harness clause and the
# trace parser (condor: AGENT / AGENT_CONFIG). Harbor agent -> PostTrainBench
# agent name; model without the provider prefix (anthropic/claude-opus-4-8 -> claude-opus-4-8).
case "$AGENT" in
    claude-code) PTB_AGENT="claude" ;;
    gemini-cli)  PTB_AGENT="gemini" ;;
    *)           PTB_AGENT="$AGENT" ;;
esac
VERIFIER_ENV=(--ve "PTB_AGENT_NAME=$PTB_AGENT" --ve "PTB_AGENT_CONFIG=${MODEL#*/}")

TASK_SHORT="$(basename "$TASK" | sed 's/^posttrainbench-//')"
JOB_NAME="${JOB_NAME:-$(date +%Y%m%d-%H%M%S)}"
# Modal volume names: [a-zA-Z0-9._-], max 64 chars.
VOLUME="$(printf 'ptb-%s-%s' "$JOB_NAME" "$TASK_SHORT" | sed 's/[^A-Za-z0-9._-]/-/g' | cut -c1-64)"

echo "task:    $TASK"
echo "job:     $JOB_NAME"
echo "volume:  $VOLUME  (mounted at /mnt/ptb_final_model in agent + verifier)"

"${MODAL[@]}" volume create "$VOLUME"

set +e
harbor run \
    --path "$TASK" \
    --agent "$AGENT" \
    --model "$MODEL" \
    "${AGENT_KWARGS[@]}" \
    "${AGENT_ENV[@]}" \
    "${VERIFIER_ENV[@]}" \
    --env modal \
    --ek "volumes={\"/mnt/ptb_final_model\":\"$VOLUME\"}" \
    -n 1 \
    --job-name "$JOB_NAME" \
    "${EXTRA[@]}"
RC=$?
set -e

echo
echo "harbor exit code: $RC"
echo "results:  jobs/$JOB_NAME/"
if [ "$DELETE_VOLUME" = 1 ]; then
    "${MODAL[@]}" volume delete "$VOLUME" --yes
    echo "volume $VOLUME deleted"
else
    echo "model:    $HARBOR_PY -m modal volume get $VOLUME / ./final_model_$TASK_SHORT"
    echo "cleanup:  $HARBOR_PY -m modal volume delete $VOLUME --yes"
fi
exit $RC

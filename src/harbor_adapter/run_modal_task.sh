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
#       [--job-name NAME] [--agent-kwarg version=2.1.251] [--delete-volume] \
#       [-- <extra harbor run args>]
#
# Auth: ANTHROPIC_API_KEY for claude-code, or a Claude Max subscription via
#   export CLAUDE_CODE_OAUTH_TOKEN="$(cat ../../agents/claude_non_api/oauth_token)" CLAUDE_FORCE_OAUTH=1
# OPENAI_API_KEY is always required (contamination judge in the verifier).
set -euo pipefail

TASK=""; AGENT="claude-code"; MODEL=""; JOB_NAME=""; DELETE_VOLUME=0
AGENT_KWARGS=(); EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
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

#!/bin/bash
# Runs a command next to a HumanEval answer sandbox: a separate apptainer container in which the final-evaluation
# scorer (evaluate_final_eval.py) runs the model's answers, one fresh Python process per sample (see
# answer_sandbox.py). The sandbox has no network, none of the host filesystems and its own PID namespace, so the
# answers cannot reach the HumanEval tests, which only the scorer's checker in the eval container holds. It lives as
# long as the command.
#
# Usage: with_answer_sandbox.sh <sif> <command...>
#
# <sif> should be the eval container, so answers see the same Python packages as under upstream's scorer. <command>
# should be the `apptainer exec` of the evaluation: the sandbox's socket directory is added to it through
# APPTAINER_BIND, and APPTAINERENV_ANSWER_SANDBOX_SOCKET tells evaluate_final_eval.py where the socket is.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <sif> <command...>" >&2
    exit 1
fi
SIF="$1"
shift
if [ ! -f "$SIF" ]; then
    echo "ERROR: container $SIF not found" >&2
    exit 1
fi

SERVER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/answer_sandbox.py"
SOCKET_IN_CONTAINERS="/answer_sandbox/answers.sock"

# CONTROL_DIR stays on the host. Its socket/ subdirectory is the only host directory the sandbox sees.
CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ptb_answer_sandbox.XXXXXX")"
SOCKET_DIR="${CONTROL_DIR}/socket"
KEEPALIVE_FIFO="${CONTROL_DIR}/keepalive"
mkdir "$SOCKET_DIR"
mkfifo "$KEEPALIVE_FIFO"

SANDBOX_PID=""
KEEPALIVE_FD=""
cleanup() {
    # Closing the keepalive pipe makes the sandbox's server exit, which ends its PID namespace and every answer
    # process in it.
    if [ -n "$KEEPALIVE_FD" ]; then
        exec {KEEPALIVE_FD}>&-
    fi
    if [ -n "$SANDBOX_PID" ]; then
        for _ in $(seq 1 30); do
            kill -0 "$SANDBOX_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SANDBOX_PID" 2>/dev/null; then
            echo "WARNING: answer sandbox still running 30s after its keepalive closed; killing it" >&2
            kill -9 "$SANDBOX_PID"
        fi
        wait "$SANDBOX_PID" || true
    fi
    rm -rf "$CONTROL_DIR"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# The sandbox's (empty, container-only) home; not taken from $HOME, which batch jobs may not set.
SANDBOX_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
if [ -z "$SANDBOX_HOME" ]; then
    echo "ERROR: no passwd home directory for uid $(id -u)" >&2
    exit 1
fi

# The server's stdin is the keepalive FIFO. This script holds it open (read-write, so opening never blocks) until
# the command is done; the sandbox must not inherit that descriptor, or it would keep itself alive.
exec {KEEPALIVE_FD}<>"$KEEPALIVE_FIFO"
# env -i: the sandbox gets none of this shell's environment (no API keys, no APPTAINER_BIND of the caller).
(
    trap - EXIT TERM INT
    exec env -i PATH="$PATH" HOME="$SANDBOX_HOME" \
        apptainer exec -c --cleanenv --pid --no-init --net --network none \
            --bind "${SOCKET_DIR}:/answer_sandbox" \
            --bind "${SERVER}:/answer_sandbox.py:ro" \
            "$SIF" python /answer_sandbox.py serve "$SOCKET_IN_CONTAINERS"
) < "$KEEPALIVE_FIFO" {KEEPALIVE_FD}>&- &
SANDBOX_PID=$!

for _ in $(seq 1 120); do
    [ -S "${SOCKET_DIR}/answers.sock" ] && break
    if ! kill -0 "$SANDBOX_PID" 2>/dev/null; then
        echo "ERROR: answer sandbox exited before it was ready" >&2
        exit 1
    fi
    sleep 1
done
if [ ! -S "${SOCKET_DIR}/answers.sock" ]; then
    echo "ERROR: answer sandbox not ready after 120s" >&2
    exit 1
fi

export APPTAINER_BIND="${APPTAINER_BIND:+${APPTAINER_BIND},}${SOCKET_DIR}:/answer_sandbox"
export APPTAINERENV_ANSWER_SANDBOX_SOCKET="$SOCKET_IN_CONTAINERS"
# The command must not inherit the keepalive pipe, or the sandbox would outlive it.
set +e
"$@" {KEEPALIVE_FD}>&-
exit_code=$?
set -e
exit "$exit_code"

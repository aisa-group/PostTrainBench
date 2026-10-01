#!/bin/bash
# The EVAL_RUNTIME=local counterpart of with_answer_sandbox.sh, for callers that are already inside the eval image and
# cannot start a second apptainer container (the Harbor verifier: a Modal/gVisor sandbox, where apptainer's image and
# network setup do not work). Same contract: it runs a command next to a HumanEval answer sandbox in which the
# final-evaluation scorer (evaluate_final_eval.py) runs the model's answers, one fresh Python process per sample (see
# answer_sandbox.py), and the sandbox lives as long as the command.
#
# The sandbox is built from the namespaces apptainer itself uses, inside a user namespace (which needs no privileges):
#   - no network (its own network namespace, loopback only);
#   - its own PID namespace, the server as PID 1, so answers cannot see or signal anything outside;
#   - an allow-list filesystem: a fresh tmpfs root with only /usr, /lib, /lib64, /bin, /sbin and /etc bound in
#     read-only (the eval image's Python and packages, as in the apptainer sandbox), a private /tmp, /proc, a few
#     /dev nodes, the socket directory and answer_sandbox.py. The tests, logs, HF cache, keys and the rest of the
#     image's filesystem do not exist in there;
#   - run as `nobody` when started as root, and with an empty environment (no API keys).
# If any of that cannot be set up, the sandbox never starts and the command is not run: answers are never run
# unsandboxed.
#
# Usage: with_answer_sandbox_local.sh <command...>
# The command gets ANSWER_SANDBOX_SOCKET, the socket evaluate_final_eval.py connects to.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <command...>" >&2
    exit 1
fi
for tool in unshare chroot mount python3; do
    command -v "$tool" > /dev/null || { echo "ERROR: $tool not found; cannot build the answer sandbox" >&2; exit 1; }
done

SERVER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/answer_sandbox.py"

# CONTROL_DIR stays outside the sandbox. Its socket/ subdirectory is the only one of its directories the sandbox sees.
CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ptb_answer_sandbox.XXXXXX")"
SOCKET_DIR="${CONTROL_DIR}/socket"
KEEPALIVE_FIFO="${CONTROL_DIR}/keepalive"
mkdir "$SOCKET_DIR"
mkfifo "$KEEPALIVE_FIFO"
chmod 755 "$CONTROL_DIR"

# Started as root (the Harbor verifier), the sandbox runs as nobody, which then owns the socket directory.
DROP_PRIVS=()
if [ "$(id -u)" = 0 ]; then
    command -v setpriv > /dev/null || { echo "ERROR: setpriv not found; cannot drop root for the answer sandbox" >&2; exit 1; }
    chown 65534:65534 "$SOCKET_DIR"
    DROP_PRIVS=(setpriv --reuid=65534 --regid=65534 --clear-groups --inh-caps=-all)
fi

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

# Runs inside the new user, network, PID and mount namespaces: builds the allow-list root and becomes the server.
# $1 = socket dir, $2 = answer_sandbox.py.
SANDBOX_SETUP='
set -eu
ROOT="$(mktemp -d)"
mount -t tmpfs -o size=512m,mode=755 none "$ROOT"
for d in usr lib lib64 lib32 libx32 bin sbin etc; do
    [ -e "/$d" ] || continue
    if [ -L "/$d" ]; then
        ln -s "$(readlink "/$d")" "$ROOT/$d"
    else
        mkdir "$ROOT/$d"
        mount --rbind "/$d" "$ROOT/$d"
        mount -o remount,bind,ro "$ROOT/$d"
    fi
done
mkdir -p "$ROOT/answer_sandbox" "$ROOT/proc" "$ROOT/tmp" "$ROOT/dev"
mount --bind "$1" "$ROOT/answer_sandbox"
touch "$ROOT/answer_sandbox.py"
mount --bind "$2" "$ROOT/answer_sandbox.py"
mount -o remount,bind,ro "$ROOT/answer_sandbox.py"
mount -t proc proc "$ROOT/proc"
mount -t tmpfs -o mode=1777 none "$ROOT/tmp"
for n in null zero random urandom; do
    touch "$ROOT/dev/$n"
    mount --bind "/dev/$n" "$ROOT/dev/$n"
done
cd /
exec chroot "$ROOT" /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 \
    python3 /answer_sandbox.py serve /answer_sandbox/answers.sock
'

# The server's stdin is the keepalive FIFO. This script holds it open (read-write, so opening never blocks) until
# the command is done; the sandbox must not inherit that descriptor, or it would keep itself alive.
exec {KEEPALIVE_FD}<>"$KEEPALIVE_FIFO"
(
    trap - EXIT TERM INT
    exec env -i PATH="$PATH" \
        ${DROP_PRIVS[@]+"${DROP_PRIVS[@]}"} \
        unshare --user --map-root-user --net --pid --fork --mount --propagation private \
        sh -c "$SANDBOX_SETUP" answer-sandbox "$SOCKET_DIR" "$SERVER"
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

export ANSWER_SANDBOX_SOCKET="${SOCKET_DIR}/answers.sock"
echo "answer sandbox ready (namespace jail, pid ${SANDBOX_PID}): ${ANSWER_SANDBOX_SOCKET}" >&2
# The command must not inherit the keepalive pipe, or the sandbox would outlive it.
set +e
"$@" {KEEPALIVE_FD}>&-
exit_code=$?
set -e
exit "$exit_code"

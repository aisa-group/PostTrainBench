#!/bin/bash
# Auto-update an agent's CLI harness to the latest npm release and record the
# version that will actually run. Invoked from each agents/<agent>/solve.sh just
# before the CLI is launched (run_task.sh copies this script into the sandbox at
# /home/ben/update_agent_cli.sh).
#
# Usage: update_agent_cli.sh <cli-binary>
#   e.g. update_agent_cli.sh claude
#
# The binary -> npm package mapping below is the single source of truth; add a
# line here when introducing an agent that uses a new CLI.
#
# The agent sandbox runs as a non-root user (no --fakeroot), so the baked-in,
# root-owned global prefix (/usr/lib/node_modules) is read-only. We therefore
# install into the user-writable $HOME/.local prefix, which run_task.sh places
# ahead of the system path in PATH so the updated binary shadows the pinned one.
#
# The update is best-effort: if it fails (e.g. registry unreachable) we keep the
# container's pinned version and still record whatever is actually installed.

set -u

BIN="${1:?usage: update_agent_cli.sh <cli-binary>}"
VERSION_FILE="${CLI_VERSION_FILE:-$HOME/cli_version.txt}"

case "$BIN" in
    claude)   PKG="@anthropic-ai/claude-code" ;;
    codex)    PKG="@openai/codex" ;;
    gemini)   PKG="@google/gemini-cli" ;;
    opencode) PKG="opencode-ai" ;;
    *)
        echo "[update_agent_cli] ERROR: no npm package mapping for binary '$BIN'" >&2
        exit 1
        ;;
esac

# Opt-out: when POST_TRAIN_BENCH_SKIP_CLI_UPDATE is truthy, keep the container's
# pinned CLI and just record what's installed. Lets a run pin exact CLI versions
# by choosing the container alone.
SKIP="${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-}"
case "${SKIP,,}" in
    1|true|yes|on) SKIP_UPDATE=1 ;;
    *)             SKIP_UPDATE=0 ;;
esac

UPDATE_STATUS="success"
if [ "$SKIP_UPDATE" = "1" ]; then
    UPDATE_STATUS="skipped"
    echo "[update_agent_cli] POST_TRAIN_BENCH_SKIP_CLI_UPDATE set; using pinned ${BIN}"
else
    echo "[update_agent_cli] updating ${BIN} (${PKG}) to latest ..."
    if ! timeout 300 npm install -g --prefix "$HOME/.local" --no-fund --no-audit "${PKG}@latest"; then
        UPDATE_STATUS="failed"
        echo "[update_agent_cli] WARNING: update failed; falling back to pinned ${BIN}" >&2
    fi
fi

# Forget cached command locations so the freshly installed binary is resolved.
hash -r 2>/dev/null || true

RESOLVED_PATH="$(command -v "$BIN" || echo "<not found>")"
VERSION_OUTPUT="$("$BIN" --version 2>&1 || echo "<version lookup failed>")"

{
    echo "binary: ${BIN}"
    echo "package: ${PKG}"
    echo "path: ${RESOLVED_PATH}"
    echo "version: ${VERSION_OUTPUT}"
    echo "update: ${UPDATE_STATUS}"
    echo "recorded_at: $(date -Iseconds)"
} > "$VERSION_FILE"

echo "[update_agent_cli] ${BIN} -> ${VERSION_OUTPUT} (${RESOLVED_PATH})"
echo "[update_agent_cli] wrote ${VERSION_FILE}"

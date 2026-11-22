#!/bin/bash
# Smoke test of agents/opencode_neon/solve.sh in the agent container, launched like run_task.sh does (apptainer -c
# --cleanenv --pid --no-init, home = a job dir, cwd /home/ben/task), with a short test prompt instead of a task.
# A decoy OPENROUTER_API_KEY is in the sandbox, as on the writing benchmarks (their grader key); the run only works if
# opencode uses OPENROUTER_API_KEY_GEMINI_NEON. Checks, failing the script if one fails: the pinned opencode install,
# the model check and the xhigh variant check, model turns through OpenRouter, one resume after the session ends
# without final_model, reasoning events in the trace, ~/agent_logs/opencode (session database), trace parsing, and that
# the key is in no output file.
# Run on a compute node (smoke_test.sub) from the repo root. Costs a few cents on the Neon key.
# Output: <parent of POST_TRAIN_BENCH_RESULTS_DIR>/smoke/opencode_neon_<run id>
# The prompt has no commas: apptainer splits an --env value at commas when a later piece holds a '=' ("invalid
# argument").
# Usage: bash dev_utils/opencode_neon/smoke_test.sh <run id> [agent_config]
set -euo pipefail
RUN_ID="${1:?usage: smoke_test.sh <run id> [agent_config]}"
AGENT_CONFIG="${2:-openrouter/google/gemini-4-neon:xhigh}"
REPO="$(pwd)"
[ -f "$REPO/agents/opencode_neon/solve.sh" ] || { echo "ERROR: run from the repo root" >&2; exit 1; }
env_value() { bash -c 'set -a; source <(grep "^$1=" .env); printf %s "${!1}"' _ "$1"; }
KEY="$(env_value OPENROUTER_API_KEY_GEMINI_NEON)"
[ -n "$KEY" ] || { echo "ERROR: no OPENROUTER_API_KEY_GEMINI_NEON in .env" >&2; exit 1; }
CONTAINERS_DIR="$(env_value POST_TRAIN_BENCH_CONTAINERS_DIR)"
CONTAINER_NAME="$(env_value POST_TRAIN_BENCH_CONTAINER_NAME)"
RESULTS_DIR="$(env_value POST_TRAIN_BENCH_RESULTS_DIR)"
[ -n "$RESULTS_DIR" ] || { echo "ERROR: no POST_TRAIN_BENCH_RESULTS_DIR in .env" >&2; exit 1; }
OUT="$(dirname "$RESULTS_DIR")/smoke/opencode_neon_${RUN_ID}"
[ ! -e "$OUT" ] || { echo "ERROR: $OUT exists; use another run id" >&2; exit 1; }
mkdir -p "$OUT"

JOB_DIR="$(mktemp -d /tmp/opencode_neon_smoke.XXXXXX)"
mkdir -p "$JOB_DIR/task" "$JOB_DIR/tmp"
bash src/utils/create_timer.sh 1 "$JOB_DIR/task/timer.sh"
cp agents/opencode_neon/solve.sh "$JOB_DIR/agent_solve.sh"
cp src/utils/update_agent_cli.sh "$JOB_DIR/update_agent_cli.sh"

PROMPT='This is a short test of the agent harness and not a real task. Do exactly these steps and nothing else.
1. Run this command: printenv | grep -o ^OPENROUTER[A-Z_]*
2. Run this command: timeout 5 sleep 30; echo timeout exit $?
3. Reply with one line that repeats the output of step 1 and then end your turn. Do not create /home/ben/task/final_model now.
If you later get a message that your session ended: create the file /home/ben/task/final_model/README.txt that contains the word ok and then reply DONE.'

set +e
apptainer exec \
    -c \
    --cleanenv \
    --pid \
    --no-init \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
    --env OPENROUTER_API_KEY_GEMINI_NEON="$KEY" \
    --env OPENROUTER_API_KEY="sk-or-v1-decoy0000000000000000000000000000000000000000000000000000" \
    --env PYTHONNOUSERSITE="1" \
    --env PROMPT="$PROMPT" \
    --env AGENT_CONFIG="$AGENT_CONFIG" \
    --bind "$JOB_DIR/tmp:/tmp" \
    --home "$JOB_DIR:/home/ben" \
    --pwd /home/ben/task \
    --writable-tmpfs \
    "$CONTAINERS_DIR/$CONTAINER_NAME.sif" \
    bash -c 'bash /home/ben/agent_solve.sh; echo "=== smoke: solve.sh exit $?"' \
    > "$OUT/solve_out.txt" 2>&1
echo "apptainer exit $?" >> "$OUT/solve_out.txt"
set -e

cp -r "$JOB_DIR/agent_logs" "$OUT/agent_logs"
cp "$JOB_DIR/cli_version.txt" "$JOB_DIR/task/opencode.json" "$OUT/"
ls -la "$JOB_DIR/task" "$JOB_DIR/task/final_model" > "$OUT/task_listing.txt" 2>&1 || true
FINAL_MODEL_OK=0
grep -qx ok "$JOB_DIR/task/final_model/README.txt" 2>/dev/null && FINAL_MODEL_OK=1
rm -rf "$JOB_DIR"
(cd src/trace_parsing && python3 parse_trace.py --agent opencode_neon "$OUT/solve_out.txt" -o "$OUT/solve_parsed.txt")

failed=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; failed=1; fi
}
check "solve.sh exit 0" 'grep -q "^=== smoke: solve.sh exit 0$" "$OUT/solve_out.txt"'
check "opencode 1.19.6 installed" 'grep -q "^version: 1.19.6$" "$OUT/cli_version.txt"'
check "model turns" '[ "$(grep -c "\"type\":\"step_finish\"" "$OUT/solve_out.txt")" -ge 2 ]'
check "one resume" '[ "$(grep -c "^=== solve.sh: resume" "$OUT/solve_out.txt")" -eq 1 ]'
check "final_model written after the resume" '[ "$FINAL_MODEL_OK" = 1 ]'
check "reasoning events" 'grep -q "\"type\":\"reasoning\"" "$OUT/solve_out.txt"'
check "session database in agent_logs" '[ -s "$OUT/agent_logs/opencode/opencode.db" ]'
check "no provider error" '! grep -q "\"type\":\"error\"" "$OUT/solve_out.txt"'
check "key in no output file" '! grep -rqF "$KEY" "$OUT"'
echo "output: $OUT"
exit "$failed"

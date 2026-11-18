#!/bin/bash
# Smoke test of agents/codex_openrouter_max/solve.sh in the agent container, launched like run_task.sh does
# (apptainer -c --cleanenv --pid --no-init, home = a job dir, cwd /home/ben/task, auth.json bind), with a short test
# prompt instead of a task. Checks: OpenRouter model check, codex 0.170.0 install, relay start (random port, token),
# model turns through OpenRouter, web search through the ChatGPT subscription, what the agent's commands can see
# (harness_check.txt, written by a script, not reported by the model: the OpenRouter key in their environment and in
# /proc/*/environ, the relay without its token), relay restart after the agent kills it, the resume loop after the
# agent kills codex, ~/agent_logs (relay log, codex sessions), trace parsing, and that the key is in no output file.
# Run on a compute node (smoke_test.sub) from the repo root. Costs about $0.50 on the OpenRouter key.
# Output: /fast/brank/auto-posttrain/smoke/codex_openrouter_<run id>
# Usage: bash dev_utils/codex_openrouter/smoke_test.sh <run id> [agent_config]
set -euo pipefail
RUN_ID="${1:?usage: smoke_test.sh <run id> [agent_config]}"
AGENT_CONFIG="${2:-gpt-6.2-astra}"
REPO="$(pwd)"
[ -f "$REPO/agents/codex_openrouter_max/solve.sh" ] || { echo "ERROR: run from the repo root" >&2; exit 1; }
KEY="$(bash -c 'set -a; source <(grep "^OPENROUTER_API_KEY_OPENAI=" .env); printf %s "$OPENROUTER_API_KEY_OPENAI"')"
[ -n "$KEY" ] || { echo "ERROR: no OPENROUTER_API_KEY_OPENAI in .env" >&2; exit 1; }
CONTAINERS_DIR="$(bash -c 'set -a; source <(grep "^POST_TRAIN_BENCH_CONTAINERS_DIR=" .env); printf %s "$POST_TRAIN_BENCH_CONTAINERS_DIR"')"
CONTAINER_NAME="$(bash -c 'set -a; source <(grep "^POST_TRAIN_BENCH_CONTAINER_NAME=" .env); printf %s "$POST_TRAIN_BENCH_CONTAINER_NAME"')"
AUTH_SRC="$(readlink -f agents/codex_openrouter_max/auth.json)"
[ -s "$AUTH_SRC" ] || { echo "ERROR: agents/codex_openrouter_max/auth.json does not resolve to a file" >&2; exit 1; }
OUT="/fast/brank/auto-posttrain/smoke/codex_openrouter_${RUN_ID}"
mkdir -p "$OUT"

JOB_DIR="$(mktemp -d /tmp/codex_openrouter_smoke.XXXXXX)"
mkdir -p "$JOB_DIR/task" "$JOB_DIR/tmp"
cp -r containers/other_home_data/.codex "$JOB_DIR/"
: > "$JOB_DIR/.codex/auth.json"
bash src/utils/create_timer.sh 1 "$JOB_DIR/task/timer.sh"
cat > "$JOB_DIR/task/harness_check.sh" <<'EOF'
#!/bin/bash
# What the agent's commands can see (written by dev_utils/codex_openrouter/smoke_test.sh).
{
python3 - <<'PY'
import glob
import os

print("own environment: values that are OpenRouter keys:", sum(v.startswith("sk-or-") for v in os.environ.values()))
print("own environment has OPENROUTER_API_KEY_OPENAI:", "OPENROUTER_API_KEY_OPENAI" in os.environ)
hits = []
for path in glob.glob("/proc/[0-9]*/environ"):
    pid = path.split("/")[2]
    try:
        environ = open(path, "rb").read()
        cmdline = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")[:100].decode(errors="replace")
    except OSError:
        continue
    if b"sk-or-" in environ:
        hits.append(f"{pid}: {cmdline}")
print("processes with an OpenRouter key in /proc/<pid>/environ:", hits)
PY
port=$(grep -o '127.0.0.1:[0-9]*' ~/.codex_openrouter/config.toml | cut -d: -f2)
echo "relay port: $port"
echo "relay POST without token: HTTP $(curl -s -o /dev/null -w '%{http_code}' -X POST -d '{}' "http://127.0.0.1:$port/api/v1/responses")"
} 2>&1 | tee /home/ben/task/harness_check.txt
EOF
cp agents/codex_openrouter_max/solve.sh "$JOB_DIR/agent_solve.sh"
cp src/utils/update_agent_cli.sh "$JOB_DIR/update_agent_cli.sh"

PROMPT='This is a short test of the agent harness, not a real task. Do exactly these steps, one tool call each:
1. Run in the shell: echo $((6*7))
2. Create the file notes.txt with the single line alpha, using apply_patch. Then change alpha to beta with apply_patch.
3. Do one web search for "Hugging Face TRL SFTTrainer documentation" and give the title of one result.
4. Run: bash /home/ben/task/harness_check.sh
5. Run: pkill -f or_relay.py; echo killed
6. Run: echo after-relay-kill
7. Run this exactly once, never again, even later: pkill -f "codex --search exec"
8. When you are told that your session ended, run: echo after-resume
Then reply with one short line per step and end. Do not create /home/ben/task/final_model.'

set +e
apptainer exec \
    -c \
    --cleanenv \
    --pid \
    --no-init \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
    --env OPENROUTER_API_KEY_OPENAI="$KEY" \
    --env PYTHONNOUSERSITE="1" \
    --env PROMPT="$PROMPT" \
    --env AGENT_CONFIG="$AGENT_CONFIG" \
    --bind "$JOB_DIR/tmp:/tmp" \
    --bind "$AUTH_SRC:/home/ben/.codex/auth.json" \
    --home "$JOB_DIR:/home/ben" \
    --pwd /home/ben/task \
    --writable-tmpfs \
    "$CONTAINERS_DIR/${CONTAINER_NAME}.sif" \
    bash -c 'bash /home/ben/agent_solve.sh; echo "=== smoke: solve.sh exit $?"; echo "=== smoke: relay processes left:"; pgrep -af "[o]r_relay.py" || echo none' \
    > "$OUT/solve_out.txt" 2>&1
echo "apptainer exit $?" >> "$OUT/solve_out.txt"
set -e

cp -r "$JOB_DIR/agent_logs" "$OUT/agent_logs"
cp "$JOB_DIR/cli_version.txt" "$OUT/"
cp "$JOB_DIR/.codex_openrouter/config.toml" "$OUT/codex_config.toml"
cp "$JOB_DIR/task/harness_check.txt" "$OUT/" || echo "WARNING: no harness_check.txt (the agent did not run step 4)"
ls -la "$JOB_DIR/task" > "$OUT/task_listing.txt" 2>&1 || true
(cd src/trace_parsing && python3 parse_trace.py --agent codex_openrouter_max "$OUT/solve_out.txt" -o "$OUT/solve_parsed.txt")
if grep -q -F "$KEY" -r "$OUT"; then echo "WARNING: the OpenRouter key appears in $OUT"; else echo "key not in $OUT"; fi
rm -rf "$JOB_DIR"
echo "done: $OUT"

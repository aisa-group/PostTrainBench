#!/bin/bash
# No `set -e`, like src/run_task.sh: every step here is best-effort, and harbor
# ignores this script's exit code. The one hard requirement is that reward.txt
# ends up holding a single finite number — anything that aborts the script
# before that turns a finished run into a trial error with no reward.

# PostTrainBench verification script
# Runs condor's own judge phase (src/judges/judge_lib.sh run_all_judges) and
# final evaluation (src/eval/run_final_eval.sh), in run_task.sh's order.
#
# Tamper-resistance design (harbor 0.7.0 separate-verifier mode):
#   - This script runs in a SEPARATE container from the agent (see
#     [verifier].environment_mode = "separate" in task.toml). The agent
#     never has shell or filesystem access to this container, so it
#     can't tamper with the final-eval scripts, templates/, the Python interpreter,
#     installed packages (vllm, inspect_evals, transformers), or this
#     script itself.
#   - All verifier-side files (metadata.json and ptb/ = the repo slice with
#     src/judges, src/trace_parsing, src/eval/run_final_eval.sh, the task dir,
#     templates and src/utils helpers) are
#     BAKED INTO the verifier image at build time (see tests/Dockerfile)
#     and live at /tests/.
#   - The agent's code arrives as a size-filtered snapshot in
#     /logs/artifacts/workspace (staged by ptb_collect.sh, transferred by
#     harbor's conventional artifact dir); the judges get a copy of it as
#     their task dir (run_all_judges, src/judges/judge_lib.sh).
#   - The agent's final_model is the only file the verifier executes
#     code against (via vllm). Bad weights are penalized by the eval
#     score, not by tampering.
#   - The weights arrive via a shared Modal volume mounted read-write in the
#     agent sandbox and mounted here at $PTB_MODEL_DIR (populated by the
#     [[verifier.collect]] hook in task.toml). The workspace transfer carries
#     only code. See task.toml 

TESTS="/tests"
WORKSPACE="/home/agent/workspace"
LOGS_DIR="/logs/verifier"
# Where the trained model lives. task.toml sets PTB_MODEL_DIR to the shared
# Modal volume mount (/mnt/ptb_final_model) that the [[verifier.collect]] hook
# populated from the agent's workspace; without it we fall back to the
# workspace copy (shared-verifier / non-Modal setups).
MODEL_DIR="${PTB_MODEL_DIR:-$WORKSPACE/final_model}"
# Where the agent's code lives for the contamination judge. ptb_collect.sh
# stages a size-filtered snapshot into /logs/artifacts/workspace on the agent
# side; harbor re-materializes /logs/artifacts here. Fall back to the
# workspace (shared-verifier setups, or if the snapshot is missing).
CODE_DIR="/logs/artifacts/workspace"
if [ ! -d "$CODE_DIR" ] || [ -z "$(ls -A "$CODE_DIR" 2>/dev/null)" ]; then
    echo "WARNING: no code snapshot at $CODE_DIR, judge will read $WORKSPACE"
    CODE_DIR="$WORKSPACE"
fi

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Verifier ==="
echo "Tests dir: $TESTS"
echo "Workspace: $WORKSPACE"
echo "Logs dir: $LOGS_DIR"
echo "Model dir: $MODEL_DIR"
echo "Code dir (judge): $CODE_DIR"
# [verifier.env] HF_TOKEN: gpqamain's evaluate.py loads the gated dataset from the Hub.
echo "HF_TOKEN: $([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"

# Check GPU availability
echo ""
echo "=== GPU Check ==="
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

# No submitted model: record the base-model score and explicit skipped-judge
# verdicts so the exported run remains aggregatable without calling any judges.
write_no_model_results() {
    python3 - "$TESTS/metadata.json" "$LOGS_DIR" "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

metadata_path, logs_path, reason = sys.argv[1:]
metadata = json.loads(Path(metadata_path).read_text())
accuracy = metadata["baseline_accuracy"]
if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
    raise ValueError(f"Invalid baseline_accuracy in {metadata_path}: {accuracy!r}")

logs = Path(logs_path)
judge_fields = {
    "gpt5_4": ("contamination", "disallowed_model"),
    "api": ("disallowed_api_usage",),
    "ptb_lookup": ("disallowed_ptb_lookup",),
    "general": ("general_anomaly",),
}
for judge_id, fields in judge_fields.items():
    verdict = {}
    for field in fields:
        verdict[field] = False
        verdict[f"justification_{field}"] = reason
    (logs / f"judgement_{judge_id}.json").write_text(json.dumps(verdict, indent=2) + "\n")

(logs / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "error": reason}, indent=2) + "\n")
(logs / "reward.txt").write_text(f"{accuracy}\n")
print(f"{reason}; skipping judges and evaluation, using baseline accuracy {accuracy}")
PY
}

# Check if final_model exists in agent's workspace
echo ""
echo "=== Checking final_model ==="
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: final_model directory not found at $MODEL_DIR"
    { ls -la "$WORKSPACE"; echo "--- $(dirname "$MODEL_DIR") ---"; ls -la "$(dirname "$MODEL_DIR")"; } > "$LOGS_DIR/workspace_listing.txt" 2>&1 || true
    write_no_model_results "No final model submitted"
    exit 0
fi

# Check if final_model has required files
echo "Contents of final_model:"
# Trailing slash: MODEL_DIR is the volume mount, which Modal exposes as a
# symlink; without it `ls -la` lists the link itself, not the model files.
ls -la "$MODEL_DIR/" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    write_no_model_results "No valid final model submitted: final_model/config.json not found"
    exit 0
fi

# Show model config
echo ""
echo "=== Model config.json ==="
cat "$MODEL_DIR/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Check for tokenizer
echo ""
echo "=== Checking tokenizer files ==="
ls -la "$MODEL_DIR/"*token* 2>/dev/null || echo "No tokenizer files found with 'token' in name"
ls -la "$MODEL_DIR/"*.json 2>/dev/null || echo "No json files found"

# ============================================================
# Read metadata for benchmark and model info — from /tests, NOT workspace,
# so the agent can't redirect the verifier by overwriting metadata.json.
# ============================================================
BENCHMARK_ID=""
BENCHMARK_NAME=""
MODEL_ID=""

if [ -f "$TESTS/metadata.json" ]; then
    BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])" 2>/dev/null || echo "")
    BENCHMARK_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_name'])" 2>/dev/null || echo "Unknown")
    MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])" 2>/dev/null || echo "Unknown")
    echo "Benchmark ID: $BENCHMARK_ID"
    echo "Benchmark Name: $BENCHMARK_NAME"
    echo "Model: $MODEL_ID"
fi

# ============================================================
# Reward-hacking judges (PostTrainBench v1.1, src/judges/)
#
# Port of src/judges/judge_lib.sh + the judge loop in src/run_task.sh to the
# harbor verifier. The judge set, order, prompts, per-judge model/effort/CLI
# pins and tools all come verbatim from /tests/ptb/src/judges (baked in by the
# adapter), so a judge added upstream is picked up on regeneration.
#
# Sandbox layout (condor: /home/ben/{task,solve_parsed.txt,...}; here
# $JUDGE_HOME), exactly what the prompts reference relative to the task dir:
#   task/                    writable copy of the agent's code snapshot, with
#                            final_model -> $MODEL_DIR (read-only volume)
#   solve_out.txt            raw agent trace (harbor's /logs/agent/<agent>.txt,
#                            shipped via ptb_collect.sh)
#   solve_parsed.txt         human-readable trace (src/trace_parsing)
#   test_data.json           benchmark test set (n-gram checker reference)
#   contamination_check.py, model_identity_check.py, reference_configs/
#   final_model_config.json  copy of final_model/config.json
#
# Auth: OpenAI API key (OPENAI_API_KEY / CODEX_API_KEY from [verifier.env]).
# Condor uses a ChatGPT-subscription auth.json bind mount instead; the models
# and CLI invocation are the same.
#
# A judge that produces no judgement.json is a WARNING (as in run_task.sh):
# the agent's work is already done and must still be evaluated; verdicts can
# be re-run later on the exported result dir with src/judges/run_judges.sh.
# ============================================================
echo ""
echo "=== Running reward-hacking judges ==="

PTB="$TESTS/ptb"                      # mini repo layout: src/judges, src/trace_parsing, src/eval/tasks/<id>/info.json
JUDGES_DIR="$PTB/src/judges"
TRACE_PARSER="$PTB/src/trace_parsing/parse_trace.py"
JUDGE_HOME="${PTB_JUDGE_HOME:-/tmp/ptb_judge}"
# Per judge run; task.toml sets it and derives the verifier timeout from it
# (adapter.py: judge runs x this + the eval budget).
JUDGE_TIMEOUT_SEC="${PTB_JUDGE_TIMEOUT_SEC:-3000}"

# PostTrainBench agent name (selects the trace parser in src/trace_parsing and
# the harness clause in the api judge). Derived from harbor's agent transcript
# file name (claude-code.txt / codex.txt / opencode.txt / gemini-cli.txt), so
# the same task works for any harbor agent; metadata.json's agent_name (the
# prompt-generation agent) is the fallback.
AGENT_LOGS_DIR="${PTB_AGENT_LOGS_DIR:-/logs/artifacts/agent_logs}"
RAW_TRACE=$(ls -S "$AGENT_LOGS_DIR"/*.txt 2>/dev/null | head -1 || true)
AGENT_NAME=""
if [ -n "$RAW_TRACE" ]; then
    case "$(basename "$RAW_TRACE" .txt)" in
        claude*)   AGENT_NAME="claude" ;;
        codex*)    AGENT_NAME="codex" ;;
        opencode*) AGENT_NAME="opencode" ;;
        gemini*)   AGENT_NAME="gemini" ;;
        cursor*)   AGENT_NAME="cursor" ;;
    esac
fi
if [ -z "$AGENT_NAME" ] && [ -f "$TESTS/metadata.json" ]; then
    AGENT_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json')).get('agent_name','claude'))" 2>/dev/null || echo "claude")
fi
# run_modal_task.sh passes the authoritative values via `harbor run --ve`.
AGENT_NAME="${PTB_AGENT_NAME:-${AGENT_NAME:-claude}}"

# ---- sandbox ------------------------------------------------------------
rm -rf "$JUDGE_HOME"
mkdir -p "$JUDGE_HOME/task"
cp -a "$CODE_DIR/." "$JUDGE_HOME/task/"
rm -rf "$JUDGE_HOME/task/judgement.json" "$JUDGE_HOME/task/final_model"
[ -d "$MODEL_DIR" ] && ln -s "$MODEL_DIR" "$JUDGE_HOME/task/final_model"

# ---- traces -------------------------------------------------------------
# ptb_collect.sh stages harbor's agent logs under /logs/artifacts/agent_logs/.
# The agent's own transcript is <harbor-agent-name>.txt; pick the largest
# non-empty .txt so codex/gemini/opencode agents work too.
if [ -n "$RAW_TRACE" ] && [ -s "$RAW_TRACE" ]; then
    cp "$RAW_TRACE" "$JUDGE_HOME/solve_out.txt"
    cp "$RAW_TRACE" "$LOGS_DIR/solve_out.txt"
    echo "Agent trace: $RAW_TRACE ($(wc -c < "$RAW_TRACE") bytes)"
    # parse_trace.py also writes *_sanitized companions (needs $PTB/.env, empty here)
    if python3 "$TRACE_PARSER" --agent "$AGENT_NAME" "$JUDGE_HOME/solve_out.txt" -o "$JUDGE_HOME/solve_parsed.txt"; then
        cp "$JUDGE_HOME/solve_parsed.txt" "$LOGS_DIR/solve_parsed.txt"
        rm -f "$JUDGE_HOME"/*_sanitized.txt
        echo "Parsed trace: $(wc -l < "$JUDGE_HOME/solve_parsed.txt") lines"
    else
        echo "WARNING: trace parsing failed; judges fall back to the raw trace (../solve_out.txt)"
    fi
else
    echo "WARNING: no agent trace found under $AGENT_LOGS_DIR — judges will run without one"
fi

# The agent harness model (api judge's {agent_harness} clause): the parsed
# claude trace carries a "Model: <name>" line; PTB_AGENT_CONFIG overrides.
AGENT_CONFIG="${PTB_AGENT_CONFIG:-}"
if [ -z "$AGENT_CONFIG" ] && [ -f "$JUDGE_HOME/solve_parsed.txt" ]; then
    AGENT_CONFIG=$(grep -m1 -E '^\s*Model: ' "$JUDGE_HOME/solve_parsed.txt" | sed -E 's/^\s*Model: //' || true)
fi
echo "Judge context: benchmark=$BENCHMARK_ID model=$MODEL_ID agent=$AGENT_NAME agent_config=${AGENT_CONFIG:-<unknown>}"

# ---- judges: condor's own judge phase (src/judges/judge_lib.sh) ------------
# run_all_judges is the loop src/run_task.sh runs — judge set and order,
# best-of-N contamination slots (judgement_multi_runs/), codex flags and
# version pins, output names, _meta tagging — with JUDGE_RUNTIME=local: codex
# runs directly in this image, authenticated by OPENAI_API_KEY, each run
# bounded by JUDGE_TIMEOUT_SEC. Judge tooling, the test set and the model's
# config reach the sandbox via prepare_judge_sandbox, as on condor.
export JUDGE_RUNTIME=local JUDGE_TIMEOUT_SEC
source "$JUDGES_DIR/judge_lib.sh"
prepare_judge_sandbox "$JUDGE_HOME" "$BENCHMARK_ID" "$MODEL_DIR/config.json"
if setup_judge_codex_auth "$JUDGE_HOME"; then
    echo "Judges: ${ALL_JUDGES[*]} (defaults: model=$JUDGE_DEFAULT_MODEL effort=$JUDGE_DEFAULT_REASONING_EFFORT codex=${JUDGE_DEFAULT_CODEX_VERSION:-image}; contamination x$CONTAMINATION_JUDGE_SLOTS; ${JUDGE_TIMEOUT_SEC}s per run)"
    run_all_judges "$JUDGE_HOME" "" "$LOGS_DIR" "$BENCHMARK_ID" "$MODEL_ID" "$AGENT_NAME" "$AGENT_CONFIG" \
        || echo "ERROR: a judge.conf could not be loaded; the remaining judges did not run"
    # parse_trace.py writes *_sanitized companions; with the image's empty .env
    # they are byte-identical copies, so drop them to keep /logs/verifier lean.
    rm -f "$LOGS_DIR"/judge_output_*_sanitized.* "$LOGS_DIR"/judgement_multi_runs/judge_output_*_sanitized.*
else
    echo "WARNING: no OpenAI key in the verifier env — skipping all judges; the exported run will lack verdicts"
fi

# ============================================================
# Final evaluation: condor's own src/eval/run_final_eval.sh, baked in under
# /tests/ptb by the adapter and run with EVAL_RUNTIME=local. Seeds, the
# greedy-model check, the max-tokens retry cascade and the seed average are
# condor's code. Per-seed metrics and attempt logs go to /logs/verifier/
# evaluation/, the seed average to /logs/verifier/metrics.json. Each attempt is
# bounded by PTB_EVAL_ATTEMPT_TIMEOUT_SEC (task.toml), so a hung attempt still
# leaves verifier time for the rest.
# ============================================================
echo ""
echo "=== Running the final evaluation (src/eval/run_final_eval.sh) ==="
EVAL_STATUS=0
( cd "$PTB" && EVAL_RUNTIME=local EVAL_ATTEMPT_TIMEOUT_SEC="${PTB_EVAL_ATTEMPT_TIMEOUT_SEC:-7200}" \
    bash src/eval/run_final_eval.sh "$BENCHMARK_ID" "$MODEL_DIR" "$LOGS_DIR/evaluation" "$LOGS_DIR/metrics.json" ) \
    || EVAL_STATUS=$?
[ "$EVAL_STATUS" = 0 ] || echo "ERROR: the final evaluation failed (exit $EVAL_STATUS); see its output above"

# ============================================================
# Extract accuracy and write reward
# ============================================================
echo ""
echo "=== Evaluation complete ==="

# Reward = metrics.json's `accuracy`, read as strictly as scripts/utils.py
# load_metrics (every evaluate.py writes a float `accuracy`). Without a usable
# accuracy the reward is the base model's zero-shot score, as scripts/collect.py's
# baseline fallback does for broken runs; metrics.json is left as it is, so the
# exported run takes the same fallback there. harbor parses reward.txt with
# float() and rejects non-finite values, so only the number goes to stdout.
[ -f "$LOGS_DIR/metrics.json" ] && { echo "metrics.json contents:"; cat "$LOGS_DIR/metrics.json"; echo; }
REWARD=$(python3 - "$LOGS_DIR/metrics.json" "$TESTS/metadata.json" <<'PY'
import json, math, sys

metrics_path, metadata_path = sys.argv[1:]

def usable(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

try:
    with open(metrics_path) as f:
        accuracy = json.load(f).get("accuracy")
    if usable(accuracy):
        print(accuracy)
        sys.exit(0)
    reason = f"metrics.json has no finite numeric 'accuracy' (got {accuracy!r})"
except FileNotFoundError:
    reason = "metrics.json not written (the first seed failed at every stage, or the evaluation errored)"
except Exception as e:  # malformed JSON, top level not an object, ...
    reason = f"metrics.json unreadable ({type(e).__name__}: {e})"
with open(metadata_path) as f:
    baseline = json.load(f)["baseline_accuracy"]
print(f"ERROR: {reason}; reward = baseline accuracy {baseline}", file=sys.stderr)
print(baseline)
PY
)
if [ -n "$REWARD" ]; then
    echo "Reward: $REWARD"
    echo "$REWARD" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: could not determine a reward (metadata.json has no usable baseline_accuracy?); reward.txt not written"
fi

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"

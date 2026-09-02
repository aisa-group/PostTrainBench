#!/bin/bash
set -e

# PostTrainBench verification script
# Runs the PostTrainBench v1.1 reward-hacking judges (src/judges, codex CLI) and
# the 3-phase evaluation with retry logic.
# Matches the original run_task.sh evaluation pipeline.
#
# Tamper-resistance design (harbor 0.7.0 separate-verifier mode):
#   - This script runs in a SEPARATE container from the agent (see
#     [verifier].environment_mode = "separate" in task.toml). The agent
#     never has shell or filesystem access to this container, so it
#     can't tamper with evaluate.py, templates/, the Python interpreter,
#     installed packages (vllm, inspect_evals, transformers), or this
#     script itself.
#   - All verifier-side files (evaluate.py, templates/, metadata.json,
#     evaluation_code/, bfcl_evaluation_code.py, ptb/ = src/judges +
#     src/trace_parsing + info.json, test_data.json) are
#     BAKED INTO the verifier image at build time (see tests/Dockerfile)
#     and live at /tests/.
#   - The agent's code arrives as a size-filtered snapshot in
#     /logs/artifacts/workspace (staged by ptb_collect.sh, transferred by
#     harbor's conventional artifact dir); the contamination judge reads it
#     there (cd $CODE_DIR && codex exec ...).
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

# Check GPU availability
echo ""
echo "=== GPU Check ==="
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

# Check if final_model exists in agent's workspace
echo ""
echo "=== Checking final_model ==="
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: final_model directory not found at $MODEL_DIR"
    { ls -la "$WORKSPACE"; echo "--- $(dirname "$MODEL_DIR") ---"; ls -la "$(dirname "$MODEL_DIR")"; } > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Check if final_model has required files
echo "Contents of final_model:"
ls -la "$MODEL_DIR" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    echo '{"error": "invalid model - no config.json", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
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
JUDGE_TIMEOUT_SEC="${PTB_JUDGE_TIMEOUT_SEC:-3000}"   # per judge; verifier budget is 5h incl. eval

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

cp "$JUDGES_DIR/judge_tools/contamination_check.py" "$JUDGES_DIR/judge_tools/model_identity_check.py" "$JUDGE_HOME/"
cp -r "$JUDGES_DIR/judge_tools/reference_configs" "$JUDGE_HOME/reference_configs"
[ -f "$TESTS/test_data.json" ] && cp "$TESTS/test_data.json" "$JUDGE_HOME/test_data.json"
[ -f "$MODEL_DIR/config.json" ] && cp "$MODEL_DIR/config.json" "$JUDGE_HOME/final_model_config.json"

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

# ---- codex config (condor: containers/other_home_data/.codex) -------------
export CODEX_HOME="$JUDGE_HOME/.codex"
mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<EOF_CODEX
[projects."$JUDGE_HOME/task"]
trust_level = "trusted"

[shell_environment_policy]
inherit = "all"
EOF_CODEX

# ---- judge set + order: ALL_JUDGES from judge_lib.sh ---------------------
ALL_JUDGES=($(grep -oE '^ALL_JUDGES=\([^)]*\)' "$JUDGES_DIR/judge_lib.sh" | sed -E 's/^ALL_JUDGES=\((.*)\)$/\1/'))
JUDGE_DEFAULT_MODEL=$(grep -oE '^JUDGE_DEFAULT_MODEL="[^"]*"' "$JUDGES_DIR/judge_lib.sh" | cut -d'"' -f2)
JUDGE_DEFAULT_REASONING_EFFORT=$(grep -oE '^JUDGE_DEFAULT_REASONING_EFFORT="[^"]*"' "$JUDGES_DIR/judge_lib.sh" | cut -d'"' -f2)
echo "Judges: ${ALL_JUDGES[*]} (defaults: ${JUDGE_DEFAULT_MODEL:-gpt-5.4} / ${JUDGE_DEFAULT_REASONING_EFFORT:-xhigh})"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${CODEX_API_KEY:-}" ]; then
    echo "WARNING: no OPENAI_API_KEY/CODEX_API_KEY in the verifier env — skipping all judges"
    ALL_JUDGES=()
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-$CODEX_API_KEY}"
export CODEX_API_KEY="${CODEX_API_KEY:-$OPENAI_API_KEY}"

for JUDGE_NAME in "${ALL_JUDGES[@]}"; do
    JUDGE_LABEL=""; JUDGE_OUTPUT_ID=""; JUDGE_PROMPT_FILE=""
    JUDGE_MODEL="${JUDGE_DEFAULT_MODEL:-gpt-5.4}"
    JUDGE_REASONING_EFFORT="${JUDGE_DEFAULT_REASONING_EFFORT:-xhigh}"
    JUDGE_CODEX_VERSION=""
    # judge.conf is plain KEY="value" lines (sourced by judge_lib.sh too)
    source "$JUDGES_DIR/$JUDGE_NAME/judge.conf"
    if [ -z "$JUDGE_LABEL" ] || [ -z "$JUDGE_OUTPUT_ID" ]; then
        echo "WARNING: $JUDGE_NAME/judge.conf incomplete, skipping"; continue
    fi
    echo ""
    echo "--- Judge: $JUDGE_LABEL (model=$JUDGE_MODEL effort=$JUDGE_REASONING_EFFORT codex=${JUDGE_CODEX_VERSION:-image}) ---"

    # Per-judge codex CLI pin (judge_lib.sh installs it into the sandbox home)
    CODEX_BIN="codex"
    if [ -n "$JUDGE_CODEX_VERSION" ]; then
        PIN_PREFIX="$JUDGE_HOME/.codex-cli-$JUDGE_CODEX_VERSION"
        if [ ! -x "$PIN_PREFIX/bin/codex" ]; then
            echo "  installing @openai/codex@$JUDGE_CODEX_VERSION ..."
            npm install -g --prefix "$PIN_PREFIX" --no-fund --no-audit "@openai/codex@$JUDGE_CODEX_VERSION" > "$LOGS_DIR/codex_install_$JUDGE_CODEX_VERSION.log" 2>&1 || true
        fi
        if [ -x "$PIN_PREFIX/bin/codex" ]; then
            CODEX_BIN="$PIN_PREFIX/bin/codex"
        else
            echo "  WARNING: pinned codex install failed, using the image's codex ($(codex --version 2>/dev/null))"
        fi
    fi

    JUDGE_PROMPT=$(python3 "$JUDGES_DIR/get_judge_prompt.py" --judge "$JUDGE_NAME" \
        --benchmark-id "$BENCHMARK_ID" --model "$MODEL_ID" \
        --agent "$AGENT_NAME" ${AGENT_CONFIG:+--agent-config "$AGENT_CONFIG"}) || {
        echo "WARNING: prompt generation failed for $JUDGE_NAME, skipping"; continue; }

    rm -f "$JUDGE_HOME/task/judgement.json"
    # stdin MUST be closed: `codex exec` appends piped stdin to the prompt and
    # reads it to EOF, and under harbor's exec the verifier's stdin is an open
    # pipe that never closes — codex then hangs before its first API call.
    set +e
    (
        cd "$JUDGE_HOME/task" && \
        timeout --signal=TERM --kill-after=60s "$JUDGE_TIMEOUT_SEC" \
        "$CODEX_BIN" --search -a never exec --json \
            -c model_reasoning_summary=detailed \
            -c model_reasoning_effort="$JUDGE_REASONING_EFFORT" \
            --skip-git-repo-check --yolo --model "$JUDGE_MODEL" "$JUDGE_PROMPT" 2>&1 < /dev/null
    ) | tee "$LOGS_DIR/judge_output_$JUDGE_OUTPUT_ID.json" > /dev/null
    JUDGE_EXIT=${PIPESTATUS[0]}
    set -e
    echo "  exit code: $JUDGE_EXIT"

    python3 "$TRACE_PARSER" --agent codex "$LOGS_DIR/judge_output_$JUDGE_OUTPUT_ID.json" \
        -o "$LOGS_DIR/judge_output_$JUDGE_OUTPUT_ID.txt" > /dev/null 2>&1 || true
    # parse_trace.py also writes *_sanitized companions; with the empty .env
    # they are byte-identical copies, so drop them to keep /logs/verifier lean.
    rm -f "$LOGS_DIR"/judge_output_"$JUDGE_OUTPUT_ID"_sanitized.*

    if [ -f "$JUDGE_HOME/task/judgement.json" ]; then
        cp "$JUDGE_HOME/task/judgement.json" "$LOGS_DIR/judgement_$JUDGE_OUTPUT_ID.json"
        echo "  $JUDGE_LABEL judgement: $(cat "$LOGS_DIR/judgement_$JUDGE_OUTPUT_ID.json")"
    else
        echo "  WARNING: judgement.json not created by $JUDGE_LABEL (see judge_output_$JUDGE_OUTPUT_ID.txt); continuing"
    fi
done

# ============================================================
# Evaluation with 3-phase retry logic
# Matches run_task.sh evaluation pipeline.
#
# evaluate.py is run from /tests (untamperable). Some evaluate.py scripts
# (arenahardwriting, healthbench) `from evaluation_code.X import Y`, so
# /tests must be cwd for the import to resolve. final_model lives in
# the agent's workspace (only place it could exist), so --model-path is
# absolute.
# ============================================================
echo ""
echo "=== Running evaluation on final_model ==="

cd "$TESTS"

EVAL_COUNTER=0

kill_gpu_processes() {
    echo "Killing GPU processes..."
    # Kill GPU-holding processes EXCEPT PID 1 (container init / dumb-init).
    # In Docker/Modal, the agent's vLLM process can get reparented to PID 1,
    # which still holds GPU memory when the verifier starts. Killing PID 1
    # would destroy the entire container.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -v '^$' \
        | while read pid; do
            if [ "$pid" -gt 1 ] 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    sleep 5
}

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"

    kill_gpu_processes

    set +e
    python3 "$TESTS/evaluate.py" \
        --model-path "$MODEL_DIR" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        --templates-dir "$TESTS/templates" \
        --limit -1 \
        ${max_tokens_arg} \
        2>&1 | tee "$LOGS_DIR/final_eval_${eval_num}.txt"
    local exit_code=$?
    set -e
    return $exit_code
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "$LOGS_DIR/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        run_evaluation "$max_tokens_arg" "$EVAL_COUNTER"

        if [ -f "$LOGS_DIR/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# Determine token limit args per benchmark for phase 2 and 3
get_phase2_tokens() {
    case "$BENCHMARK_ID" in
        aime2025)    echo "--max-tokens 12000" ;;
        arenahardwriting) echo "--max-new-tokens 12288" ;;
        bfcl)        echo "--max-tokens 12000" ;;
        gpqamain)    echo "--max-tokens 12000" ;;
        gsm8k)       echo "--max-tokens 3000" ;;
        healthbench) echo "--max-new-tokens 12288" ;;
        humaneval)   echo "--max-tokens 3000" ;;
        *)           echo "" ;;
    esac
}

get_phase3_tokens() {
    case "$BENCHMARK_ID" in
        aime2025)    echo "--max-tokens 8000" ;;
        arenahardwriting) echo "--max-new-tokens 8192" ;;
        bfcl)        echo "--max-tokens 8000" ;;
        gpqamain)    echo "--max-tokens 8000" ;;
        gsm8k)       echo "--max-tokens 2000" ;;
        healthbench) echo "--max-new-tokens 8192" ;;
        humaneval)   echo "--max-tokens 2000" ;;
        *)           echo "" ;;
    esac
}

# Phase 1: up to 4 attempts with default tokens
echo ""
echo "--- Phase 1: default token limits (up to 4 attempts) ---"
run_evaluation_with_retry 4 ""

# Phase 2: up to 3 attempts with reduced tokens
PHASE2_TOKENS=$(get_phase2_tokens)
echo ""
echo "--- Phase 2: reduced tokens [${PHASE2_TOKENS}] (up to 3 attempts) ---"
run_evaluation_with_retry 3 "$PHASE2_TOKENS"

# Phase 3: up to 2 attempts with further reduced tokens
PHASE3_TOKENS=$(get_phase3_tokens)
echo ""
echo "--- Phase 3: further reduced tokens [${PHASE3_TOKENS}] (up to 2 attempts) ---"
run_evaluation_with_retry 2 "$PHASE3_TOKENS"

# ============================================================
# Extract accuracy and write reward
# ============================================================
echo ""
echo "=== Evaluation complete (${EVAL_COUNTER} total attempts) ==="

if [ -f "$LOGS_DIR/metrics.json" ]; then
    echo "metrics.json contents:"
    cat "$LOGS_DIR/metrics.json"

    # Try to extract accuracy from the metrics JSON
    PARSE_ERROR_LOG="$LOGS_DIR/metrics_parse_error.txt"
    ACCURACY=$(python3 -c "
import json
try:
    with open('$LOGS_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    # Try common metric names
    for key in ['accuracy', 'pass@1', 'score', 'exact_match']:
        if key in metrics:
            print(metrics[key])
            break
    else:
        # If no known metric, use first numeric value
        for v in metrics.values():
            if isinstance(v, (int, float)):
                print(v)
                break
        else:
            print(0)
except Exception as e:
    print(f'Error parsing metrics: {e}', file=__import__('sys').stderr)
    print(0)
" 2>"$PARSE_ERROR_LOG")

    if [ -s "$PARSE_ERROR_LOG" ]; then
        cat "$PARSE_ERROR_LOG" >&2
    else
        rm -f "$PARSE_ERROR_LOG"
    fi

    echo "Accuracy: $ACCURACY"
    echo "$ACCURACY" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: metrics.json not created after all evaluation attempts"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"

#!/bin/bash
set -e

# PostTrainBench verification script
# Runs contamination judge (codex CLI) and 3-phase evaluation with retry logic.
# Matches the original run_task.sh evaluation pipeline.
#
# Tamper-resistance design (harbor 0.7.0 separate-verifier mode):
#   - This script runs in a SEPARATE container from the agent (see
#     [verifier].environment_mode = "separate" in task.toml). The agent
#     never has shell or filesystem access to this container, so it
#     can't tamper with evaluate.py, templates/, the Python interpreter,
#     installed packages (vllm, inspect_evals, transformers), or this
#     script itself.
#   - All verifier-side files (evaluate.py, templates/, contamination_judge.py,
#     metadata.json, evaluation_code/, bfcl_evaluation_code.py) are
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
# Run contamination judge (codex CLI)
# Matches run_task.sh lines 180-201.
#
# The judge prompt is built by /tests/contamination_judge.py (untamperable).
# Codex still runs with cwd=$WORKSPACE so its read tools naturally see the
# agent's training code. Codex writes contamination_judgement.txt and
# disallowed_model_judgement.txt into cwd; we copy them out to LOGS_DIR.
# (This matches condor's behavior; agent could pre-place these files but
# codex normally overwrites them when it produces a verdict.)
# ============================================================
echo ""
echo "=== Running Contamination Judge ==="

if [ -f "$TESTS/contamination_judge.py" ] && [ -n "$BENCHMARK_NAME" ]; then
    # Generate the judge prompt from the untampered /tests/ copy
    JUDGE_TASK=$(python3 "$TESTS/contamination_judge.py" \
        --model "$MODEL_ID" \
        --benchmark "$BENCHMARK_NAME" 2>/dev/null) || true

    if [ -n "$JUDGE_TASK" ] && [ -n "$CODEX_API_KEY" ]; then
        echo "Running codex CLI contamination judge..."
        # The weights live on the volume, outside the judge's cwd. Expose them
        # at the condor-parity path so the judge can inspect final_model/
        # (e.g. spot a downloaded instruct checkpoint). Read-only mount, so
        # the judge cannot alter what evaluate.py will load.
        if [ "$MODEL_DIR" != "$CODE_DIR/final_model" ] && [ ! -e "$CODE_DIR/final_model" ]; then
            ln -s "$MODEL_DIR" "$CODE_DIR/final_model" && echo "Linked $CODE_DIR/final_model -> $MODEL_DIR for the judge"
        fi
        set +e
        cd "$CODE_DIR"
        codex --search -a never exec --json -c model_reasoning_summary=detailed \
            --skip-git-repo-check --yolo --model "gpt-5.4" "$JUDGE_TASK" \
            2>&1 | tee "$LOGS_DIR/judge_output.json"
        JUDGE_EXIT_CODE=$?
        set -e
        echo "Judge exit code: $JUDGE_EXIT_CODE"

        # Codex writes these files directly into the workspace
        if [ -f "$CODE_DIR/contamination_judgement.txt" ]; then
            cp "$CODE_DIR/contamination_judgement.txt" "$LOGS_DIR/contamination_judgement.txt"
            echo "Contamination judgement: $(cat "$LOGS_DIR/contamination_judgement.txt")"
        else
            echo "no contamination detected (codex did not produce output)" > "$LOGS_DIR/contamination_judgement.txt"
        fi
        if [ -f "$CODE_DIR/disallowed_model_judgement.txt" ]; then
            cp "$CODE_DIR/disallowed_model_judgement.txt" "$LOGS_DIR/disallowed_model_judgement.txt"
            echo "Model judgement: $(cat "$LOGS_DIR/disallowed_model_judgement.txt")"
        else
            echo "only allowed use detected (codex did not produce output)" > "$LOGS_DIR/disallowed_model_judgement.txt"
        fi
    else
        echo "Warning: CODEX_API_KEY not set or prompt generation failed, skipping judge"
        echo "no contamination detected (judge skipped - no API key)" > "$LOGS_DIR/contamination_judgement.txt"
        echo "only allowed use detected (judge skipped - no API key)" > "$LOGS_DIR/disallowed_model_judgement.txt"
    fi
else
    echo "Warning: contamination_judge.py or metadata not found in /tests, skipping judge"
    echo "no contamination detected (judge not available)" > "$LOGS_DIR/contamination_judgement.txt"
    echo "only allowed use detected (judge not available)" > "$LOGS_DIR/disallowed_model_judgement.txt"
fi

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
" 2>&1)

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

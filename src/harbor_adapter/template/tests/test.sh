#!/bin/bash
set -e

# PostTrainBench Harbor verifier.
#
# Runs the judge-v2 two-judge pipeline directly in the separate verifier
# container, then evaluates final_model with the same 3-phase retry strategy as
# src/run_task.sh. The agent can write only to /home/agent/workspace; verifier
# code and benchmark metadata are baked into /tests.

TESTS="/tests"
WORKSPACE="/home/agent/workspace"
JUDGE_ROOT="/home/agent"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Harbor Verifier ==="
echo "Tests dir: $TESTS"
echo "Workspace: $WORKSPACE"
echo "Logs dir: $LOGS_DIR"

echo ""
echo "=== GPU Check ==="
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

echo ""
echo "=== Checking final_model ==="
if [ ! -d "$WORKSPACE/final_model" ]; then
    echo "ERROR: final_model directory not found"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1 || true
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

ls -la "$WORKSPACE/final_model" | tee "$LOGS_DIR/final_model_listing.txt"
if [ ! -f "$WORKSPACE/final_model/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    echo '{"error": "invalid model - no config.json", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi
cp "$WORKSPACE/final_model/config.json" "$JUDGE_ROOT/final_model_config.json"

BENCHMARK_ID=""
BENCHMARK_NAME=""
MODEL_ID=""
MODEL_SHORT_NAME=""

if [ -f "$TESTS/metadata.json" ]; then
    BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])" 2>/dev/null || echo "")
    BENCHMARK_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_name'])" 2>/dev/null || echo "Unknown")
    MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])" 2>/dev/null || echo "Unknown")
    MODEL_SHORT_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_short_name'])" 2>/dev/null || echo "model")
fi
echo "Benchmark ID: $BENCHMARK_ID"
echo "Benchmark Name: $BENCHMARK_NAME"
echo "Model: $MODEL_ID"

find_agent_trace() {
    for candidate in \
        /logs/agent/codex.jsonl \
        /logs/agent/codex.txt \
        /logs/agent/claude-code.txt \
        /logs/agent/gemini.txt \
        /logs/agent/opencode.txt
    do
        if [ -s "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    find /logs/agent -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.txt' \) -size +0 -print | head -1
}

parser_for_trace() {
    case "$(basename "$1")" in
        codex*) echo "codex" ;;
        claude*) echo "claude" ;;
        gemini*) echo "gemini" ;;
        opencode*) echo "opencode" ;;
        *) echo "codex" ;;
    esac
}

prepare_judge_inputs() {
    local trace_source
    trace_source="$(find_agent_trace || true)"
    if [ -n "$trace_source" ]; then
        echo "Using agent trace: $trace_source"
        cp "$trace_source" "$JUDGE_ROOT/solve_out.txt"
    else
        echo "WARNING: no agent trace found under /logs/agent"
        echo "No agent trace was available in /logs/agent." > "$JUDGE_ROOT/solve_out.txt"
    fi

    cat > "$TESTS/.env" <<EOF
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
OPENCODE_API_KEY="${OPENCODE_API_KEY:-}"
CODEX_API_KEY="${CODEX_API_KEY:-}"
EOF

    local parser
    parser="$(parser_for_trace "$trace_source")"
    python3 "$TESTS/src/trace_parsing/parse_trace.py" \
        --agent "$parser" \
        "$JUDGE_ROOT/solve_out.txt" \
        -o "$JUDGE_ROOT/solve_parsed.txt" \
        2>&1 | tee "$LOGS_DIR/parse_trace.txt" || cp "$JUDGE_ROOT/solve_out.txt" "$JUDGE_ROOT/solve_parsed.txt"

    cp "$TESTS/src/disallowed_usage_judge/judge_tools/contamination_check.py" "$JUDGE_ROOT/contamination_check.py"
    cp "$TESTS/src/disallowed_usage_judge/judge_tools/model_identity_check.py" "$JUDGE_ROOT/model_identity_check.py"
    rm -rf "$JUDGE_ROOT/reference_configs"
    cp -r "$TESTS/src/disallowed_usage_judge/judge_tools/reference_configs" "$JUDGE_ROOT/reference_configs"

    local test_data="$TESTS/src/eval/tasks/$BENCHMARK_ID/test_data.json"
    if [ -f "$test_data" ]; then
        cp "$test_data" "$JUDGE_ROOT/test_data.json"
    fi
}

run_codex_judge() {
    local kind="$1"
    local output_json="$2"
    local output_txt="$3"
    local judgement_json="$4"
    local label="$5"
    local prompt_args=(
        --benchmark-id "$BENCHMARK_ID"
        --model "$MODEL_ID"
    )
    if [ "$kind" = "api" ]; then
        prompt_args+=(--kind api --agent "${HARBOR_AGENT:-codex}" --agent-config "${CODEX_MODEL:-unknown}")
    fi

    rm -f "$WORKSPACE/judgement.json" "$judgement_json"
    local judge_prompt
    judge_prompt="$(python3 "$TESTS/src/disallowed_usage_judge/get_judge_prompt.py" "${prompt_args[@]}")"

    echo ""
    echo "=== Running $label ==="
    set +e
    (
        cd "$WORKSPACE"
        codex --search -a never exec --json \
            -c model_reasoning_summary=detailed \
            -c model_reasoning_effort=xhigh \
            --skip-git-repo-check --yolo \
            --model "${CODEX_MODEL:-openai/gpt-5.1-codex}" \
            "$judge_prompt"
    ) 2>&1 | tee "$output_json"
    local judge_exit=${PIPESTATUS[0]}
    set -e
    echo "$label exit code: $judge_exit"

    python3 "$TESTS/src/trace_parsing/parse_trace.py" \
        --agent codex \
        "$output_json" \
        -o "$output_txt" \
        2>&1 | tee -a "$LOGS_DIR/parse_trace.txt" || cp "$output_json" "$output_txt"

    if [ ! -f "$WORKSPACE/judgement.json" ]; then
        echo "ERROR: $label did not create judgement.json" >&2
        return 1
    fi
    cp "$WORKSPACE/judgement.json" "$judgement_json"
    echo "$label judgement: $(cat "$judgement_json")"
}

echo ""
echo "=== Running Judge V2 ==="
export CODEX_API_KEY="${CODEX_API_KEY:-${OPENAI_API_KEY:-}}"
if [ -z "$CODEX_API_KEY" ]; then
    echo "ERROR: CODEX_API_KEY/OPENAI_API_KEY is required for Harbor verifier judges" >&2
    exit 1
fi
prepare_judge_inputs
run_codex_judge \
    data_and_model \
    "$LOGS_DIR/judge_output_gpt5_4.json" \
    "$LOGS_DIR/judge_output_gpt5_4.txt" \
    "$LOGS_DIR/judgement_gpt5_4.json" \
    "contamination judge"
rm -f "$WORKSPACE/judgement.json"
run_codex_judge \
    api \
    "$LOGS_DIR/judge_output_api.json" \
    "$LOGS_DIR/judge_output_api.txt" \
    "$LOGS_DIR/judgement_api.json" \
    "API judge"

echo ""
echo "=== Running evaluation on final_model ==="
cd "$TESTS"

EVAL_COUNTER=0

kill_gpu_processes() {
    echo "Killing GPU processes..."
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
        --model-path "$WORKSPACE/final_model" \
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

get_phase2_tokens() {
    case "$BENCHMARK_ID" in
        aime2025) echo "--max-tokens 12000" ;;
        arenahardwriting) echo "--max-new-tokens 12288" ;;
        bfcl) echo "--max-tokens 12000" ;;
        gpqamain) echo "--max-tokens 12000" ;;
        gsm8k) echo "--max-tokens 3000" ;;
        healthbench) echo "--max-new-tokens 12288" ;;
        humaneval) echo "--max-tokens 3000" ;;
        *) echo "" ;;
    esac
}

get_phase3_tokens() {
    case "$BENCHMARK_ID" in
        aime2025) echo "--max-tokens 8000" ;;
        arenahardwriting) echo "--max-new-tokens 8192" ;;
        bfcl) echo "--max-tokens 8000" ;;
        gpqamain) echo "--max-tokens 8000" ;;
        gsm8k) echo "--max-tokens 2000" ;;
        healthbench) echo "--max-new-tokens 8192" ;;
        humaneval) echo "--max-tokens 2000" ;;
        *) echo "" ;;
    esac
}

run_evaluation_with_retry 4 ""
run_evaluation_with_retry 3 "$(get_phase2_tokens)"
run_evaluation_with_retry 2 "$(get_phase3_tokens)"

echo ""
echo "=== Evaluation complete (${EVAL_COUNTER} total attempts) ==="

if [ -f "$LOGS_DIR/metrics.json" ]; then
    cat "$LOGS_DIR/metrics.json"
    ACCURACY=$(python3 -c "
import json
import sys
try:
    with open('$LOGS_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    for key in ['accuracy', 'pass@1', 'score', 'exact_match']:
        if key in metrics:
            print(metrics[key])
            break
    else:
        for value in metrics.values():
            if isinstance(value, (int, float)):
                print(value)
                break
        else:
            print(0)
except Exception as exc:
    print(f'Error parsing metrics: {exc}', file=sys.stderr)
    print(0)
" 2>&1)
else
    echo "ERROR: metrics.json not created after all evaluation attempts"
    ACCURACY="0"
fi

if python3 -c "import json, sys; data=json.load(open(sys.argv[1])); sys.exit(0 if data.get('contamination') or data.get('disallowed_model') else 1)" "$LOGS_DIR/judgement_gpt5_4.json"; then
    echo "Judge flagged contamination or disallowed model usage; reward set to 0"
    echo "0" > "$LOGS_DIR/reward.txt"
else
    echo "Accuracy: $ACCURACY"
    echo "$ACCURACY" > "$LOGS_DIR/reward.txt"
fi

echo ""
echo "=== Verification complete ==="
ls -la "$LOGS_DIR/"

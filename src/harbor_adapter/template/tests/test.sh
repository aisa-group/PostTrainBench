#!/bin/bash
set -e

# PostTrainBench verification script
# This runs the contamination judge and evaluation on the agent's final_model
# Note: Agent execution time is tracked by Harbor in result.json (agent_execution field)

WORKSPACE="/home/agent/workspace"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Verifier ==="
echo "Workspace: $WORKSPACE"
echo "Logs dir: $LOGS_DIR"

# Check GPU availability
echo ""
echo "=== GPU Check ==="
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

# Check if final_model exists
echo ""
echo "=== Checking final_model ==="
if [ ! -d "$WORKSPACE/final_model" ]; then
    echo "ERROR: final_model directory not found"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Check if final_model has required files
echo "Contents of final_model:"
ls -la "$WORKSPACE/final_model" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$WORKSPACE/final_model/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    echo '{"error": "invalid model - no config.json", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Show model config
echo ""
echo "=== Model config.json ==="
cat "$WORKSPACE/final_model/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Check for tokenizer
echo ""
echo "=== Checking tokenizer files ==="
ls -la "$WORKSPACE/final_model/"*token* 2>/dev/null || echo "No tokenizer files found with 'token' in name"
ls -la "$WORKSPACE/final_model/"*.json 2>/dev/null || echo "No json files found"

# Run contamination judge (before evaluation)
echo ""
echo "=== Running Contamination Judge ==="

# Read metadata for benchmark and model info
if [ -f "$WORKSPACE/metadata.json" ]; then
    BENCHMARK_NAME=$(python3 -c "import json; print(json.load(open('$WORKSPACE/metadata.json'))['benchmark_name'])" 2>/dev/null || echo "Unknown")
    MODEL_ID=$(python3 -c "import json; print(json.load(open('$WORKSPACE/metadata.json'))['model_id'])" 2>/dev/null || echo "Unknown")
    echo "Benchmark: $BENCHMARK_NAME"
    echo "Model: $MODEL_ID"

    if [ -f "$WORKSPACE/contamination_judge.py" ]; then
        set +e
        python3 "$WORKSPACE/contamination_judge.py" \
            --workspace "$WORKSPACE" \
            --model "$MODEL_ID" \
            --benchmark "$BENCHMARK_NAME" \
            --output-dir "$LOGS_DIR" \
            2>&1 | tee "$LOGS_DIR/judge_output.txt"
        JUDGE_EXIT_CODE=$?
        set -e

        echo "Judge exit code: $JUDGE_EXIT_CODE"

        if [ -f "$LOGS_DIR/contamination_judgement.txt" ]; then
            echo "Contamination judgement: $(cat $LOGS_DIR/contamination_judgement.txt)"
        fi
        if [ -f "$LOGS_DIR/disallowed_model_judgement.txt" ]; then
            echo "Model judgement: $(cat $LOGS_DIR/disallowed_model_judgement.txt)"
        fi
    else
        echo "Warning: contamination_judge.py not found, skipping judge"
        echo "no contamination detected (judge not available)" > "$LOGS_DIR/contamination_judgement.txt"
        echo "only allowed use detected (judge not available)" > "$LOGS_DIR/disallowed_model_judgement.txt"
    fi
else
    echo "Warning: metadata.json not found, skipping contamination judge"
    echo "no contamination detected (metadata not available)" > "$LOGS_DIR/contamination_judgement.txt"
    echo "only allowed use detected (metadata not available)" > "$LOGS_DIR/disallowed_model_judgement.txt"
fi

# Run evaluation
echo ""
echo "=== Running evaluation on final_model ==="

cd "$WORKSPACE"

# Run evaluation and capture metrics
set +e
python evaluate.py \
    --model-path final_model \
    --json-output-file "$LOGS_DIR/metrics.json" \
    --templates-dir templates/ \
    2>&1 | tee "$LOGS_DIR/eval_output.txt"
EVAL_EXIT_CODE=$?
set -e

echo ""
echo "=== Evaluation exit code: $EVAL_EXIT_CODE ==="

# Extract the accuracy metric and write as reward
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
    echo "ERROR: metrics.json not created"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"

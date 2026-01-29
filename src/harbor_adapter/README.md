# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations on cloud GPUs.


## Installation

```bash
# use the included pyproject.toml file to get the python environment with harbor and modal 
uv sync
```

## Quick Start

### 1. Generate a task

```bash
cd src/harbor_adapter

# Generate a single task
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./tasks

# Or generate all task combinations
python run_adapter.py --all --output ./tasks
```

### 2. Run with Harbor

```bash
# Set your API keys
python -m modal setup
export ANTHROPIC_API_KEY=<your-key>


# Run on Modal
harbor run \
    --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --env modal
```

## Task Structure

Each generated task follows Harbor's standard format:

```
posttrainbench-gsm8k-qwen3-1.7b/
├── task.toml           # Task configuration (GPU, timeout, etc.)
├── instruction.md      # Instructions for the agent
├── environment/
│   ├── Dockerfile      # Container definition
│   ├── evaluate.py     # Evaluation script
│   └── templates/      # Chat templates for different models
└── tests/
    └── test.sh         # Verification script (runs evaluation)
```

## Configuration

The default configuration is:
- **GPU**: 1x H100
- **Memory**: 64GB
- **Storage**: 100GB
- **Timeout**: 10 hours
- **Internet**: Enabled

You can adjust the timeout with `--num-hours`:

```bash
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --num-hours 5 --output ./tasks
```

## Scoring

The verifier runs the evaluation script on the agent's `final_model` and extracts the accuracy metric as the reward (0-1 scale). Results are stored in:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score

## Parity with Original PostTrainBench

This table tracks feature parity between the Harbor adapter and the original PostTrainBench implementation (`src/run_task.sh`).

| Feature | Original | Harbor Adapter | Notes |
|---------|----------|----------------|-------|
| Agent timeout | Configurable hours | Configurable hours | Parity via `--num-hours` |
| GPU access | H100 via HTCondor | H100/A100 via Modal | Parity |
| Timer script | Created at job start | Created at task generation | **Difference**: See note 1 |
| Evaluation | inspect-ai + vLLM | inspect-ai + vLLM | Parity |
| Contamination judge | Runs after agent | Runs in verifier | Parity - uses OpenAI API (gpt-4o-mini) |
| Agent duration | time_taken.txt | Harbor result.json | Parity - see note 4 |
| HuggingFace cache overlay | fuse-overlayfs | Docker volume | Functionally equivalent |
| task_context/ | Copied if exists | Not implemented | **TODO**: Add if needed |
| .codex directory | Copied for Codex agent | Not implemented | Harbor uses different agent configs |
| Container format | Apptainer .sif | Docker | Functionally equivalent |

### Known Differences

1. **Timer.sh timing**: The original creates `timer.sh` at job start time with the actual creation timestamp. The Harbor adapter creates it at task generation time. This means the timer may show less remaining time than expected if there's a delay between task generation and job start. (TODO)

2. **Container format**: Original uses Apptainer/Singularity `.sif` files, Harbor uses Docker containers. Both support GPU passthrough and are functionally equivalent.

3. **Cache management**: Original uses `fuse-overlayfs` for copy-on-write HuggingFace cache. Harbor relies on Docker volumes which achieve similar isolation. (TODO VERIFY)

4. **Agent duration**: The original writes `time_taken.txt` with the agent execution time. Harbor tracks this automatically in its `result.json` file under `agent_execution.started_at` and `agent_execution.finished_at`. To get agent duration from a Harbor job:
   ```python
   import json
   from datetime import datetime

   result = json.load(open("path/to/result.json"))
   start = datetime.fromisoformat(result["agent_execution"]["started_at"])
   end = datetime.fromisoformat(result["agent_execution"]["finished_at"])
   duration = end - start
   print(f"Agent duration: {duration}")
   ``` 

### Contamination Judge

The contamination judge runs in the verifier before evaluation. It uses OpenAI's API (gpt-4o-mini by default) to analyze the agent's code for:
- **Data contamination**: Using benchmark test data for training
- **Model violations**: Using a different model than the specified base model

To enable the judge, set `OPENAI_API_KEY` in your environment before running Harbor. If no API key is set, the judge is skipped and default "pass" results are written.

Output files:
- `contamination_judgement.txt`: "no contamination detected" or "contamination detected"
- `disallowed_model_judgement.txt`: "only allowed use detected" or "disallowed use detected"
- `judge_analysis.json`: Detailed analysis from the LLM


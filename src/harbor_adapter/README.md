# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations on cloud GPUs.

## Supported Benchmarks

| Benchmark ID | Name | Type | Notes |
|-------------|------|------|-------|
| gsm8k | GSM8K (Grade School Math 8K) | inspect-ai | |
| humaneval | HumanEval | inspect-ai | |
| aime2025 | AIME 2025 | inspect-ai | |
| gpqamain | GPQA | inspect-ai | |
| bfcl | Berkeley Function Calling Leaderboard | inspect-ai | Includes `bfcl_evaluation_code.py` via task_context |
| arenahardwriting | Arena-Hard-v2.0 (Writing) | vLLM + OpenAI judge | Requires `OPENAI_API_KEY` for agent |
| healthbench | HealthBench | vLLM + OpenAI judge | Requires `OPENAI_API_KEY` for agent |

## Supported Models

| Key | HuggingFace Model ID |
|-----|---------------------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

Total: **28 tasks** (7 benchmarks x 4 models).

## Installation

```bash
# Use the included pyproject.toml file to get the python environment with harbor and modal
uv sync
```

## Quick Start

### 1. Generate tasks

```bash
cd src/harbor_adapter

# Generate a single task
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./tasks

# Or generate all 28 task combinations
python run_adapter.py --all --output ./tasks

# List available benchmarks and models
python run_adapter.py --list
```

### 2. Set API keys

```bash
python -m modal setup                # Modal cloud setup
export ANTHROPIC_API_KEY=<your-key>  # For Claude agent
export OPENAI_API_KEY=<your-key>     # For contamination judge (codex CLI) + arenahardwriting/healthbench eval
```

### 3. Run with Harbor

```bash
harbor run \
    --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --env modal
```

## API Key Requirements

| Key | Used By | Required For |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | Agent (Claude) | All benchmarks |
| `OPENAI_API_KEY` | Contamination judge (codex CLI), evaluation judge | All benchmarks (judge), arenahardwriting/healthbench (agent eval) |

- The verifier receives `OPENAI_API_KEY` as both `OPENAI_API_KEY` and `CODEX_API_KEY` (codex CLI reads `CODEX_API_KEY`).
- For arenahardwriting and healthbench, `OPENAI_API_KEY` is also passed to the agent environment since their `evaluate.py` scripts call the OpenAI API for judging.

## Task Structure

Each generated task follows Harbor's standard format:

```
posttrainbench-gsm8k-qwen3-1.7b/
├── task.toml              # Task configuration (GPU, timeout, env vars)
├── instruction.md         # Instructions for the agent
├── environment/
│   ├── Dockerfile         # Container definition (CUDA + vLLM + ML packages)
│   ├── .dockerignore      # Excludes Dockerfile from COPY
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── contamination_judge.py  # Generates judge prompt for codex CLI
│   ├── timer.sh           # Countdown timer (sentinel-file based)
│   ├── metadata.json      # Benchmark/model metadata for verifier
│   ├── templates/         # Chat templates for different models
│   ├── evaluation_code/   # (arenahardwriting, healthbench only)
│   └── bfcl_evaluation_code.py  # (bfcl only, from task_context)
└── tests/
    └── test.sh            # Verifier: contamination judge + 3-phase eval retry
```

## Evaluation Retry Logic

The verifier (`test.sh`) uses a 3-phase evaluation retry strategy matching `run_task.sh`:

| Phase | Max Attempts | Token Limits |
|-------|-------------|-------------|
| 1 | 4 | Default |
| 2 | 3 | Reduced (see below) |
| 3 | 2 | Further reduced (see below) |

Token limits per benchmark:

| Benchmark | Phase 2 | Phase 3 |
|-----------|---------|---------|
| aime2025 | `--max-tokens 12000` | `--max-tokens 8000` |
| arenahardwriting | `--max-new-tokens 12288` | `--max-new-tokens 8192` |
| bfcl | `--max-tokens 12000` | `--max-tokens 8000` |
| gpqamain | `--max-tokens 12000` | `--max-tokens 8000` |
| gsm8k | `--max-tokens 3000` | `--max-tokens 2000` |
| healthbench | `--max-new-tokens 12288` | `--max-new-tokens 8192` |
| humaneval | `--max-tokens 3000` | `--max-tokens 2000` |

GPU processes are killed between attempts to free VRAM.

## Contamination Judge

The contamination judge uses OpenAI's Codex CLI to analyze the agent's code:

```bash
codex --search -a never exec --json -c model_reasoning_summary=detailed \
    --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_PROMPT"
```

It checks for:
- **Data contamination**: Using benchmark test data for training
- **Model violations**: Using a different model than the specified base model

Codex reads the workspace code and writes `contamination_judgement.txt` and `disallowed_model_judgement.txt` directly. The judge prompt is synced with `src/disallowed_usage_judge/prompt.txt`.

## Timer

The timer uses a sentinel-file approach: on the first `bash timer.sh` call, the current timestamp is recorded in `.timer_start`. This ensures the countdown is accurate even if the task is generated long before the agent starts.

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| GPU | 1x H100 | Configured in task.toml |
| Memory | 64 GB | |
| Storage | 100 GB | |
| Agent timeout | 10 hours | Adjustable via `--num-hours` |
| Verifier timeout | 3 hours | Accommodates 3-phase retry |
| Internet | Enabled | |

## Scoring

The verifier extracts the accuracy metric from `metrics.json` as the reward (0-1 scale). Results are stored in:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score
- `/logs/verifier/contamination_judgement.txt` - Data contamination verdict
- `/logs/verifier/disallowed_model_judgement.txt` - Model usage verdict

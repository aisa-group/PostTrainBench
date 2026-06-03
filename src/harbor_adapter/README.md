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
| arenahardwriting | Arena-Hard-v2.0 (Writing) | vLLM + OpenAI-compatible judge | Requires `OPENAI_API_KEY` for agent eval |
| healthbench | HealthBench | vLLM + OpenAI-compatible judge | Requires `OPENAI_API_KEY` for agent eval |

## Supported Models

| Key | HuggingFace Model ID |
|-----|---------------------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

Total: **28 tasks** (7 benchmarks x 4 models).

## Installation

The adapter itself only uses the Python standard library. Install Harbor and
its backend dependencies separately in the environment where you will run the
generated tasks.

## Quick Start

### 1. Generate tasks

```bash
# Generate a single task
uv run python3 src/harbor_adapter/run_adapter.py \
    --benchmark gsm8k \
    --model qwen3-1.7b \
    --output ./tasks

# Or generate all 28 task combinations
uv run python3 src/harbor_adapter/run_adapter.py --all --output ./tasks

# List available benchmarks and models
uv run python3 src/harbor_adapter/run_adapter.py --list
```

### 2. Set API keys

```bash
modal setup                          # Modal cloud setup
export ANTHROPIC_API_KEY=<your-key>  # For Claude agent
export OPENAI_API_KEY=<your-key>     # OpenAI-compatible key for Codex judge + selected evals
export OPENAI_BASE_URL=https://api.pinference.ai/api/v1
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
| `OPENAI_API_KEY` | Codex judge, evaluation judge | All benchmarks (judge), arenahardwriting/healthbench (agent eval) |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | Defaults to Pinference in generated tasks |
| `CODEX_MODEL` | Codex judge model | Defaults to `openai/gpt-5.1-codex` |

- The verifier receives `OPENAI_API_KEY` as both `OPENAI_API_KEY` and `CODEX_API_KEY` (codex CLI reads `CODEX_API_KEY`).
- For arenahardwriting and healthbench, `OPENAI_API_KEY` and `OPENAI_BASE_URL` are also passed to the agent environment since their `evaluate.py` scripts call an OpenAI-compatible API for judging.

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
│   ├── timer.sh           # Countdown timer backed by /timer_start
│   ├── metadata.json      # Benchmark/model metadata for verifier
│   ├── templates/         # Chat templates for different models
│   ├── evaluation_code/   # (arenahardwriting, healthbench only)
│   └── bfcl_evaluation_code.py  # (bfcl only, from task_context)
└── tests/
    ├── test.sh            # Verifier: judge-v2 + 3-phase eval retry
    ├── src/
    │   ├── disallowed_usage_judge/  # Judge-v2 prompts and tools
    │   ├── trace_parsing/           # Agent/Codex trace parsers
    │   └── eval/tasks/<benchmark>/  # Judge metadata and optional test data
    └── ...                         # Untampered verifier copy of eval files
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

## Judge V2

The verifier runs the same judge-v2 flow as `src/disallowed_usage_judge/run_judge.sh`,
but directly inside the Harbor verifier container:

```bash
codex --search -a never exec --json \
    -c model_reasoning_summary=detailed \
    -c model_reasoning_effort=xhigh \
    --skip-git-repo-check --yolo \
    --model "${CODEX_MODEL:-openai/gpt-5.1-codex}" \
    "$JUDGE_PROMPT"
```

It runs two judges:

- The canonical contamination/base-model judge writes `/logs/verifier/judgement_gpt5_4.json`.
- The API usage judge writes `/logs/verifier/judgement_api.json` for audit/postmortem use.

The reward is zeroed only when `judgement_gpt5_4.json` flags contamination or
disallowed base-model usage.

## Timer

The timer reads `/timer_start`, which the Harbor healthcheck writes immediately
before the agent launches. That keeps the countdown tied to the actual run
window rather than image build or container boot time.

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
- `/logs/verifier/judgement_gpt5_4.json` - Canonical contamination/base-model verdict
- `/logs/verifier/judgement_api.json` - API usage verdict
- `/logs/verifier/judge_output_gpt5_4.txt` - Parsed canonical judge transcript
- `/logs/verifier/judge_output_api.txt` - Parsed API judge transcript

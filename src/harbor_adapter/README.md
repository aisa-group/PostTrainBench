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

### 2. Set up Modal and credentials

```bash
uv tool install 'harbor[modal]'      # the `harbor` CLI (this repo's .venv only holds modal for volume commands)
harbor_py="$(dirname "$(readlink -f "$(command -v harbor)")")/python"
$harbor_py -m modal setup             # Modal cloud login

export OPENAI_API_KEY=<your-key>      # contamination judge (codex CLI) + arenahardwriting/healthbench eval
export ANTHROPIC_API_KEY=<your-key>   # Claude agent via API ...
# ... or a Claude Max subscription instead of the API key:
export CLAUDE_CODE_OAUTH_TOKEN="$(cat ../../agents/claude_non_api/oauth_token)"   # from `claude setup-token`
export CLAUDE_FORCE_OAUTH=1           # harbor drops ANTHROPIC_API_KEY from the sandbox so the CLI uses the token
```

Behind an HTTP proxy (`https_proxy` set), Modal's client additionally needs
`python-socks` in the *same* environment as `harbor`, otherwise every call
fails with "Could not connect to the Modal server":

```bash
uv pip install --python "$harbor_py" python-socks
```

### 3. Run with Harbor

The trained model is handed from the agent sandbox to the separate verifier
sandbox through a **shared Modal volume** (see [Model hand-off](#model-hand-off-shared-modal-volume)).
Harbor does not create or delete volumes, so use the wrapper, which does:

```bash
bash run_modal_task.sh \
    --task ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code --model anthropic/claude-opus-4-8 \
    --job-name run1
```

It creates the volume `ptb-run1-gsm8k-qwen3-1.7b`, runs

```bash
harbor run --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code --model anthropic/claude-opus-4-8 \
    --env modal --ek 'volumes={"/mnt/ptb_final_model":"ptb-run1-gsm8k-qwen3-1.7b"}' \
    -n 1 --job-name run1
```

and leaves the volume in place: it *is* the trained model
(`modal volume get <volume> / ./final_model`), delete it with
`modal volume delete <volume> --yes` or `--delete-volume`. One volume per task
run — loop over tasks for a full sweep (28 tasks = 28 `harbor run`s).

Useful extra flags (pass after `--`): `--ak version=2.1.251` installs a
specific Claude Code CLI in the sandbox instead of the image's pinned one;
`--agent-timeout-multiplier 0.1` for short debugging runs.

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
├── task.toml              # Task configuration (GPU, timeout, env vars, volume hand-off hook)
├── instruction.md         # Instructions for the agent
├── environment/
│   ├── Dockerfile         # Container definition (CUDA + vLLM + ML packages)
│   ├── .dockerignore      # Excludes Dockerfile from COPY
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── contamination_judge.py  # Generates judge prompt for codex CLI
│   ├── timer.sh           # Countdown timer (healthcheck-written start time)
│   ├── ptb_collect.sh     # Post-agent hook: weights -> volume, code snapshot -> /logs/artifacts
│   ├── metadata.json      # Benchmark/model metadata for verifier
│   ├── templates/         # Chat templates for different models
│   ├── evaluation_code/   # (arenahardwriting, healthbench only)
│   └── bfcl_evaluation_code.py  # (bfcl only, from task_context)
└── tests/
    └── test.sh            # Verifier: contamination judge + 3-phase eval retry (reads $PTB_MODEL_DIR)
```

## Model Hand-off: Shared Modal Volume

The verifier runs in a **separate sandbox** (`[verifier] environment_mode = "separate"`)
so the agent cannot tamper with `evaluate.py`, the judge, or installed packages.
That means the trained weights must cross sandboxes. Harbor's built-in artifact
transfer cannot do this on Modal: it tars the workspace on the agent sandbox and
downloads the tar through Modal's filesystem API, which has a **hard 5 GiB
per-file limit** (`SandboxFilesystemFileTooLargeError`). Every PostTrainBench
model except Qwen3-1.7B in bf16 exceeds it.

Instead (all in `template/task.toml`):

1. A Modal volume, named at launch via `--ek 'volumes={"/mnt/ptb_final_model":"<name>"}'`,
   is mounted at `/mnt/ptb_final_model` in **both** sandboxes (harbor passes the
   same environment kwargs to the separate verifier env).
2. The agent works in `/home/agent/workspace` exactly as on condor and writes
   `final_model/` there — nothing about the volume is visible in its instructions.
3. After the agent exits and before its container is stopped, a
   `[[verifier.collect]] service = "main"` hook runs `ptb_collect.sh`, which
   copies `final_model/.` onto the volume (~7 GB in under 90 s; Modal commits
   the writes in the background, the verifier sees them without an explicit
   commit) and stages a **size-filtered code snapshot** of the workspace
   (≤ 512 MiB per file, ≤ 2 GiB total, smallest-first so code always fits; weight formats and caches skipped) into
   `/logs/artifacts/workspace`.
4. Harbor always transfers `/logs/artifacts` to the host
   (`<trial>/artifacts/logs/artifacts/workspace/`) and into the verifier, where
   the contamination judge reads the code (`CODE_DIR` in `tests/test.sh`, with
   `final_model` symlinked to the volume so the judge can inspect the weights'
   config). There is deliberately **no `[[artifacts]]` entry for the whole
   workspace**: agents leave arbitrary multi-GB dirs behind (a leftover
   `final_model2/` broke the first 1 h run), and one such dir blows harbor's
   120 s tar timeout and the 5 GiB cap — exclude patterns cannot keep up.
5. `[verifier.env] PTB_MODEL_DIR = "/mnt/ptb_final_model"`; `tests/test.sh`
   evaluates `MODEL_DIR="${PTB_MODEL_DIR:-$WORKSPACE/final_model}"`.

Do **not** mount the volume inside the workspace (e.g. at
`/home/agent/workspace/final_model`): Modal exposes nested mounts as symlinks,
harbor's pre-upload cleanup in the verifier deletes that symlink, and an agent
doing `rm -rf final_model && cp -r ckpt final_model` would silently write
off-volume.


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
    --skip-git-repo-check --yolo --model "gpt-5.4" "$JUDGE_PROMPT"
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

The trained model itself stays on the run's Modal volume (`modal volume get <volume> / ./final_model`); the host-side `artifacts/logs/artifacts/workspace/` holds the agent's code snapshot (plus `.ptb_workspace_sizes.txt`, what was left in the workspace).

## Known Gotchas

- **Claude Code CLI version**: the image pins `@anthropic-ai/claude-code@2.1.251`.
  Older pins (2.1.76, the condor image) are rejected by the API for
  `claude-opus-4-8` and newer (`"thinking.type.enabled" is not supported`).
  Override per run with `--ak version=<x.y.z>` (harbor installs it at agent setup).
- **Judge model**: `gpt-5.1-codex` / `gpt-5.2-codex` are gone from the OpenAI
  Responses API (still listed under `/v1/models`); the judge uses `gpt-5.4`.
  When the codex CLI fails, `test.sh` records `no contamination detected (codex did not produce output)`.
- **GPU type**: Modal may hand out an H200 despite `gpu_types = ["H100"]`.
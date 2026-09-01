# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations on cloud GPUs.

## Supported Benchmarks

Benchmarks are discovered from `src/eval/tasks/*/info.json` (`python run_adapter.py --list`).
As of PostTrainBench v1.1: aime2025, arenahardwriting, bfcl, gpqamain, gsm8k, healthbench,
humaneval. `aime2026` is skipped (upstream ships no test-data downloader for it, so it
cannot run under `run_task.sh` either).

| Benchmark ID | Type | Notes |
|-------------|------|-------|
| aime2025, gpqamain, gsm8k, humaneval, bfcl | inspect-ai | bfcl includes `bfcl_evaluation_code.py` via task_context |
| arenahardwriting, healthbench | vLLM + OpenAI judge | `info.json` declares `required_api_keys: ["OPENAI_API_KEY"]`, which the adapter turns into `[environment.env]` (harbor's per-task agent-sandbox env) |

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

The adapter reads everything benchmark-specific from the PostTrainBench tree —
`info.json`, `benchmark.txt`, `evaluate.py`, and the **agent prompt is rendered by
`src/eval/general/get_prompt.py` itself**, so the Harbor instruction is byte-for-byte
the condor prompt (v1.1 rules incl. the decontamination tool section).

Prerequisite (same as `run_task.sh`): every benchmark needs its gitignored
`src/eval/tasks/<id>/test_data.json` (the agent gets it as `../test_data.json` next to
`../contamination_check.py`; the judges use it too):

```bash
# from the repo root; gpqamain needs MY_HF_TOKEN (gated dataset)
uv run --no-project --with datasets --with huggingface_hub --with pyarrow \
    python src/judges/test_data_download/download_test_data.py
```

```bash
cd src/harbor_adapter

# Generate a single task
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./tasks

# Or generate all task combinations (7 benchmarks x 4 models)
python run_adapter.py --all --output ./tasks

# The PostTrainBench agent name only affects agent-specific prompt clauses
# (default 'claude', matching harbor's claude-code agent)
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --agent-name claude --output ./tasks

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
├── instruction.md         # Agent prompt, rendered by src/eval/general/get_prompt.py
├── environment/
│   ├── Dockerfile         # Container definition (CUDA + vLLM + ML packages)
│   ├── .dockerignore      # Excludes Dockerfile from COPY
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── contamination_check.py, test_data.json  # -> /home/agent/ (agent self-decontamination)
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

## Reward-Hacking Judges (PostTrainBench v1.1)

The verifier runs the four judges from `src/judges/` (`ALL_JUDGES` in `judge_lib.sh`
decides set and order) exactly as `src/run_task.sh` does, via the codex CLI:

| Judge | Output id | Verdict | Model / CLI |
|---|---|---|---|
| `data_contamination_judge` | `gpt5_4` | `contamination`, `disallowed_model` | gpt-5.4, xhigh |
| `api_usage_judge` | `api` | `disallowed_api_usage` | gpt-5.4, xhigh |
| `ptb_lookup_judge` | `ptb_lookup` | `disallowed_ptb_lookup` | gpt-5.4, xhigh |
| `general_judge` | `general` | `general_anomaly` | gpt-5.6-terra on codex 0.144.5 (installed at judge time) |

The adapter bakes `src/judges/` (prompts, `judge.conf`s, `get_judge_prompt.py`,
`judge_tools/` with the n-gram checker, model-identity check and reference configs),
`src/trace_parsing/` and the benchmark's `info.json` into the verifier image under
`/tests/ptb/`, so upstream judge changes are picked up on regeneration. Each judge gets
the condor sandbox layout: the agent's code snapshot as its task dir (with `final_model`
symlinked to the volume), `../solve_out.txt` / `../solve_parsed.txt` (the Harbor agent
transcript, staged by `ptb_collect.sh` and parsed by `parse_trace.py`), `../test_data.json`,
the checker tools and `../final_model_config.json`.

Differences from condor: judges authenticate with `OPENAI_API_KEY` (condor bind-mounts a
ChatGPT-subscription `auth.json`); each judge has a `PTB_JUDGE_TIMEOUT_SEC` (default
3000 s) so four judges fit the 5 h verifier budget; the agent harness model for the
api judge is read from the parsed trace (`Model:` line) or `PTB_AGENT_CONFIG`.

A judge that produces no verdict is a warning, not a failure (the run is still
evaluated); judges can be re-run on an exported result dir with `src/judges/run_judges.sh`.

### What the judges see: condor vs Harbor

Both pipelines give the judges the same prompt, tools, trace layout and `final_model` config,
but the **task-directory copy** differs:

| | condor (`run_task.sh`) | Harbor (`ptb_collect.sh`) |
|---|---|---|
| Source | the entire `task/` dir copied to the result dir after `containers/delete_hf_models.py` removed every directory that looks like a HF model (has `*.safetensors`, or ≥2 of `config.json` / `pytorch_model.bin` / `tokenizer_config.json`) | a size-budgeted snapshot of the workspace |
| Size limits | none beyond the model-dir deletion — datasets, checkpoint trees without HF markers (e.g. optimizer states), eval logs, everything else is kept | ≤ 512 MiB per file, ≤ 2 GiB total, files taken smallest-first; `*.safetensors/.bin/.pt/.pth/.ckpt/.gguf/.npy/.npz/.h5/.msgpack/.onnx` never taken; `.git`, `__pycache__`, `.cache`, `.huggingface`, `wandb`, `.venv`, `venv`, `node_modules`, `.ipynb_checkpoints` pruned |
| `final_model` | deleted from `task/` (it is a HF model dir); the judge gets `../final_model_config.json` | excluded from the snapshot; symlinked into the judge's task dir from the read-only volume, so the judge can additionally list/inspect the weights |
| Leftover model dirs (`final_model2/`, checkpoints) | deleted entirely | their small files (configs, tokenizer JSON) survive, the weights do not |
| Record of what was dropped | `output.log` lists the deleted model dirs | `.ptb_workspace_sizes.txt` in the snapshot lists top-level sizes at collection time |

In practice the judges read source files, JSONL/data files and logs, which both variants keep;
the Harbor budget only bites on a workspace holding > 2 GiB of non-weight data, where the
largest files are dropped first. Rationale for the budget: Modal's file download caps single
files at 5 GiB and harbor's transfer has a fixed 120 s gzip timeout, so an unbounded copy
would fail exactly on the runs where it matters most.

## Timer

The timer uses a sentinel-file approach: on the first `bash timer.sh` call, the current timestamp is recorded in `.timer_start`. This ensures the countdown is accurate even if the task is generated long before the agent starts.

## Configuration & Resource Parity

`task.toml` requests the same resources as `src/commit_utils/single_task.sub`:

| Resource | condor (`single_task.sub`) | Harbor (`task.toml`, agent and verifier env) | On Modal |
|---|---|---|---|
| GPU | 1x `NVIDIA H100 80GB HBM3` | `gpus = 1`, `gpu_types = ["H100"]` | H100 80GB (Modal may substitute an H200 — see gotchas) |
| CPUs | `request_cpus = 16` | `cpus = 16` | honoured (`nproc` = 16) |
| RAM | `request_memory = 131072` (128 GB) | `memory_mb = 131072` | passed to Modal as the sandbox memory request/limit (not visible from inside gVisor) |
| Disk | `request_disk = 400G` | `storage_mb = 409600` | **not applied** — Modal Sandboxes take no ephemeral-disk request; the root filesystem is host-backed and effectively unbounded |
| Agent budget | `num_hours` (timeout `+5 min`) | `[agent] timeout_sec = num_hours * 3600` | — |
| Verifier | same node, no explicit limit | `[verifier] timeout_sec = 18000` (5 h) | — |
| Internet | unrestricted | `allow_internet = true` | — |

Other settings: the healthcheck writes `/timer_start` right before the agent launches; the
verifier runs in a separate sandbox built from `tests/`.

## Scoring

The verifier extracts the accuracy metric from `metrics.json` as the reward (0-1 scale). This is the **pre-fallback** score: applying the baseline fallback for judge-flagged runs is done at aggregation time (condor's `scripts/collect.py`), not in the verifier. Results are stored in:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score
- `/logs/verifier/judgement_<id>.json` - per-judge verdicts (`gpt5_4`, `api`, `ptb_lookup`, `general`)
- `/logs/verifier/judge_output_<id>.{json,txt}` - raw and parsed judge traces
- `/logs/verifier/solve_out.txt`, `solve_parsed.txt` - the agent transcript the judges saw

The trained model itself stays on the run's Modal volume (`modal volume get <volume> / ./final_model`); the host-side `artifacts/logs/artifacts/workspace/` holds the agent's code snapshot (plus `.ptb_workspace_sizes.txt`, what was left in the workspace).

## Agent Launch Parity (claude-code)

`run_modal_task.sh` reproduces what PostTrainBench v1.1's `agents/claude*/solve.sh` set:

| condor (`solve.sh`) | Harbor (`run_modal_task.sh`) |
|---|---|
| `CLAUDE_CODE_EFFORT_LEVEL=high` | `--ak reasoning_effort=high` (default; `--effort <level>` / `--effort none`) |
| `BASH_MAX_TIMEOUT_MS=36000000` | `--ae BASH_MAX_TIMEOUT_MS=36000000` (always) |
| `update_agent_cli.sh`: CLI upgraded to `@latest` at run start, `cli_version.txt` | image pin by default (`opus_5.def` era, 2.1.219); `--cli-version latest` resolves the current release via `npm view` and passes `--ak version=`; `--cli-version x.y.z` pins explicitly. The version that ran is in `result.json` `agent_info` (exported as `cli_version.txt`) |
| prompt via stdin (`printf '%s' "$PROMPT" \| claude --print …`) | same (harbor's agent) |
| `--thinking-display summarized` | **not replicated** — see below |

**Known difference — `--thinking-display summarized`.** Harbor's built-in claude-code agent has a
fixed command line and no kwarg for this flag. It matters for the judges: without it the
stream-json trace contains `thinking` blocks with **empty** text (verified: opus-4-8 at high
effort emits non-empty thinking only with the CLI flag; a `thinkingDisplay` key in
`--settings` does not help). The condor v1.1 traces therefore carry the agent's summarised
reasoning and the Harbor traces do not. Parity needs a ~20-line custom agent
(`harbor run --agent <module>:<Class>` subclassing harbor's `ClaudeCode` to append the flag);
tracked as a follow-up.

## Other Agents (codex, opencode, gemini)

The task is agent-agnostic: the verifier picks the trace parser and the judges' harness
clause from harbor's transcript file name (`claude-code.txt`, `codex.txt`, `opencode.txt`,
`gemini-cli.txt`), and `ptb_collect.sh` ships whichever transcript exists. Generate the
task with the matching PostTrainBench agent name so agent-specific prompt clauses match
(`run_adapter.py --agent-name codex`). What harbor's built-in agents do vs the PTB `solve.sh`:

| PTB agent | Harbor agent | Launch parity | Auth | Status |
|---|---|---|---|---|
| `codex`, `codex_non_api[_high/_xhigh]` | `codex` | harbor: `codex exec --json --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check`; wrapper adds `-c model_reasoning_effort` (`--effort`, default high; plain `codex`/`codex_non_api` = `--effort medium`), `model_reasoning_summary=detailed`, `web_search=live` (= `--search`) | `OPENAI_API_KEY`, or subscription `--codex-auth-json agents/codex_non_api/auth.json` (harbor's `CODEX_AUTH_JSON_PATH`) | wired, **not yet smoke-tested** |
| `codex_*_reprompt` | — | PTB's resume-and-reprompt loop has no harbor equivalent | | not supported |
| `opencode` | `opencode` | harbor: `opencode --model=<provider/model> run --format=json --thinking --dangerously-skip-permissions`; writes `opencode.json` with the provider from the model name | provider key inferred from the model name (`anthropic/…`, `openai/…`); PTB's `opencode/…`, `zai/…` providers use `OPENCODE_API_KEY` / `ZAI_API_KEY` via `opencode.json` — needs `--ak opencode_config=…` | **untested** |
| `gemini` | `gemini-cli` | harbor: `gemini --yolo --model=… --prompt=…` (no `--output-format stream-json`; PTB's `gemini_parser` expects stream-json) | `GEMINI_API_KEY` | **untested**; trace parsing likely degrades |
| `cursor_cli`, `grok_cli`, `kimi_claude`, `glmx` | — | | | not supported |

## Exporting to the PostTrainBench Results Layout

Harbor's reward is the pre-fallback accuracy. Baseline fallback for judge-flagged runs,
aggregation, flagged-run review and judge reruns all live in the condor-side tooling, so
export Harbor trials into the same results layout and use those tools unchanged:

```bash
python harbor_to_results.py jobs/gsm8k-1h-2                       # one job (all trials)
python harbor_to_results.py jobs/* --experiment-name _harbor      # everything, suffixed method dirs
python harbor_to_results.py jobs/gsm8k-1h-2 --with-model          # also fetch final_model from the volume

# then, from the repo root:
python scripts/collect.py --data-dir $POST_TRAIN_BENCH_RESULTS_DIR   # final_<method>.csv with baseline fallback
python scripts/find_flagged_runs.py
bash src/judges/run_judges.sh <results>/<method>/<run>               # re-judge a run
```

Layout: `<results>/<agent>_<agent_model>_<N>h[<experiment>]/<benchmark>_<Org>_<Model>_<run_id>/`
(e.g. `claude-code_anthropic_claude-opus-4-8_1h/gsm8k_Qwen_Qwen3-1.7B-Base_1788169511`; `run_id` is
the trial start time in Unix seconds). Each run dir carries `metrics.json`, the four
`judgement_<id>.json` verdicts and `judge_output_<id>.{json,txt}`, `solve_out.txt` /
`solve_parsed.txt` (re-parsed on the host so `*_sanitized` companions redact the keys in your
`.env`), `prompt.txt`, `time_taken.txt`, `cli_version.txt`, `final_eval_<n>.txt`,
`system_monitor.log`, `output.log`, `error.log`, `task/` (code snapshot) and `harbor/`
(`result.json` + `config.json`). Requires the task dir the trial was generated from
(for `prompt.txt` and metadata); otherwise it falls back to parsing the task name.

## Known Gotchas

- **Container era**: the images mirror `containers/opus_5.def` (PostTrainBench v1.1): Claude Code 2.1.219, codex 0.144.0, gemini-cli 0.18.4, opencode 1.17.18; the Grok/Cursor CLIs from that def are not installed.
- **Claude Code CLI version**: the image pins `@anthropic-ai/claude-code@2.1.219`.
  Older pins (2.1.76, the condor image) are rejected by the API for
  `claude-opus-4-8` and newer (`"thinking.type.enabled" is not supported`).
  Override per run with `--ak version=<x.y.z>` (harbor installs it at agent setup).
- **Judge models** come from `src/judges/*/judge.conf` (gpt-5.4; gpt-5.6-terra for the general judge). `gpt-5.1-codex`/`gpt-5.2-codex` no longer exist on the Responses API.
- **GPU type**: Modal may hand out an H200 despite `gpu_types = ["H100"]`.
- **codex must not inherit stdin in the verifier**: `codex exec` appends piped stdin to the
  prompt and reads it to EOF; under harbor's exec the verifier's stdin is a pipe that never
  closes, so codex hangs before its first API call (exit 124 after the judge timeout).
  `test.sh` runs every judge with `< /dev/null`. Condor is unaffected (apptainer closes stdin).
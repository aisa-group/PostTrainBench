# AGENTS.md

Guidelines for AI agents working on the PostTrainBench codebase.

## Project Overview

PostTrainBench is a benchmark framework that measures AI agents' ability to perform **LLM post-training** - improving pre-trained small language models on specific benchmarks through automated research. Agents are typically given 10 hours on an H100 GPU to improve base models, with performance measured by benchmark scores.

## Directory Structure

```
PostTrainBench/
├── agents/              # Agent implementations (claude, codex, gemini, opencode, qwen3max, glm5, ...)
├── assets/              # Static assets (e.g. images for the README)
├── cluster/             # Cluster-specific helper artifacts
├── containers/          # Apptainer/Singularity container definitions and helpers
├── dev_utils/           # Development/debugging utilities (run failure triage, trace extraction, ...)
├── scripts/             # Result aggregation and analysis
├── src/
│   ├── baselines/             # Baseline score computation
│   ├── commit_utils/          # HTCondor job submission utilities (incl. set_env_vars.sh)
│   ├── disallowed_usage_judge/ # Contamination + disallowed-API judges (2 judges; see Safety)
│   ├── eval/
│   │   ├── general/           # Prompt generation (get_prompt.py, prompt.txt)
│   │   ├── tasks/             # Evaluation benchmarks (aime2025, aime2026, gsm8k, ...)
│   │   └── templates/         # Chat templates (Jinja2)
│   ├── trace_parsing/         # Per-agent trace parsers (claude/codex/gemini/opencode)
│   ├── utils/                 # Utility scripts (check_cuda, system_monitor, timestamp_lines, ...)
│   └── run_task.sh            # Main task execution orchestrator
└── results/             # Evaluation results storage (path controlled by POST_TRAIN_BENCH_RESULTS_DIR)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/run_task.sh` | Main task execution orchestrator (runs agent, then 2 judges, then evaluation) |
| `src/commit_utils/commit.sh` | Batch job submission across agents × benchmarks × models |
| `src/commit_utils/set_env_vars.sh` | Sources `.env` and exports `POST_TRAIN_BENCH_*` env vars |
| `src/commit_utils/single_task.sub` | HTCondor submission template |
| `src/commit_utils/single_task_gemini.sub` | Gemini-specific HTCondor submission template |
| `src/eval/general/get_prompt.py` | Generates agent prompts |
| `src/eval/general/prompt.txt` | Agent prompt template |
| `src/trace_parsing/parse_trace.py` | Dispatches to per-agent parser to produce human-readable trace |
| `src/disallowed_usage_judge/run_judge.sh` | Runs both judges (GPT-5.4 contamination, GPT-5.4 API); each writes its own per-judge JSON |
| `src/disallowed_usage_judge/get_judge_prompt.py` | Generates judge prompts (`--kind api` for API judge) |
| `containers/standard.def` | Main container definition (other `.def` files exist per-agent) |
| `scripts/constants.py` | Agent/benchmark mappings |
| `example.env` | Template for the `.env` file (API keys + `POST_TRAIN_BENCH_*` paths) |

## Adding a New Agent

1. Create directory: `agents/<agent_name>/`
2. Create entry point: `agents/<agent_name>/solve.sh`
3. The `solve.sh` script receives the system prompt via the `$PROMPT` environment variable
   (also some agents take it as a CLI arg). It runs inside the apptainer sandbox with `/home/ben/task`
   as the working directory and should produce `final_model/` there when finished.
4. Add an agent-specific submission template if needed: `src/commit_utils/single_task_<agent>.sub`
5. Per-agent parsing: add a parser in `src/trace_parsing/<agent>_parser.py` and register it in
   `parse_trace.py` so `solve_parsed.txt` is human-readable.
6. If the agent needs persistent OAuth state (e.g. `codex_non_api`), drop an `auth.json` or
   `oauth_token` next to `solve.sh`; `run_task.sh` bind-mounts them into the sandbox.

Example structure:
```bash
#!/bin/bash
# agents/myagent/solve.sh
myagent-cli --model "$AGENT_CONFIG" "$PROMPT"
```

Currently supported agents include: `claude`, `claude_non_api`, `claude_non_api_max`, `codex`,
`codex_non_api` (and `_high`, `_xhigh`, `_reprompt`, ...), `codexhigh`, `codexlow`, `gemini`,
`glm5`, `opencode`, `qwen3max`.

## Adding a New Evaluation Task

1. Create directory: `src/eval/tasks/<task_name>/`
2. Required files:
   - `evaluate.py` - Evaluation script using Inspect AI framework
   - `benchmark.txt` - Official benchmark name (single line)
   - `info.json` - Benchmark metadata used by judges/prompt generation
   - `test_data.json` - Test items used both to compute scores and by the contamination judge
3. Optional files:
   - `evaluation_code/` - Supporting evaluation code copied into the agent sandbox
   - `task_context/` - Additional context (e.g. dataset hints) copied into the agent sandbox

The `evaluate.py` must:
- Use `inspect_ai` framework
- Accept `--model-path` for the model directory and `--templates-dir` for chat templates
- Write metrics to the path given by `--json-output-file` (consumed by aggregation scripts)

## Running Jobs

```bash
# Submit the full agent × benchmark × model sweep (edit the loop / EXPERIMENT_NAME at the top)
bash src/commit_utils/commit.sh

# Submit a single task
condor_submit_bid 100 \
    -a "agent=codex" \
    -a "agent_config=gpt-5.1-codex-max" \
    -a "eval=gsm8k" \
    -a "model_to_train=Qwen/Qwen3-4B-Base" \
    -a "num_hours=10" \
    src/commit_utils/single_task.sub
```

`single_task.sub` accepts extra `-a` overrides such as `num_gpus`, `request_memory`,
`request_cpus`, and `experiment_name` (which becomes the `_run*` suffix on the result dir).

**HTCondor scheduler logs go to `logs/`.** Every `.sub` file writes its per-job `error`,
`output`, and `log` files to `logs/<prefix>_$(Cluster).{err,out,log}` (relative to the submit
directory, which is always the repo root).

## Important Conventions

- **Bash for orchestration**: All entry points and job scripts are bash
- **Python for evaluation**: Use Inspect AI framework for benchmark evaluation
- **Jinja2 for templates**: Chat templates in `src/eval/templates/`
- **HTCondor for scheduling**: Job submission via `.sub` files. Scheduler `error`/`output`/`log`
  files are written under `logs/` (gitignored), never to the repo root.
- **Apptainer for containers**: Container definitions in `.def` files; built `.sif` files live in
  `$POST_TRAIN_BENCH_CONTAINERS_DIR`
- **Sandbox layout**: Inside the container, the home is `/home/ben` and the working dir is
  `/home/ben/task`. The agent must place its trained checkpoint at `task/final_model/`.

## Safety Mechanisms

The project includes contamination detection via two agent-as-judge runs invoked by
`src/run_task.sh` after the agent finishes (also exposed standalone via
`src/disallowed_usage_judge/run_judge.sh`):

1. **GPT-5.4 contamination judge** (codex CLI, subscription auth) — checks for test-data usage,
   eval tampering, model substitution, and forbidden fine-tuning practices.
2. **GPT-5.4 third-party API usage judge** (codex CLI) — separate schema (`disallowed_api_usage`),
   checks whether the agent called external LLM APIs in a disallowed way. It writes its own
   `judgement_api.json` for the record but is **not** consumed by the scoring/aggregation flow.

Each judge writes its own per-judge file; there is no aggregation step. The canonical
contamination verdict consumed downstream is `judgement_gpt5_4.json` (or
`judgement_gpt5_4_rerun.json` when the rerun pipeline has produced it).

Reruns: `src/disallowed_usage_judge/rerun_judge/` holds the batch-rerun pipeline. Rerun outputs
always carry a `_rerun` suffix so original judge files produced during `run_task.sh` are
preserved.

The judge tooling itself lives in:
- `src/disallowed_usage_judge/judge_tools/` — `contamination_check.py`,
  `model_identity_check.py`, and `reference_configs/` (copied into the judge sandbox so the
  judge can run them as tools)
- `src/disallowed_usage_judge/contamination_check_tool/` — helpers to (re)download test data

## Results Structure

```
results/{agent}_{agent_config}_{num_hours}h[_{num_gpus}gpu]{experiment_name}/
└── {benchmark}_{model_name}_{cluster_id}/
    ├── output.log               # run_task.sh stdout
    ├── error.log                # run_task.sh stderr
    ├── prompt.txt               # Generated agent prompt
    ├── time_taken.txt           # Agent execution duration (HH:MM:SS)
    ├── solve_out.txt            # Raw agent trace (stdout+stderr from the agent CLI)
    ├── solve_parsed.txt         # Human-readable trace from src/trace_parsing/parse_trace.py
    ├── task/                    # Snapshot of the agent's working directory (post-cleanup)
    ├── final_model/             # Trained model checkpoint
    ├── system_monitor.log       # GPU/CPU/RAM samples from src/utils/system_monitor.sh
    ├── judge_output_gpt5_4.{json,txt}   # Judge 1 raw + parsed trace
    ├── judgement_gpt5_4.json            # Judge 1 structured verdict
    ├── judge_output_api.{json,txt}      # Judge 2 raw + parsed trace
    ├── judgement_api.json               # Judge 2 structured verdict (archival; not consumed)
    ├── final_eval_*.txt                 # vLLM/inspect-ai evaluation logs (one per retry)
    └── metrics.json                     # Final benchmark scores
```

Result directories with the `_rerun` suffix on `judgement_*.json` come from the rerun-judge
pipeline; original files are kept side-by-side. The canonical contamination verdict is
`judgement_gpt5_4.json` (or `judgement_gpt5_4_rerun.json` when present).

## Code Style

- Shell scripts: Use bash, include `#!/bin/bash` shebang
- Python: Standard library preferred, use type hints
- Error handling: Fail explicitly - do not silently handle errors

## Environment variables

**When writing scripts/tooling, `.env` (at the repo root) is the canonical source for the
`POST_TRAIN_BENCH_*` variables.** Read them directly from `.env` — do **not** assume they are
already exported in the environment, and do **not** source `set_env_vars.sh` from a script
(its module-loading block fails on compute nodes without `tclsh`). Examples of the correct
pattern: the `commit_*.sh` scripts in `src/disallowed_usage_judge/rerun_judge/` (`grep` the var
out of `.env`) and `find_disallowed_api_usage.py` (`load_results_dir_from_env()`, which falls
back to `.env` when the var is unset).

Common ones (defined in `.env`; also sourced via `src/commit_utils/set_env_vars.sh`):
- `POST_TRAIN_BENCH_RESULTS_DIR` — where per-run result directories are written
- `POST_TRAIN_BENCH_CONTAINERS_DIR` — where built `.sif` containers live
- `POST_TRAIN_BENCH_CONTAINER_NAME` — default container for the agent sandbox
- `POST_TRAIN_BENCH_EXPERIMENT_NAME` — suffix added to the result directory name
- `POST_TRAIN_BENCH_JOB_SCHEDULER` — controls which scheduler branch in `commit.sh` runs
- `HF_HOME` — host-side Hugging Face cache that gets overlay-mounted into the sandbox

## Co-Authorship

Never add yourself as a co-author to commits.

## Testing

Usually when testing, it is best to use the openai model `gpt-5.1-codex-max`, because we have a
lot of api credits for openai.

## Notes

Add your own notes to the directory `/home/brank/Documents/agents_notes` and index them via
`/home/brank/Documents/agents_notes/index.md`. Add those notes for other agents and for yourself
for future use.

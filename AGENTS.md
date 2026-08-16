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
│   ├── judges/                # Reward-hacking judges (one folder per judge; see Safety)
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
| `src/run_task.sh` | Main task execution orchestrator (runs agent, then 4 judges, then evaluation) |
| `src/commit_utils/commit.sh` | Batch job submission across agents × benchmarks × models |
| `src/commit_utils/set_env_vars.sh` | Sources `.env` and exports `POST_TRAIN_BENCH_*` env vars |
| `src/commit_utils/single_task.sub` | HTCondor submission template |
| `src/commit_utils/single_task_gemini.sub` | Gemini-specific HTCondor submission template |
| `src/eval/general/get_prompt.py` | Generates agent prompts |
| `src/eval/general/prompt.txt` | Agent prompt template |
| `src/trace_parsing/parse_trace.py` | Dispatches to per-agent parser to produce human-readable trace |
| `src/utils/update_agent_cli.sh` | Auto-updates an agent's CLI harness to latest and records its version |
| `src/judges/run_judges.sh` | Runs judges on an existing result dir (`--judges` to select a subset); each writes its own per-judge JSON |
| `src/judges/get_judge_prompt.py` | Generates judge prompts (`--judge <judge_name>`) |
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
7. Auto-update the CLI harness: call `bash /home/ben/update_agent_cli.sh <cli-binary>` in
   `solve.sh` just before launching the CLI (e.g. `... update_agent_cli.sh claude`). This upgrades
   the harness to the latest npm release and writes its version to `cli_version.txt` (surfaced in
   the result dir). The helper (`src/utils/update_agent_cli.sh`, copied into the sandbox by
   `run_task.sh`) holds the binary→npm-package mapping; add a `case` entry there if the agent uses
   a CLI not already covered (`claude`, `codex`, `gemini`, `mcode`, `opencode`). The update is
   best-effort — a failure falls back to the container's pinned version and still records what
   actually ran.
   Set `POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1` in `.env` to disable the update globally and pin CLI
   versions to whatever the container ships; `cli_version.txt` still records what ran
   (`update: skipped`).

Example structure:
```bash
#!/bin/bash
# agents/myagent/solve.sh
myagent-cli --model "$AGENT_CONFIG" "$PROMPT"
```

**Required: `agents/<agent_name>/api_keys.json`** declares the third-party API keys the agent is
allowed to receive, e.g. `{"allowed_api_keys": ["CODEX_API_KEY"]}` (use `[]` for subscription-auth
agents like `codex_non_api`). `run_task.sh` passes ONLY these keys — plus any the benchmark
requires — into the `-c --cleanenv` sandbox; every other provider key is never injected (so it is
unset inside the sandbox). A missing `api_keys.json` is a hard error. See "API Key Provisioning"
below. With the allowlist in place there is **no** need to `unset`/blank keys inside `solve.sh`.

Currently supported agents include: `claude`, `claude_non_api`, `claude_non_api_max`, `codex`,
`codex_non_api` (and `_high`, `_xhigh`, `_reprompt`, ...), `codexhigh`, `codexlow`, `cursor_cli`,
`gemini`, `glm5`, `grok_cli`, `mcode`, `opencode`, `qwen3max`.

`cursor_cli` uses the Cursor CLI (`agent`) with subscription auth: `solve.sh` installs the CLI
via the official curl installer (`cursor.com/install`, drops `agent` + `cursor-agent` symlinks in
`$HOME/.local/bin` pointing at `$HOME/.local/share/cursor-agent/versions/<v>/cursor-agent`) and
runs it with `--print --force --trust --output-format stream-json --workspace /home/ben/task`.
Auth tokens live at `~/.config/cursor/auth.json`; `run_task.sh` bind-mounts
`agents/cursor_cli/cursor_auth.json` → `/home/ben/.config/cursor/auth.json` so the CLI reads *and*
rotates tokens against the shared host file. Set it up:

```
curl -fsS https://cursor.com/install | bash    # installs to ~/.local/bin/{agent,cursor-agent}
export PATH="$HOME/.local/bin:$PATH"
cursor-agent login                             # completes OAuth in a browser
cp ~/.config/cursor/auth.json agents/cursor_cli/cursor_auth.json
chmod 600 agents/cursor_cli/cursor_auth.json
```

Pass the Cursor model ID as `agent_config` (e.g. `grok-4.5`, `gpt-5`, `sonnet-4-thinking`). Run
`agent --list-models` from a *compute-node* shell (head-node routes to Cursor's API are typically
blocked by corp proxy) to see the full list.

`grok_cli` uses the xAI Grok Build CLI (`grok`) with subscription auth: `solve.sh` installs the CLI
via the official curl installer (`x.ai/cli/install.sh`, drops binary at `$HOME/.grok/bin/grok`) and
runs it with `--oauth --always-approve --no-auto-update --output-format streaming-json`. Auth
tokens live at `~/.grok/auth.json` (not `config.toml`, which is preferences only); `run_task.sh`
bind-mounts `agents/grok_cli/grok_auth.json` → `/home/ben/.grok/auth.json` so the CLI reads *and*
rotates tokens against the shared host file. Set it up:

```
grok login                # or `grok login --device-auth` on a headless machine
cp ~/.grok/auth.json agents/grok_cli/grok_auth.json
chmod 600 agents/grok_cli/grok_auth.json
```

The explicit model ID for Grok 4.5 is `grok-4.5` (per `docs.x.ai/docs/models`) — pass it as
`agent_config` so `-m grok-4.5` pins the exact version regardless of the floating `grok-build`
alias in the CLI's `[models] default`.

## Adding a New Evaluation Task

1. Create directory: `src/eval/tasks/<task_name>/`
2. Required files:
   - `evaluate.py` - Evaluation script using Inspect AI framework
   - `benchmark.txt` - Official benchmark name (single line)
   - `info.json` - Benchmark metadata used by judges/prompt generation. Add
     `"required_api_keys": ["OPENAI_API_KEY"]` here if the benchmark's `evaluate.py` needs a
     provider key for grading (only `arenahardwriting`/`healthbench` do today); the key is then
     provisioned into the agent sandbox for that benchmark. Defaults to none if omitted.
   - `test_data.json` - Test items used to compute scores and by the contamination judge; when
     present, `run_task.sh` also copies it (plus `contamination_check.py`) into the agent sandbox
     home and `get_prompt.py` adds a "Decontamination Tool" prompt section, so the agent can
     screen its own training data for test-set overlap
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

## API Key Provisioning

The agent sandbox is launched by `run_task.sh` with `apptainer -c --cleanenv` (plus `--pid
--no-init`), so it inherits **nothing** from the host environment — a variable exists inside the
sandbox only if an explicit
`--env` flag passes it. `run_task.sh` builds that `--env` list from an allowlist, so the policy is
declarative and lives in two places:

- **`agents/<agent>/api_keys.json`** → `{"allowed_api_keys": [...]}` — the provider keys this agent
  is permitted to use (e.g. `["ANTHROPIC_API_KEY"]`, `["CODEX_API_KEY"]`, or `[]` for
  subscription-auth agents).
- **`src/eval/tasks/<task>/info.json`** → `"required_api_keys": [...]` — keys the benchmark's own
  grading needs (only `arenahardwriting`/`healthbench`, which need `OPENAI_API_KEY`).

The sandbox receives exactly the **union** of the two. Consequences:

- `OPENAI_API_KEY` reaches an agent **only** when it runs `arenahardwriting`/`healthbench`, so the
  agent can run `evaluate.py`'s OpenAI judge; for every other benchmark it is absent.
- Subscription-auth agents (`*_non_api`) list `[]` and authenticate via their bind-mounted
  `auth.json`/`oauth_token`; on the two writing benchmarks they additionally receive
  `OPENAI_API_KEY` purely for grading.
- `solve.sh` scripts must NOT `unset`/blank keys — the allowlist already withholds them. Genuine
  per-agent logic stays (e.g. `glm5`/`qwen3max` remap their provider key into `ANTHROPIC_API_KEY`;
  `codex_non_api` sets `forced_login_method`).
- The host-side `CODEX_API_KEY`/`OPENAI_API_KEY` setup (renaming the OpenAI key for codex agents and
  restoring `OPENAI_API_KEY` for the two writing benchmarks) lives near the top of `run_task.sh`.

Note: `VLLM_API_KEY` (the local inspect/vLLM eval server) is operational, not a third-party key, and
is always passed as the constant `inspectai`; it is not part of the allowlist.

## Safety Mechanisms

The project includes reward-hacking detection via agent-as-judge runs invoked by
`src/run_task.sh` after the agent finishes (also exposed standalone via
`src/judges/run_judges.sh`). Each judge is a folder under `src/judges/` with a `judge.conf`
and a prompt template (see `src/judges/README.md`, including how to add a new judge):

1. **`data_contamination_judge`** (GPT-5.4 via codex CLI, subscription auth) — checks for
   test-data usage, eval tampering, model substitution, and forbidden fine-tuning practices.
2. **`api_usage_judge`** (GPT-5.4 via codex CLI) — separate schema (`disallowed_api_usage`),
   checks whether the agent called external LLM APIs in a disallowed way. Its verdict
   (`judgement_api.json`) **is** consumed by scoring: a flagged run falls back to the baseline
   score in `scripts/collect.py`.
3. **`ptb_lookup_judge`** (GPT-5.4 via codex CLI) — separate schema
   (`disallowed_ptb_lookup`), checks whether the agent looked up PostTrainBench itself (the
   website, the GitHub repo, or published traces of past runs, e.g. to copy strategies). Its
   verdict (`judgement_ptb_lookup.json`) is archival — it does not feed score fallback — but
   `scripts/collect.py` raises an error if it ever flags, so a firing lookup judge cannot pass
   unnoticed.
4. **`general_judge`** (GPT-5.6 Terra via codex CLI pinned to 0.144.5) — separate schema
   (`general_anomaly`), an open-ended sweep for **unknown unknowns**: run-integrity problems
   the other judges were not designed to catch, whether caused by the agent (novel reward
   hacking outside the other judges' scope) or suffered by it (premature stop without visible
   reason, agent usage limits / token exhaustion, grader-API credit exhaustion on
   arenahardwriting/healthbench, harness or infra failures). Its verdict
   (`judgement_general.json`) is purely archival: `scripts/collect.py` ignores it entirely —
   no score impact and no error, whether it is missing or flagged. Review its flags with
   `scripts/find_flagged_runs.py --judge general_judge`.

Each judge writes its own per-judge file; there is no aggregation step. The canonical verdicts
consumed downstream are `judgement_gpt5_4.json` (contamination/disallowed_model) and
`judgement_api.json` (API usage) — or their `_rerun` variants when the rerun pipeline has
produced them. `scripts/collect.py` enforces judge coverage: every scored run (metrics.json
present) must carry a contamination and an API verdict, and runs with ids >=
`NEWER_JUDGES_MIN_RUN_ID` (`scripts/utils.py`) a PTB-lookup verdict too; a method containing a
scored run without a required verdict is skipped (warning, no CSVs) instead of aggregated.
Runs below that id may legitimately lack the PTB-lookup file (they predate the judge) — a
verdict present on such a run is still read as a tripwire. The general judgement is never
required.

Reruns: `src/judges/rerun/` holds the batch-rerun pipeline. Rerun outputs
always carry a `_rerun` suffix so original judge files produced during `run_task.sh` are
preserved.

The judge tooling itself lives in:
- `src/judges/judge_lib.sh` — shared bash helpers used by both `run_task.sh` and
  `run_judges.sh` (sandbox prep, codex invocation, output collection)
- `src/judges/judge_tools/` — `contamination_check.py`,
  `model_identity_check.py`, and `reference_configs/` (copied into the judge sandbox so the
  judge can run them as tools). `contamination_check.py` (with the benchmark's
  `test_data.json`) is additionally copied into the agent sandbox home before the run, so the
  agent can self-decontaminate its training data with the exact checker the judge uses; the
  judge phase re-copies both files, so the judges never run agent-modified versions.
- `src/judges/test_data_download/` — helpers to (re)download test data

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
    ├── cli_version.txt          # Auto-updated agent CLI harness version (src/utils/update_agent_cli.sh)
    ├── task/                    # Snapshot of the agent's working directory (post-cleanup)
    ├── final_model/             # Trained model checkpoint
    ├── system_monitor.log       # GPU/CPU/RAM samples from src/utils/system_monitor.sh
    ├── judge_output_gpt5_4.{json,txt}   # data_contamination_judge raw + parsed trace
    ├── judgement_gpt5_4.json            # data_contamination_judge structured verdict
    ├── judge_output_api.{json,txt}      # api_usage_judge raw + parsed trace
    ├── judgement_api.json               # api_usage_judge structured verdict (flag ⇒ baseline score)
    ├── judge_output_ptb_lookup.{json,txt} # ptb_lookup_judge raw + parsed trace
    ├── judgement_ptb_lookup.json        # ptb_lookup_judge structured verdict (archival; collect.py errors if flagged)
    ├── judge_output_general.{json,txt}  # general_judge raw + parsed trace
    ├── judgement_general.json           # general_judge structured verdict (archival; ignored by collect.py)
    ├── final_eval_*.txt                 # vLLM/inspect-ai evaluation logs (one per retry)
    └── metrics.json                     # Final benchmark scores
```

Result directories with the `_rerun` suffix on `judgement_*.json` come from the rerun-judge
pipeline; original files are kept side-by-side. The canonical contamination verdict is
`judgement_gpt5_4.json` (or `judgement_gpt5_4_rerun.json` when present); the canonical
API-usage verdict is `judgement_api.json` (or `judgement_api_rerun.json`).

## Code Style

- Shell scripts: Use bash, include `#!/bin/bash` shebang
- Python: Standard library preferred, use type hints
- Error handling: Fail explicitly - do not silently handle errors

## Environment variables

**When writing scripts/tooling, `.env` (at the repo root) is the canonical source for the
`POST_TRAIN_BENCH_*` variables.** Read them directly from `.env` — do **not** assume they are
already exported in the environment, and do **not** source `set_env_vars.sh` from a script
(its module-loading block fails on compute nodes without `tclsh`). Examples of the correct
pattern: `src/judges/rerun/commit_rerun_judges.sh` (`grep` the var
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

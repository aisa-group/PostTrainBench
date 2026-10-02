# PostTrainBench Harbor Adapter

Run PostTrainBench on cloud GPUs through [Harbor](https://harborframework.com) and
[Modal](https://modal.com). The original pipeline runs on an HTCondor cluster. This document calls
it "condor". The adapter runs the same prompt, the same judges and the same evaluation as condor.
The intended differences are listed in [Differences from condor](#differences-from-condor).

## Benchmarks and models

The 6 scored benchmarks (PostTrainBench v1.2): aime2025, arenahardwriting, gpqamain, gsm8k,
healthbench, humaneval.

| Key | Base model |
|-----|-----------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

The word "all" selects all benchmarks or all models: 6 x 4 = 24 tasks.

## What you need

- A Modal account. Each task uses one H100 GPU.
- An OpenAI API key. The judges use it on every run. arenahardwriting and healthbench also grade
  with it.
- A read-only Hugging Face token. Two of the repositories are gated.
- The API key of your agent. For claude-code that is ANTHROPIC_API_KEY.
- [uv](https://docs.astral.sh/uv/).

## Setup (once)

1. Install Harbor and log in to Modal:

    ```bash
    uv tool install --force 'harbor[modal]>=0.23.0'
    harbor_py="$(dirname "$(readlink -f "$(command -v harbor)")")/python"
    $harbor_py -m modal setup
    ```

    If you are behind an HTTP proxy, also install the two proxy packages:

    ```bash
    uv pip install --python "$harbor_py" python-socks 'modal[api-proxy-support]'
    ```

    Without python-socks, each Modal call fails with "Could not connect to the Modal server".
    Without aiohttp-socks (the api-proxy-support extra), uploads over 2 MiB fail.

2. Copy example.env to .env at the repository root. Write your keys into it (see [Keys](#keys)).
   All scripts here read the keys from this file. You do not have to export them.

3. Download the test sets. The script writes each one to src/eval/tasks/\<id\>/test_data.json
   (gitignored). Task generation stops if one is missing.

    ```bash
    # from the repository root
    uv run --no-project --with datasets --with huggingface_hub --with pyarrow \
        python src/judges/test_data_download/download_test_data.py
    ```

## Run

One command generates the tasks and runs them. For the full set:

```bash
cd src/harbor_adapter
bash run_modal_task.sh --benchmark all --base-model all \
    --agent claude-code --model anthropic/claude-opus-4-8 --job-name sweep1
```

- The options --benchmark and --base-model accept "all" or comma-separated lists. The option
  --num-hours sets the agent budget in hours (default 10).
- The tasks go to tasks/\<job-name\>/. All of them start at once; --parallel N limits how many run
  at the same time. Each task runs as its own harbor job, with its results in
  jobs/\<job-name\>/\<task\>/ and its log in jobs/\<job-name\>/\<task\>.log. The command waits and
  then prints a table of exit codes, rewards and exceptions.
- Each task gets a Modal volume, ptb-\<job-name\>-\<task\>. The volume keeps the trained model
  after the run. To download a model: `$harbor_py -m modal volume get <volume> / ./final_model`.
  The option --delete-volume removes the volume after the run.

To run one task in the foreground:

```bash
bash run_modal_task.sh --task tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code --model anthropic/claude-opus-4-8 --job-name run1
```

Extra harbor run flags go after `--`. For example, `-- --agent-timeout-multiplier 0.1` gives a
short test run, and `-- --yes` runs a single task unattended.

## Results

Harbor's reward is the accuracy before the baseline fallback. The fallback, the aggregation, the
review of flagged runs and the judge reruns stay in the condor tooling. First export the trials
into the condor results layout (POST_TRAIN_BENCH_RESULTS_DIR from .env; --output overrides it):

```bash
python harbor_to_results.py jobs/sweep1                       # every trial below the dir
python harbor_to_results.py jobs/* --experiment-name _harbor  # suffix for the method dirs
python harbor_to_results.py jobs/sweep1 --with-model          # also download each final_model

# then, from the repository root:
python scripts/collect.py       # final_<method>.csv, with the baseline fallback
python scripts/find_flagged_runs.py --judge data_contamination_judge
bash src/judges/run_judges.sh <results>/<method>/<run>        # judge a run again
```

Each trial becomes \<results\>/\<agent\>_\<agent-model\>_\<N\>h/\<benchmark\>_\<Org\>_\<Model\>_\<run-id\>/.
It has the same files as a condor run dir, plus harbor/ (result.json, config.json). The run-id is
the trial start time in Unix seconds. The exporter parses the traces again on the host, so the
*_sanitized copies redact the keys in your .env. The agent's API cost is in result.json
(agent_result.cost_usd).

## Keys

| Key | Used by | Needed for |
|-----|---------|-----------|
| OPENAI_API_KEY | the four judges (codex CLI); arenahardwriting/healthbench grading; the codex agent | every run |
| the agent's keys, from condor's agents/\<agent\>/api_keys.json | the agent | claude-code: ANTHROPIC_API_KEY (or Claude subscription, below); gemini-cli: GEMINI_API_KEY; opencode: OPENCODE_API_KEY / ZAI_API_KEY / MODEL_API_KEY |
| HF_TOKEN (read-only) | agent and verifier Hub downloads | every run (checked at launch); gated: google/gemma-3-4b-pt (gemma3-4b tasks), Idavidrein/gpqa (gpqamain) |

- The keys come from the .env file at the repository root (PTB_ENV_FILE overrides the path). A
  variable that you export in the shell wins over the file. The launcher loads only the keys in
  this table. The other keys in .env never enter the process and cannot reach a sandbox. The
  verifier gets OPENAI_API_KEY also as CODEX_API_KEY.
- opencode needs only the key of its model's provider: ZAI_API_KEY for zai models, MODEL_API_KEY
  for meta models, OPENCODE_API_KEY for opencode models.
- To use a Claude subscription in place of an API key, add CLAUDE_CODE_OAUTH_TOKEN (from
  `claude setup-token`) and CLAUDE_FORCE_OAUTH=1 to .env. Harbor then keeps ANTHROPIC_API_KEY out
  of the sandbox.
- HF_TOKEN: the sandboxes download from the Hub, and the gated repositories need a token. For this
  key the .env file wins over the shell, so a write token that you exported for other work is not
  picked up. The agent can read the token. Because of that, check_hf_token.py rejects a token with
  write permissions before launch, and it checks access to every repo in gated_hf_resources.json.
  Create a Read token at <https://huggingface.co/settings/tokens>. A fine-grained token also needs
  "Read access to contents of all public gated repos you can access".

## Agent CLI version

The launcher resolves one exact version and passes it to harbor as `--ak version=<x.y.z>`. Harbor
installs that version at agent setup. The precedence is the same as in condor's
src/utils/update_agent_cli.sh:

1. The option --cli-version, "latest" or an exact version.
2. CLAUDE_CLI_VERSION / CODEX_CLI_VERSION / GEMINI_CLI_VERSION / OPENCODE_CLI_VERSION, exported or
   in .env. A pin is strict: it must exist on npm, and a failed install fails the trial.
3. POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1: the version in template/environment/Dockerfile.
4. Otherwise the latest release.

The launcher resolves "latest" once, so a whole sweep runs the same version. The version that ran
is in result.json (agent_info.version; the exporter writes it to cli_version.txt).

## Agent launch (claude-code)

| condor agents/claude*/solve.sh | run_modal_task.sh |
|---|---|
| CLAUDE_CODE_EFFORT_LEVEL=high | `--ak reasoning_effort=high` (option --effort) |
| BASH_MAX_TIMEOUT_MS=36000000 | `--ae BASH_MAX_TIMEOUT_MS=36000000` |
| update_agent_cli.sh | [Agent CLI version](#agent-cli-version) |
| prompt on stdin to claude --print | the same (harbor's agent) |
| --thinking-display summarized | `--ak thinking_display=summarized` (option --thinking-display) |

Without "--thinking-display summarized" the judges do not see the agent's reasoning: Claude Code's
print mode leaves the thinking blocks empty. The kwarg needs harbor 0.23.0 or newer;
"--thinking-display none" omits it for older versions.

## How it works

Task internals (layout, the model hand-off over a shared Modal volume, the verifier's judge and
evaluation phases) are described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Differences from condor

Everything the adapter mirrors from condor instead of reading it (container pins, sandbox env,
eval retry ladder, claude launch settings, CLI packages, timer, verdict fields) is compared by
check_parity.py. CI (.github/workflows/harbor-parity.yml) runs it on every change to those condor
files. Run it locally with: `uv run --no-project --python 3.12 python check_parity.py`. The
differences below are intended; the check allows exactly these.

| | condor (single_task.sub, run_task.sh) | Harbor |
|---|---|---|
| GPU | 1x NVIDIA H100 80GB HBM3 | gpu_types = ["H100!"]: the ! stops Modal from upgrading to an H200 |
| CPUs | request_cpus = 16 | cpus = 16 (nproc = 16) |
| RAM | 128 GB, hard cap | memory_mb = 131072 is a reservation only; `-- --memory guarantee` also caps it |
| Disk | request_disk = 400G | storage_mb is ignored by Modal; the host disk is effectively unbounded |
| Agent time | num_hours + 5 min, timer starts at job setup | exactly num_hours, timer starts right before the agent |
| Verifier time | no limit (eval: 8 h per attempt) | judge runs (6) x 3000 s + 4 h for the eval = 9 h, derived from judge_lib.sh by adapter.py; 2 h per eval attempt (EVAL_ATTEMPT_TIMEOUT_SEC), so a hung attempt leaves time for the retries |
| humaneval answer sandbox | a second apptainer container (with_answer_sandbox.sh) | a namespace jail in the verifier (with_answer_sandbox_local.sh; apptainer cannot run inside Modal's gVisor): same isolation - nobody, no network, own PID namespace, read-only allow-list root with only the image's system dirs; details in the script's header |
| HF cache | pre-filled HF_HOME overlay | none: the base model downloads inside the agent's budget |
| Judge auth | ChatGPT subscription auth.json in gpt_5_5.sif | OPENAI_API_KEY, directly in the verifier image (JUDGE_RUNTIME=local) |
| Judge confinement | codex in a --containall apptainer sandbox that only sees its judge dir | codex as nobody (setpriv) in the verifier container: it can write only its judge dir; /logs, the baked ptb tree and the model volume's files stay root-owned |
| CLI pins | also per-model pins in agents/claude_non_api_max, agents/glmx solve.sh | not mirrored |
| Code the judges see | the whole task/ dir, minus every dir that looks like a HF model (containers/delete_hf_models.py) | a snapshot: files <= 512 MiB, <= 2 GiB total, smallest first, no weight formats or caches |
| Record of what was dropped | output.log | .ptb_workspace_sizes.txt in the snapshot |

In practice the judges read source code, data files and logs, which both keep. The Harbor snapshot
only drops files on a workspace with more than 2 GiB of non-weight data. Leftover model
directories (final_model2/, checkpoints) lose their weights but keep their small config files.

## Known gotchas

- The images mirror containers/opus_5.def, with flash-attn 2.8.3. The Grok and Cursor CLIs from
  that def are not installed. The image's agent CLI versions (Claude Code 2.1.219, codex 0.144.0,
  gemini-cli 0.18.4, opencode 1.17.18) matter only under POST_TRAIN_BENCH_SKIP_CLI_UPDATE.
- codex must not read stdin in the verifier: codex exec appends piped stdin to its prompt and
  waits for EOF, and harbor's exec stdin never closes. test.sh runs every judge with stdin closed
  (apptainer closes stdin, so condor is unaffected).

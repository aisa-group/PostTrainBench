# PostTrainBench Harbor Adapter

Runs PostTrainBench on cloud GPUs (Modal) through [Harbor](https://harborframework.com), with the
same prompt, judges and evaluation as the condor pipeline (`src/run_task.sh`).

## Benchmarks and models

Benchmarks come from `src/eval/tasks/*/info.json` (PostTrainBench v1.2): aime2025, arenahardwriting,
gpqamain, gsm8k, healthbench, humaneval. arenahardwriting and healthbench grade with an OpenAI judge
(`required_api_keys` in their `info.json`), so the agent also receives `OPENAI_API_KEY` for them.

| Key | Base model |
|-----|-----------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

`all` (`run_modal_task.sh --benchmark all`, `run_adapter.py --all`) means the benchmarks condor
scores, `HARDCODED_BENCHMARKS` in `scripts/utils.py`: 6 benchmarks x 4 models = **24 tasks**.

**humaneval's answer sandbox.** Its v1.2 scorer runs the model's answers in a separate, isolated
sandbox: on condor a second apptainer container (`with_answer_sandbox.sh`). apptainer cannot run
inside Modal's gVisor sandbox, so in the verifier the answers run in
`src/eval/tasks/humaneval/with_answer_sandbox_local.sh`: the same answer server, jailed with the
namespaces apptainer itself uses (created inside a user namespace, which Modal allows): as `nobody`,
no network, its own PID namespace, and an allow-list root with only the image's system dirs,
read-only. The tests, logs, keys and HF cache do not exist in there. `run_final_eval.sh` uses it by
default with `EVAL_RUNTIME=local`, and never runs the answers without it.

## Quick start

### 1. Setup (once)

```bash
uv tool install --force 'harbor[modal]>=0.23.0'
harbor_py="$(dirname "$(readlink -f "$(command -v harbor)")")/python"
uv pip install --python "$harbor_py" python-socks 'modal[api-proxy-support]'   # behind an HTTP proxy only; --force above removes them
$harbor_py -m modal setup                            # Modal login
```

Behind a proxy Modal needs both: without `python-socks` every call fails with "Could not connect to the Modal server"; without `aiohttp-socks` (the `api-proxy-support` extra) uploads of files over 2 MiB fail with "the 'aiohttp-socks' package is not installed" (e.g. healthbench's test set in the image build context).

Put all keys in the repo-root `.env` (template: `example.env`; details in [Keys](#keys)):
`OPENAI_API_KEY`, a **read-only** `HF_TOKEN`, and your agent's key (`ANTHROPIC_API_KEY` for
claude-code). Every script here reads them from there; nothing needs exporting.

Then download the benchmarks' test sets, once. `download_test_data.py` writes each
one to the gitignored `src/eval/tasks/<id>/test_data.json` (question/answer pairs). Task generation stops if one is missing.

```bash
# from the repo root
uv run --no-project --with datasets --with huggingface_hub --with pyarrow \
    python src/judges/test_data_download/download_test_data.py
```

### 2. Run

One command generates the tasks and runs them, here the full set:

```bash
cd src/harbor_adapter
bash run_modal_task.sh --benchmark all --base-model all \
    --agent claude-code --model anthropic/claude-opus-4-8 --job-name sweep1
```

- `--benchmark` / `--base-model` take `all` (the scored benchmarks, see above) or comma-separated
  lists (`--benchmark gsm8k,humaneval --base-model qwen3-1.7b`). `--num-hours H` sets the agent
  budget (default 10).
- Tasks are generated into `tasks/<job-name>/`. All of them launch at once (`--parallel N` caps
  it), each as its own `harbor run` with its own Modal volume `ptb-<job-name>-<task>`, results in
  `jobs/<job-name>/<task>/` and launcher log `jobs/<job-name>/<task>.log`. The command waits and
  prints a table of exit codes, rewards and exceptions.
- The volumes are **kept**; they hold the trained models
  (`$harbor_py -m modal volume get <volume> / ./final_model`). `--delete-volume` removes each one
  after its run.
- Harbor 0.23 asks before loading host environment variables into a run; sweeps answer `--yes`.
  The variables are the keys the script resolved and printed at launch.

To run one already generated task (`python run_adapter.py --benchmark gsm8k --model qwen3-1.7b
--output ./tasks`) in the foreground, with harbor's prompt, in `jobs/<job-name>/`:

```bash
bash run_modal_task.sh --task tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code --model anthropic/claude-opus-4-8 --job-name run1
```

Extra `harbor run` flags go after `--`: e.g. `-- --agent-timeout-multiplier 0.1` for a short
debugging run, `-- --yes` for an unattended single task.

### 3. Export and aggregate

Harbor's reward is the accuracy **before** the baseline fallback. Fallback, aggregation, flagged-run
review and judge reruns stay in the condor tooling, so export the trials into the condor results
layout (`POST_TRAIN_BENCH_RESULTS_DIR` from `.env` by default, `--output` to override):

```bash
python harbor_to_results.py jobs/sweep1                         # every trial below a sweep or job dir
python harbor_to_results.py jobs/* --experiment-name _harbor    # suffix the method dirs
python harbor_to_results.py jobs/sweep1 --with-model            # also fetch each final_model from its volume

# then, from the repo root:
python scripts/collect.py                                        # final_<method>.csv, with baseline fallback
python scripts/find_flagged_runs.py --judge data_contamination_judge
bash src/judges/run_judges.sh <results>/<method>/<run>           # re-judge a run
```

Each trial becomes `<results>/<agent>_<agent-model>_<N>h[<experiment>]/<benchmark>_<Org>_<Model>_<run_id>/`
(e.g. `claude-code_anthropic_claude-opus-4-8_1h/gsm8k_Qwen_Qwen3-1.7B-Base_1788169511`; `run_id` is
the trial start in Unix seconds) with the same files as a condor run dir, plus `harbor/`
(`result.json`, `config.json`). The traces are re-parsed on the host, so the `*_sanitized` copies
redact the keys in your `.env`. The exporter needs the task dir the trial was generated from (for
`prompt.txt` and metadata) and otherwise falls back to parsing the task name.

## Configuration

### Keys

| Key | Used by | Needed for |
|-----|---------|-----------|
| `OPENAI_API_KEY` | the four judges (codex CLI); arenahardwriting/healthbench grading; the codex agent | every run |
| the agent's keys, from condor's `agents/<agent>/api_keys.json` | the agent | claude-code: `ANTHROPIC_API_KEY` (or Claude subscription, below); gemini-cli: `GEMINI_API_KEY`; opencode: `OPENCODE_API_KEY` / `ZAI_API_KEY` / `MODEL_API_KEY` |
| `HF_TOKEN` (read-only) | agent and verifier Hub downloads | every run (checked at launch); gated: `google/gemma-3-4b-pt` (gemma3-4b tasks), `Idavidrein/gpqa` (gpqamain) |

- **Where keys come from:** the repo-root `.env` (`PTB_ENV_FILE` overrides the path); a variable
  exported in the shell wins over it. `run_modal_task.sh` loads only the keys in this table, so
  the other keys in `.env` never enter the process and cannot reach a sandbox. The verifier gets
  `OPENAI_API_KEY` also as `CODEX_API_KEY`.
- **opencode** needs only the key of its model's provider: `ZAI_API_KEY` for `zai/...` models,
  `MODEL_API_KEY` for `meta/...`, `OPENCODE_API_KEY` for `opencode/...`. A missing key is passed on
  empty and only fails at the agent's first API call (condor behaves the same).
- **Claude subscription** instead of an API key: add `CLAUDE_CODE_OAUTH_TOKEN=<token from claude
  setup-token>` and `CLAUDE_FORCE_OAUTH=1` to `.env` (harbor then keeps `ANTHROPIC_API_KEY` out of
  the sandbox). condor's `agents/claude_non_api/oauth_token` holds such a token.
- **`HF_TOKEN`:** condor reads gated models and datasets from its pre-filled HF cache; Harbor
  sandboxes download from the Hub, which needs a token for gated repos. For this key `.env` beats
  the shell (so a write token exported for other work is not picked up; `$HF_HOME/token` is
  ignored); the wrapper, `check_hf_token.py` and the test-data downloader all resolve it that way.
  The token is passed to both sandboxes. The agent can read it, so before launch `check_hf_token.py` rejects any token
  with write permissions and checks that the account can open every repo in
  `gated_hf_resources.json` (curating that list: see the script's docstring). Create a *Read*
  token at <https://huggingface.co/settings/tokens>; a fine-grained one also needs "Read access to
  contents of all public gated repos you can access". `dev_utils/extract_traces.py` redacts it only
  when it is in `.env`; otherwise add it to the `POST_TRAIN_BENCH_SANITIZATION_SECRETS` file.

### Agent CLI version

The wrapper resolves one exact version and passes it as `--ak version=<x.y.z>`; harbor installs it
at agent setup (the image's version only saves that install when it matches). Precedence, as in
condor the condor version's `src/utils/update_agent_cli.sh`:

1. `--cli-version latest|<x.y.z>`
2. `CLAUDE_CLI_VERSION` / `CODEX_CLI_VERSION` / `GEMINI_CLI_VERSION` / `OPENCODE_CLI_VERSION`,
   exported or in `.env`. Strict: it must exist on npm, and a failed install fails the trial.
3. `POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1`: the version in `template/environment/Dockerfile`.
4. Otherwise the latest release.

`latest` is resolved with `npm view` once at launch, so a whole sweep runs the same version. The
version that ran is in `result.json` (`agent_info.version`, exported as `cli_version.txt`).

### Agent launch (claude-code)

| condor `agents/claude*/solve.sh` | `run_modal_task.sh` |
|---|---|
| `CLAUDE_CODE_EFFORT_LEVEL=high` | `--ak reasoning_effort=high` (`--effort <level>`, or `--effort none`) |
| `BASH_MAX_TIMEOUT_MS=36000000` | `--ae BASH_MAX_TIMEOUT_MS=36000000` |
| `update_agent_cli.sh` | [Agent CLI version](#agent-cli-version) |
| prompt on stdin to `claude --print` | same (harbor's agent) |
| `--thinking-display summarized` | `--ak thinking_display=summarized` (`--thinking-display summarized\|omitted\|none`) |

Without `--thinking-display summarized`, Claude Code's `--print` mode writes `thinking` blocks with
empty text, so the judges would not see the agent's reasoning. The kwarg needs harbor >= 0.23.0
([harbor#3030](https://github.com/harbor-framework/harbor/pull/3030)); `--thinking-display none`
omits it for older versions. 

## How it works

### Task layout

```
posttrainbench-gsm8k-qwen3-1.7b/
├── task.toml              # GPU/CPU/RAM, timeouts, env vars, verifier.collect hook
├── instruction.md         # agent prompt, rendered by src/eval/general/get_prompt.py
├── environment/           # AGENT image build context (COPY . -> /home/agent/workspace)
│   ├── Dockerfile, .dockerignore, requirements-direct.txt
│   ├── entrypoint.sh, system_monitor.sh, ptb_collect.sh  # -> /usr/local/bin (not in the workspace)
│   ├── contamination_check.py, test_data.json            # -> /home/agent/ (agent self-decontamination)
│   ├── evaluate.py, templates/, timer.sh, metadata.json  # the agent's workspace
│   └── evaluation_code/                                  # (arenahardwriting, healthbench)
└── tests/                 # VERIFIER image build context, baked into /tests
    ├── Dockerfile, requirements-direct.txt, entrypoint.sh, system_monitor.sh
    ├── test.sh            # verifier: condor's judge phase, then its final evaluation (reads $PTB_MODEL_DIR)
    ├── metadata.json      # adds baseline_accuracy (verifier only)
    └── ptb/               # repo slice: src/judges, src/trace_parsing, src/eval/run_final_eval.sh,
                           # src/eval/tasks/<id>/ (final-eval scripts, test set), templates, src/utils helpers
```

The prompt is rendered by condor's own `get_prompt.py`. `timer.sh` counts down from `/timer_start`, which the `task.toml` healthcheck writes right before the agent launches.

### Model hand-off: shared Modal volume

The verifier runs in a **separate sandbox** (`[verifier] environment_mode = "separate"`), so the
agent cannot tamper with `evaluate.py`, the judges or the installed packages. The trained weights
therefore have to cross sandboxes, and harbor's artifact transfer cannot carry them: Modal's file
download has a hard 5 GiB per-file limit, which every base model exceeds.

Instead (all in `template/task.toml`):

1. A Modal volume, passed at launch as `--ek 'volumes={"/mnt/ptb_final_model":"<name>"}'`, is
   mounted at `/mnt/ptb_final_model` in **both** sandboxes.
2. The agent writes `final_model/` in `/home/agent/workspace` exactly as on condor; its
   instructions never mention the volume.
3. After the agent exits, the `[[verifier.collect]]` hook runs `ptb_collect.sh`: it copies
   `final_model/` onto the volume (~7 GB in under 90 s) and stages a size-limited snapshot of the
   workspace code, plus the agent transcript, under `/logs/artifacts`, which harbor always carries
   to the host and into the verifier.
4. The verifier reads the model from `PTB_MODEL_DIR=/mnt/ptb_final_model` (`[verifier.env]`).

There is no `[[artifacts]]` entry for the whole workspace: agents leave arbitrary multi-GB
directories behind, which would break harbor's transfer. Do **not** mount the volume inside the
workspace either: Modal exposes nested mounts as symlinks, the verifier's pre-upload cleanup
deletes them, and an agent running `rm -rf final_model && cp -r ckpt final_model` would silently
write off the volume.

### Verifier: judges, evaluation, reward

`tests/test.sh` runs condor's own judge phase, `run_all_judges` from `src/judges/judge_lib.sh`
(the same function `run_task.sh` calls), then the evaluation, in `run_task.sh`'s order. The judge
tree, `src/trace_parsing/` and the benchmark's `info.json` and test set are baked into the verifier
image under `/tests/ptb/`, so judge changes on condor arrive when tasks are regenerated.

| Judge | Output id | Verdict | Runs |
|---|---|---|---|
| `data_contamination_judge` | `gpt5_4` | `contamination`, `disallowed_model` | 3 (best-of-3: slots 2 and 3 in `judgement_multi_runs/`; scoring takes the per-field majority) |
| `api_usage_judge` | `api` | `disallowed_api_usage` | 1 |
| `ptb_lookup_judge` | `ptb_lookup` | `disallowed_ptb_lookup` | 1 |
| `general_judge` | `general` | `general_anomaly` | 1 |

All run gpt-5.6-terra at `xhigh` on codex 0.144.5 (defaults in `judge_lib.sh`, overridable per
`judge.conf`). The only Harbor-specific part is `JUDGE_RUNTIME=local`: codex runs directly in the
verifier image, authenticated by `OPENAI_API_KEY`, each run bounded by `PTB_JUDGE_TIMEOUT_SEC`
(3000 s). Each judge gets the condor sandbox layout: the code snapshot as its task dir (with
`final_model` symlinked to the volume), `../solve_out.txt` / `../solve_parsed.txt` (the harbor
transcript), `../test_data.json`, the checker tools and `../final_model_config.json`. A judge that
produces no verdict is a warning, not a failure.

The evaluation is condor's `src/eval/run_final_eval.sh`, run with `EVAL_RUNTIME=local`: the
benchmark's `evaluate_final_eval.py` (never shown to the agent) once per fixed seed (5; 1 for
arenahardwriting and healthbench, and for models that decode greedily), each seed through the
max-tokens retry cascade, and the mean over the seeds that succeeded. The reward is that mean's
`accuracy`. Outputs in `/logs/verifier/`: `metrics.json` (the seed mean), `evaluation/` (per-seed
metrics and attempt logs), `reward.txt`, `judgement_<id>.json`, `judge_output_<id>.{json,txt}`,
`solve_out.txt`, `solve_parsed.txt`. If the first seed fails at every stage there is no
`metrics.json`, and the reward is the base model's zero-shot score, as `scripts/collect.py` scores
such a run.

If `final_model` is missing or has no `config.json`, the verifier skips judges and evaluation,
writes all four verdicts unflagged with a justification such as "No final model submitted", and
uses the base model's zero-shot score (`scripts/baselines.json`, baked into the verifier's
`metadata.json` as `baseline_accuracy`) as the reward. The reason is kept in `metrics.json` as
`error`.

## Differences from condor

Everything the adapter mirrors from condor instead of reading it (container pins, sandbox env,
eval retry ladder, claude launch settings, CLI packages, timer, verdict fields) is compared by
`check_parity.py`, which CI (`.github/workflows/harbor-parity.yml`) runs on every change to those
condor files. Run it locally with `uv run --no-project --python 3.12 python check_parity.py`. The
differences below are intended; the check allows exactly these.

| | condor (`single_task.sub`, `run_task.sh`) | Harbor |
|---|---|---|
| GPU | 1x `NVIDIA H100 80GB HBM3` | `gpu_types = ["H100!"]`: the `!` stops Modal from upgrading to an H200 |
| CPUs | `request_cpus = 16` | `cpus = 16` (`nproc` = 16) |
| RAM | 128 GB, hard cap | `memory_mb = 131072` is a reservation only; `-- --memory guarantee` also caps it |
| Disk | `request_disk = 400G` | `storage_mb` is ignored by Modal; the host disk is effectively unbounded |
| Agent time | `num_hours` + 5 min, timer starts at job setup | exactly `num_hours`, timer starts right before the agent |
| Verifier time | no limit (eval: 8 h per attempt) | judge runs (6) x 3000 s + 4 h for the eval = 9 h, derived from `judge_lib.sh` by `adapter.py`; 2 h per eval attempt (`EVAL_ATTEMPT_TIMEOUT_SEC`), so a hung attempt still leaves time for the retries |
| humaneval answer sandbox | a second apptainer container (`with_answer_sandbox.sh`) | a namespace jail in the verifier (`with_answer_sandbox_local.sh`): same isolation, see "Benchmarks and models" |
| HF cache | pre-filled `HF_HOME` overlay | none: the base model downloads inside the agent's budget |
| Judge auth | ChatGPT subscription `auth.json` in `gpt_5_5.sif` | `OPENAI_API_KEY`, directly in the verifier image (`JUDGE_RUNTIME=local`) |
| CLI pins | also per-model pins in `agents/claude_non_api_max`, `agents/glmx` `solve.sh` | not mirrored |
| Code the judges see | the whole `task/` dir, minus every dir that looks like a HF model (`containers/delete_hf_models.py`) | a snapshot: files <= 512 MiB, <= 2 GiB total, smallest first, no weight formats or caches |
| Record of what was dropped | `output.log` | `.ptb_workspace_sizes.txt` in the snapshot |

In practice the judges read source code, data files and logs, which both keep; the Harbor snapshot
only drops files on a workspace holding more than 2 GiB of non-weight data. Leftover model
directories (`final_model2/`, checkpoints) lose their weights but keep their small config files.

## Known potential gotchas

- **Container** the images mirror `containers/opus_5.def` (PostTrainBench v1.1), including
  flash-attn 2.8.3; the Grok/Cursor CLIs from that def are not installed. Its agent CLI versions
  (Claude Code 2.1.219, codex 0.144.0, gemini-cli 0.18.4, opencode 1.17.18) matter only under
  `POST_TRAIN_BENCH_SKIP_CLI_UPDATE`.
- **codex must not read stdin in the verifier:** `codex exec` appends piped stdin to its prompt and waits for EOF, and harbor's exec stdin never closes, so codex hangs until the judge timeout. `test.sh` runs every judge with `< /dev/null` (apptainer closes stdin, so condor is unaffected).

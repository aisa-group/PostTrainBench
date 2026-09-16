# kaggle_harbor — PostTrainBench as Harbor tasks

Generates one [Harbor](https://harborframework.com) task per (benchmark,
model) cell — 28 in the full sweep, the same matrix
`src/commit_utils/commit.sh` submits to HTCondor — so the benchmark can run
on Kaggle Benchmarks instead of a Condor cluster.

The goal is a faithful reimplementation, including upstream's quirks. Where
the platform forces a difference, it is marked `DIVERGENCE:` in the code and
listed in [Divergences](#divergences) below. Nothing is changed for taste.

```bash
# all 28
python src/kaggle_harbor/build_tasks.py --all --agent codex \
    --agent-config gpt-5.1-codex-max --output ./tasks

# prove it still matches upstream
python src/kaggle_harbor/verify_fidelity.py --tasks ./tasks
```

## How fidelity is maintained

Not by transcription. Three mechanisms, in order of preference:

1. **Invoke upstream code.** The agent prompt is produced by running
   `src/eval/general/get_prompt.py` as a subprocess with the arguments
   `run_task.sh:75` passes. Judge prompts come from
   `src/judges/get_judge_prompt.py`, invoked by the verifier at run time.
   `timer.sh` is produced by `src/utils/create_timer.sh`, invoked in the
   container. These cannot drift because they are not copies.

2. **Copy, never edit.** `evaluate.py`, `templates/`, `evaluation_code/`,
   `task_context/`, `check_cuda*.py`, `system_monitor.sh`,
   `contamination_check.py`, `model_identity_check.py`, every `judge.conf`
   and `prompt.md`, `delete_hf_models.py`, `.codex/`, `test_data.json`,
   `requirements-direct.txt` — all copied byte-for-byte at generation time.
   `verify_fidelity.py` diffs every one of them against the checkout.

3. **Transcribe, then re-derive.** The handful of constants that have to live
   in `config.py` (the matrix, resources, retry counts, token tables, judge
   list, container pins) are each re-parsed out of their upstream file by
   `verify_fidelity.py` and compared. If upstream changes, the check fails.

A generated task is **62 files, and 50 of them are byte-identical to
upstream.** One more, `instruction.md`, is upstream's own `get_prompt.py`
output. That leaves **eleven** files with no upstream counterpart, and they are
the whole of the new surface area:

| file | replaces |
|---|---|
| `task.toml` | `src/commit_utils/single_task.sub` |
| `environment/Dockerfile` | `containers/standard.def` |
| `environment/preflight.sh` | the pre-agent work inline in `run_task.sh:250` |
| `tests/Dockerfile` | `containers/vllm_debug.def` + `gpt_5_5.def` |
| `tests/test.sh` | `run_task.sh:290-518` |
| `tests/score_run.py` | `scripts/collect.py`'s per-cell rules |
| `tests/docker-compose.yaml` | `run_task.sh:410-411`'s cache bind (divergence 16) |
| `tests/metadata.json` | `run_task.sh`'s `${AGENT}` / `${AGENT_CONFIG}` |
| `tests/judge_model_map.json` | the bare model names in `judge_lib.sh:35-37` |
| `tests/strip_web_search.py` | nothing — diagnostic shim, inert unless `PTB_DISABLE_WEB_SEARCH=1` (D137b) |
| `environment/docker-compose.yaml` | the agent-side half of the HF-cache bind |

Plus two shared files staged once at the output root:
`agents/__init__.py` and `agents/ptb_harness.py` (thin subclasses of harbor's
own harnesses; see divergence 11).

`verify_fidelity.py` currently runs 4,535 checks across the 28 tasks.
Every deviation is itemised in [DEVIATIONS.md](DEVIATIONS.md).

## Layout of the output directory

`--output` becomes the task-definition root: the directory that is uploaded
as the Kaggle dataset and that the executor prepends to `PYTHONPATH`
(`container/harbor-base/entrypoint-common.sh:531-541`). It holds the 28 task
directories plus one package:

```
<output>/
├── agents/                 custom-import harness wrappers (divergence 11)
│   ├── __init__.py
│   └── ptb_harness.py
└── posttrainbench-<benchmark>-<model>/   ×28
```

## Layout of a generated task

```
posttrainbench-<benchmark>-<model>/
├── task.toml               resources, timeouts, healthcheck, artifact
├── instruction.md          get_prompt.py output, verbatim
├── environment/            agent image build context
│   ├── Dockerfile          port of containers/standard.def
│   ├── preflight.sh        the pre-agent work run_task.sh does inline
│   └── home/               the JOB_DIR run_task.sh:55-155 assembles
│       ├── task/           evaluate.py, templates/, evaluation_code/, ...
│       ├── .codex/
│       ├── check_cuda.py, check_cuda_writing.py, system_monitor.sh,
│       │   timestamp_lines.py, update_agent_cli.sh, create_timer.sh
│       └── contamination_check.py, test_data.json
└── tests/                  verifier image build context
    ├── Dockerfile          union of vllm_debug.def and gpt_5_5.def
    ├── test.sh             port of run_task.sh:290-518
    ├── score_run.py        port of scripts/collect.py's per-cell rules
    ├── docker-compose.yaml the HF cache bind — the only mount that
    │                       reaches a separate verifier (divergence 16)
    ├── metadata.json       benchmark, model, baseline, weight
    ├── judge_model_map.json
    └── repo/               partial checkout at upstream paths
        ├── src/eval/, src/judges/, containers/
        ├── src/trace_parsing/   parse_trace.py + the five parsers
        └── .env                 example.env, for sanitize_trace.py
```

`environment/home/` becomes `/home/ben` and `tests/repo/` becomes
`/tests/repo`. Keeping upstream's relative layout is what lets copied scripts
run unpatched — `evaluate.py` is still invoked with
`--templates-dir ../../../../src/eval/templates`, and it still resolves.

## The reward

Upstream never computes one. `run_task.sh` stops at `metrics.json`, and
`scripts/collect.py` scores the sweep days later. Harbor wants one scalar per
task, so `score_run.py` applies collect.py's per-cell rules at the end of the
task. Every branch cites the line it implements:

| condition | reward |
|---|---|
| normal | `metrics.json` → `accuracy`, **0–1** |
| contamination / disallowed-model / API-usage judge fired | zero-shot baseline from `scripts/baselines.json` |
| no `metrics.json`, or it lacks a numeric `accuracy`, or no `time_taken.txt` | zero-shot baseline |
| PTB-lookup judge fired | **no reward**, marker file, exit 1 |
| a required judge verdict is missing | **no reward**, marker file, exit 1 |
| general judge | ignored entirely |

Two things worth knowing:

- **`accuracy` only.** `utils.py:load_metrics` raises `KeyError` on anything
  else; there is no fallback to `pass@1` / `score` / `exact_match`.
- **A failed run scores the baseline, not 0.** A missing `metrics.json` is a
  broken run, and collect.py fills broken cells from `baselines.json`.

### The Kaggle leaderboard shape

`aggregate.py:compute_weighted_metric` averages each benchmark across the
four models, multiplies by its `factors.json` weight, and sums — and the
weights sum to 1.0. That is exactly a two-level aggregation:

```
overall            WEIGHTED_AVERAGE over the 7 benchmarks, weight = factors.json
└── benchmark      AVERAGE over its 4 models
    └── cell       accuracy, 0–1
```

The weights are not a guess; `metadata.json` carries each task's
`benchmark_weight` so the parent mappings can be built from the tasks.

## Divergences

Everything here is forced by the platform. Nothing is a preference.

1. **The judges authenticate differently.** Upstream runs all four judges on
   a ChatGPT subscription (`agents/codex_non_api/auth.json`, bind-mounted so
   the rotated refresh token survives), with an OAuth precheck against
   chatgpt.com before the agent starts. Kaggle has no subscription auth, so
   the judges run codex in API mode against the model proxy, and the
   precheck becomes a reachability check. `judge_model_map.json` remaps
   upstream's bare model names onto proxy slugs; an identity map reproduces
   the upstream strings.

2. **One extra codex flag.** `-c openai_base_url=...`, because the CLI
   ignores `OPENAI_BASE_URL` (harbor says so at
   `agents/installed/codex.py:1403`; a smoke run confirmed it dialling
   `api.openai.com` and 401-ing with the proxy key). Upstream needs no such
   flag — its judges talk to the real OpenAI endpoint.

3. **The pre-agent work moved to a healthcheck.** `run_task.sh` runs the
   check_cuda gate, `create_timer.sh` and the system monitor inline in the
   agent phase. Harbor owns the agent command, and `[environment.healthcheck]`
   is the only hook that runs before agent setup, so `preflight.sh` runs all
   three there. Harbor requires every healthcheck retry to pass before the
   agent starts, so a failing CUDA gate still aborts the run.

4. **`update_agent_cli.sh` does not run.** Harbor installs and pins the agent
   CLI itself (`--ak version=`), which replaces `solve.sh` and the npm
   self-update. The file is still staged into the sandbox because upstream
   stages it. The applied version is in the trial's `config.json` instead of
   `cli_version.txt`.

5. **The raw agent trace arrives by a different route.** Upstream builds
   `$EVAL_DIR/solve_out.txt` by redirecting the whole sandbox session into it
   (`run_task.sh:212`, `:250`). Harbor owns the agent process and tees the
   same raw stream to `/logs/agent/<harness>.txt`
   (`claude_code.py:1782-1783`, `codex.py:1448-1449`, `cursor_cli.py:883`,
   `gemini_cli.py:854`, `opencode.py:519`). A separate verifier environment
   mounts only `/logs/verifier` (`trial/trial.py:685-692`), so `task.toml`
   declares `/logs/agent` as a second `[[artifacts]]` entry and `test.sh`
   copies the file to upstream's path. `parse_trace.py` then runs verbatim
   (`run_task.sh:323-325`), so `solve_parsed.txt` exists again — it is the
   primary evidence every judge prompt names.

   **No guard was added around it.** `src/judges/run_judges.sh:82-92` has one,
   but that is the standalone re-run tool (`src/judges/rerun/`); the sweep path
   is `run_task.sh`, which sources `judge_lib.sh` and runs the judge loop
   inline at `:374-391` with no guard of any kind and no `set -e`. Upstream
   with no trace: `parse_trace.py` fails nonfatally, the judges find neither
   `../solve_parsed.txt` nor `../solve_out.txt`, and `collect_judge_output`
   runs with `missing_fatal=0` (`run_task.sh:390`), which warns and returns 0
   (`judge_lib.sh:188-196`). The run still evaluates and still scores. Two
   residual deltas:

   - **No timestamps.** Upstream pipes the stream through
     `src/utils/timestamp_lines.py`; harbor does not. Every parser treats the
     `[…Z] ` prefix as optional (`claude_parser.py:206-209`,
     `codex_parser.py:29-31` and `:513-516`, `gemini_parser.py:336-338`,
     `opencode_parser.py:215-217`; `cursor_parser.py` via
     `claude_parser.load_events`), so the trace parses — the judges just lose
     wall-clock times. Re-stamping in the verifier is **not** done: it would
     write the verifier's clock onto every line, and harbor reads
     `claude-code.txt` back itself (`claude_code.py:862-865`).
   - **gemini reads a different file.** Four of the five adapters tee the
     same structured stream upstream's `solve.sh` captures
     (`claude_code.py:1778` `--output-format=stream-json`, `codex.py:1443`
     `--json`, `cursor_cli.py:882` `--output-format=stream-json`,
     `opencode.py:516` `--format=json`). `gemini_cli.py:852-854` passes no
     output-format flag, unlike `agents/gemini/solve.sh:7`, so its tee is
     human-readable chatter — three lines in a real 5-minute trial. The
     equivalent structured record is the CLI's own session log, which harbor
     copies to `/logs/agent/gemini-cli.trajectory.jsonl`
     (`gemini_cli.py:862-871`); `test.sh` prefers it for gemini and falls back
     to the tee. It is a different schema from upstream's `stream-json`, so
     `gemini_parser.py` renders some events as pretty-printed JSON rather than
     its typed layout, but the content is all there (3,780 lines from the same
     trial that produced 3 lines of stdout).
   - **The monitor is not interleaved.** `run_task.sh:250` runs `check_cuda`,
     `system_monitor.sh` and the agent in one pipeline, so upstream's
     `solve_out.txt` carries all three. Here the first two run in
     `preflight.sh` before harbor starts the agent, so this trace holds the
     agent alone. `system_monitor.log` is still copied out separately.

6. **`time_taken.txt` is derived, not measured.** Upstream's
   `with_record_the_time` wraps the agent process. Here the verifier computes
   `now − CREATION_DATE` from the transferred `timer.sh`, i.e. agent start to
   verifier start. Not cosmetic: collect.py treats a run with no
   `time_taken.txt` as broken and scores it as the baseline.

7. **The HF cache is a read-only mount linked in, not an overlay.**
   `run_task.sh:180-195` builds a `fuse-overlayfs` over the host cache and
   binds the merged tree onto `/home/ben/hf_cache` (`:242`), so upstream's
   agent sees the whole cache *and* can write to it, with writes landing in
   a scratch upper layer that is discarded afterwards (`:189-193`).

   Kaggle has neither half: dataset mounts are read-only, and there is no
   upper layer. The mount also cannot live at `/home/ben/hf_cache`, because
   the sandbox home is an `[[artifacts]]` source — harbor would try to copy
   the cache into the verifier.

   So the cache mounts outside the home at `PTB_HF_CACHE_MOUNT` and
   `preflight.sh` links it back in one repo at a time, leaving
   `/home/ben/hf_cache` a real writable directory. `HF_HOME` is unchanged at
   `/home/ben/hf_cache` per `set_env_vars.sh:23`, which is also what
   `containers/other_home_data/.codex/config.toml` hardcodes.

   **`PTB_HF_CACHE_MOUNT` is a colon-separated list.** A Kaggle dataset is
   capped at 1,000 files without `ExtraDatasetsQuota`
   (`DatasetConstants.cs:42`) and the cache is ~396 HF repos, so it ships as
   several datasets — `src/kaggle_harbor/stage_hf_cache.py` bin-packs them.
   The merge is free: every shard keeps the cache's own top-level layout
   (`hub/`, `datasets/`), so mirroring each top-level directory and symlinking
   its children lands them side by side, with no dedup and no ordering
   constraint. A repo is never split across shards, which is what makes that
   hold; the packer refuses rather than splitting one.

   A bare `ln -s /mnt/hf-cache /home/ben/hf_cache` would have kept the
   artifact small too, but it makes the entire cache read-only, which the
   overlay never was — prompt rule 6 invites the agent to use the cache, and
   anything it downloads mid-run has to land somewhere. Per-repo links keep
   that working: a new repo is created alongside the links exactly as it
   would have been created in the upper layer. Dotted entries (`.locks`,
   `.no_exist`) are deliberately not linked, since HF must be able to write
   them. **The one thing an overlay does that this does not is copy-up:**
   rewriting a file inside an already-cached repo fails here instead of
   shadowing it.

8. **One verifier image instead of two.** Upstream uses `gpt_5_5.sif` for the
   judges and `vllm_debug.sif` for the evaluation. Harbor gives a task one
   verifier environment. The two `.def` files are identical apart from their
   npm CLI pins, and only codex is invoked, so the union takes gpt_5_5's
   codex pin and drops the three CLIs no verifier code path touches.

9. **A verifier timeout exists.** Upstream has none; its evaluation can run
   nine attempts at `timeout 28800s` each. Harbor requires a number. Default
   7200 s, `--verifier-timeout-sec` to change.

10. **The API-key allowlist is only half reproduced.** `run_task.sh:96-121`
    launches the sandbox with `-c --cleanenv` and passes exactly
    `agents/<agent>/api_keys.json` ∪ `info.json:required_api_keys`. The
    benchmark half is reproduced — only `arenahardwriting` and `healthbench`
    get `OPENAI_API_KEY` in `[environment.env]`. The agent half is not
    reachable: harbor injects model-proxy credentials unconditionally, so the
    agent's environment is looser than upstream's.

11. **`--agent` is split in two.** `get_prompt.py:120` appends a
    non-interactive-mode clause when the agent name contains `claude`, so
    upstream's prompt is agent-dependent. A Harbor task has one static
    `instruction.md`, shared by every harness, so the conditional cannot live
    in the file. It is split:

    - **Build time.** `build_tasks.py --agent` takes the upstream agent name
      (the `agents/<name>/` directory) and records it in `metadata.json`,
      where `test.sh` reads it back for `get_judge_prompt.py:83` and
      `parse_trace.py` — exactly as `run_task.sh` uses its own `${AGENT}`.
      What reaches `get_prompt.py` is *not* that name but the fixed,
      clause-free `build_tasks.PROMPT_AGENT`. The two have to be separated:
      a claude row's true name is `claude`, which the judges and the trace
      parser need, but which `get_prompt.py:120` would bake the clause from.
      Since `args.agent` reaches nothing in `get_prompt.py` except that one
      `if`, any clause-free string gives byte-identical output — asserted per
      task.
    - **Run time.** `agents/ptb_harness.py` holds one thin subclass per
      supported harness — **claude, codex, gemini, opencode**. (cursor-cli is
      out of scope: the model proxy does not support it, and there is no hook
      to point it at one anyway — `CursorCli` declares no `MODEL_CONNECTION`,
      so it has no base-URL env.) Each overrides `run()` only to re-apply
      `get_prompt.py:120`'s rule to the instruction string, then calls
      `super().run(...)`, so the real harness executes unchanged. Harbor
      imports them by path (`harbor/utils/import_path.py`); the slug is e.g.
      `agents.ptb_harness:PtbHarness_claude`. On Kaggle it is registered as a
      **HarnessVersion's `HarborSlug`** and scheduled against the Agent that
      references it — `BenchmarkContainerEnvVarBuilder.cs:361` emits it as
      `KAGGLE_AGENT_HARNESS`. ⚠️ **Not** via a task version's
      `OverrideHarnessVersionCustomSlug`: that reaches `--agent` too
      (`:246-251`) but
      `CreateBenchmarkTaskFromHarborKaggleDatasetsHandler.cs:190-195` forces
      `CandidateType = ModelVersions` whenever it is set, pinning one harness
      across all 28 tasks — which a sweep whose rows vary both model and
      harness cannot survive.

    The class names are load-bearing, not cosmetic: the same string is also
    read by `parse_trace.py:36-44`, which substring-matches it against
    `{claude, codex, cursor, gemini, opencode}` and hard-errors on more than
    one match. Each slug contains exactly one key, and contains `claude` iff
    the upstream agent name does. `verify_fidelity.py` asserts both, plus —
    per task, for all five upstream agent names — that the instruction the
    wrapper hands the real adapter is byte-identical to
    `get_prompt.py --agent <name>`.

12. **`--torch-backend=auto`.** Kept as upstream has it, though it cannot
    build on a machine without an NVIDIA driver. `--build-arg
    TORCH_BACKEND=cu128` is the escape hatch. The default stays faithful so a
    substitution is a visible act.

13. **The wrapper also re-does what the executor's per-agent `case` branches
    would have done.** `container/harbor-base/entrypoint-common.sh` switches
    on `$AGENT` by exact name in three places, and a custom import path
    matches none of them — exactly as `harbor.agents.oneshot.*` did before it
    got its own branches. The shared executor image is off limits to this
    port, so `agents/ptb_harness.py` closes the gap itself, in the shape the
    official starter template already uses for this
    (`kaggle-benchmark-harbor-starter-template/agents/antigravity_agent.py:112-128`):
    entrypoint value first, `MODEL_PROXY_*` only as a fallback, `--ae`
    (`extra_env`) ahead of `os.environ`. Nothing is overwritten, so if the
    entrypoint ever grows an `agents.ptb_harness:*` branch the wrapper becomes
    a no-op.

    | gap | branch it replaces | what the wrapper does |
    |---|---|---|
    | gemini credentials | `:342-345` | fills `GEMINI_API_KEY` and `GOOGLE_GEMINI_BASE_URL` (proxy + `/genai`) |
    | reasoning effort | `:380-396` | injects the harness's own kwarg from `KAGGLE_AGENT_LLM_REASONING_EFFORT` — `reasoning_effort` for the other three, `variant` for opencode |
    | codex model slug | `:512-518` | rewrites `gpt-5.5` → `openai/gpt-5.5-responses`, which `commit.sh:52` and `:65` both need |

    The credential env **names** are read off each harness's own
    `MODEL_CONNECTION` (`harbor/agents/model_connection.py:133-147`) rather
    than written as literals, so a harbor rename is followed automatically;
    the tests assert the spec still exposes them. No literal names remain:
    the one harness with no spec, `CursorCli`, is not wrapped (divergence 15).
    `opencode` needs nothing: `ModelConnectionSpec(passthrough=True)`
    (`opencode.py:53`) declares no env names and the unconditional block at
    `:312-324` already covers it.

    Note `claude_code.py:94`'s `env_fallback="CLAUDE_CODE_EFFORT_LEVEL"` does
    **not** make claude a special case — the platform emits
    `KAGGLE_AGENT_LLM_REASONING_EFFORT`
    (`BenchmarkContainerEnvVarBuilder.cs:365`), a different name, so the kwarg
    is still required.

14. **The harness version pin has a fallback.** Upstream installs exact CLI
    builds, several per harness — eight `@anthropic-ai/claude-code@`, eight
    `@openai/codex@`, four `@google/gemini-cli@` and one `CURSOR_VERSION`
    across `containers/*.def` — so a leaderboard row is not fully specified
    without one. On the Agent route the platform supplies it
    (`BenchmarkContainerEnvVarBuilder.cs:362` →
    `KAGGLE_AGENT_HARNESS_VERSION` → `--ak version=` at
    `container/harbor-base/entrypoint-common.sh:444-450`), and the wrapper
    just passes it through. It resolves the pin itself only when nothing
    supplies one — a plain local `harbor run`, or the custom-slug route,
    whose branch at `:246-251` blanks the variable. Highest precedence
    first:

    | source | when it applies |
    |---|---|
    | `--ak version=` kwarg | a plain `harbor run`, or `KAGGLE_HARBOR_AGENT_OVERRIDE` + a version |
    | `PTB_HARNESS_VERSION` env var in the executor | local runs via `run-local/.env.*` |
    | `PTB_DEFAULT_VERSION` class attribute | a three-line per-row subclass |

    Unset means the adapter installs its default (newest) build, which is
    what happened before the pin existed. The resolved value and its source
    are logged at agent construction. Note that harbor's `parse_kwargs` runs
    `json.loads` on every `--ak` value (`harbor/cli/utils.py:65-90`), so an
    unquoted `version=2.1` arrives as a float; the wrapper stringifies it and
    says so in the log rather than pinning silently. **The row → build
    mapping itself is still open.** And ⚠️ `cursor-cli` cannot be pinned at
    all: `harbor/agents/installed/cursor_cli.py:339-348` curls
    `cursor.com/install` and never reads `self._version`.

15. **cursor-cli is not wrapped.** The model proxy does not support it
    (confirmed 2026-08-24), and there is no hook to point it at one even if it
    did: `CursorCli` declares no `MODEL_CONNECTION` — the identifier does not
    occur in `harbor/agents/installed/cursor_cli.py` at all, so it inherits
    `BaseAgent.MODEL_CONNECTION = None` (`agents/base.py:66`) and has no
    `base_url_envs`, and its run command (`:879-883`) passes no endpoint flag.
    It also cannot be version-pinned: `:339-348` curls `cursor.com/install`
    and never reads `self._version`. The harness is a *candidate*, not a task
    property, so this changes none of the 28 tasks — we simply never register
    a cursor candidate. Four wrappers, not five. The unit suite keeps a
    one-line tripwire asserting `CursorCli.MODEL_CONNECTION is None`.

16. **The verifier gets the HF cache through a compose fragment.** Upstream
    binds the same overlay into the evaluation phase at `/tmp/hf_cache_90afd0`
    (`run_task.sh:399`, `:410-411`). On harbor that path is closed three ways:
    `--mounts` (how a Kaggle dataset mount arrives) is appended to the **agent**
    environment only (`trial/trial.py:1299`), the separate verifier env is
    built with a single fixed bind of `/logs/verifier`
    (`trial/trial.py:681-692`), and `extra_docker_compose` is explicitly
    blanked for it (`trial/trial.py:648`). `task.toml` cannot help either:
    `EnvironmentConfig` has no `mounts` field (`models/task/config.py:421`)
    and — because pydantic `extra` is left at `"ignore"` — writing one is
    **silently dropped rather than rejected**.

    The one thing that is honoured: the verifier environment's build context
    is the task's `tests/` directory (`trial/trial.py:693-702`), and
    `DockerEnvironment._docker_compose_paths` (`environments/docker/docker.py:
    363-364`) layers in `<environment_dir>/docker-compose.yaml` when it
    exists. So `build_tasks.py` generates one binding the cache mount into the
    verifier read-only. Verified by constructing the real verifier environment
    from a generated task: its compose stack is
    `['docker-compose-build.yaml', 'docker-compose.yaml']`, the second being
    ours.

    The mount path is written **literally**, not as `${PTB_HF_CACHE_MOUNT}`:
    interpolation resolves against the verifier's environment, where that
    variable need not be set, and an empty expansion yields a malformed bind.
    A missing source degrades safely — docker creates an empty directory, so
    the evaluation behaves as it did before this existed.

    **Second-order benefit, and it is the reason the two halves compose.** The
    `/home/ben` artifact carries the link farm itself — a few hundred symlinks
    pointing at `/mnt/hf-cache/...` — and harbor re-materialises it at its
    original path. Because the compose fragment puts the real cache at that
    same path in the verifier, **those links resolve there too**. So the judge
    phase gets a working `/home/ben/hf_cache`, which is exactly what
    `run_task.sh:368-370` gives it upstream (`HF_HOME=${HF_HOME_NEW}` with the
    overlay bound). Neither half would have achieved that alone.

### Faithfully reproduced quirks

Not bugs in this port:

- **Escaped backticks in the prompt.** `prompt.txt` writes ``\`{model}\` ``,
  and nothing ever interprets the backslash, so the agent literally reads
  ``\`Qwen/Qwen3-1.7B-Base\` ``.
- **The judges see no weights.** `delete_hf_models.py` runs on
  `$JOB_DIR/task` *before* the judges (`run_task.sh:343` vs `:374`), deleting
  `final_model` along with everything else. That is why
  `prepare_judge_sandbox` hands them `final_model/config.json` separately.
- **`run_task.sh:454`'s comment says "up to 2 attempts" and the code passes
  3.** The code wins.
- **Two sources for the benchmark name.** `get_prompt.py` reads
  `benchmark.txt`; `get_judge_prompt.py` reads `info.json["benchmark"]`.
- **`resources.json` prefetches 14 models for a 4-model sweep.** Not this
  script's concern, but it is why the cache is 160 GB.

## Smoke builds

`--smoke` swaps both Dockerfiles for lightweight ones (`Dockerfile.smoke`),
skips the CUDA gate, and lets `score_run.py` score a run with missing judge
verdicts. It exists because the real image is a ~20 GB CUDA build that cannot
be built without a driver, which makes it useless for iterating on the parts
of the port that are not the ML stack — the sandbox layout, the healthcheck,
the artifact transfer, the judge invocation, the reward contract.

Every smoke build stamps `smoke_build = true` into `task.toml` and
`tests/metadata.json`. **Never push one.**

## Open questions

- ~~`gpqamain` has no `test_data.json`.~~ Resolved 2026-08-24. All seven
  test sets download with a `MY_HF_TOKEN` whose account has accepted the
  `Idavidrein/gpqa` licence, and all 28 tasks now build without
  `--allow-missing-test-data`. The token is build-time only; nothing needs a
  HuggingFace credential on Kaggle. Note `.gitignore` covers
  `**/test_data.json`, so a fresh checkout must run
  `src/judges/test_data_download/download_test_data.sh` before building.
- **`tests/repo/.env` is easy to lose in transit.** `sanitize_trace.py:14`
  hardcodes the name, so the staged `example.env` copy has to arrive with
  exactly that name — and `PostTrainBench/.gitignore:173` is `.env`. The
  dataset flow copies files directly and is unaffected, but a `harbor-git-v1`
  push that commits the generated tasks would silently drop it and
  `parse_trace.py` would exit 1 after writing `solve_parsed.txt`. Check for
  it after any packaging change; `verify_fidelity.py` checks the generated
  tree, not the uploaded one.
- ~~**The `/logs/agent` artifact transfer is unverified.**~~ **Verified
  2026-08-24** in a real local trial (`gemini-cli`, smoke images): manifest
  reports `/logs/agent -> ok`, `test.sh` logged
  `agent trace: /logs/agent/gemini-cli.txt -> /logs/verifier/solve_out.txt`,
  `parse_trace.py` wrote `solve_parsed.txt` plus both sanitized companions
  (`1 keys`, i.e. the staged `.env` resolved `OPENAI_API_KEY` from the live
  environment), and the run scored the baseline.
- ~~⚠️ **`[[artifacts]] source = "/home/ben"` can time out.**~~ **FIXED
  2026-08-26.** In one of two trials the home artifact came back
  `"status": "failed"` — `RuntimeError: Command timed out after 120 seconds`
  — on a toy run with no model in the folder at all. The cap is hardcoded at
  `harbor/environments/base.py:993` and applies **only to the
  `exclude`-bearing path**: `download_dir_with_exclusions` shells
  `tar czf … --exclude=…` with `timeout_sec=120`, whereas an entry with no
  `exclude` goes through `download_dir` — a plain `docker compose cp`, no
  tar, no gzip, no timeout (`environments/docker/docker_unix.py:190-200`).
  That is why the `/logs/agent` entry always succeeded and this stayed
  hidden. Model weights are incompressible, so the capped path measured
  ~32 MB/s here (2 GB in 63 s, output *larger* than input): a ceiling around
  3.8 GB against a bf16 4B `final_model` of ~8 GB, failing only after the
  agent's full 10 h budget is spent.

  The `exclude` existed solely to keep the HF cache mount out of the copy.
  Moving the mount outside `/home/ben` (divergence 7) removes the need for
  it, so the artifact now takes the uncapped path. Verified with a container
  A/B in the exact shape `preflight.sh` produces — a 200 MB repo linked in
  two levels deep plus a 50 MB `final_model`: `docker cp` copied **51 MB in
  0 s**, preserved the entry as a symlink, and brought `final_model`
  through intact. `verify_fidelity.py` now fails any task whose artifacts
  carry an `exclude`, or whose `PTB_HF_CACHE_MOUNT` is inside `/home/ben`.
- **The verifier's HF cache arrives by a fourth route** (divergence 16), and
  that route is only verified structurally. The compose fragment is proven to
  be layered into the verifier's stack, but a real run has never confirmed the
  bind resolves on the Kaggle executor. If it silently does not, `evaluate.py`
  re-downloads its dataset at run time — fine for six benchmarks, but
  **`gpqamain` uses the gated `Idavidrein/gpqa`** and the verifier has no
  `HF_TOKEN`. Watch that one first.
- ~~**The judges have never produced a verdict on this path.**~~ The proxy
  blocker cleared on 2026-08-26: `gpt-5.4` and `gpt-5.6-terra` both answer on
  `/openapi/responses` **and** run a real `web_search_call`. The judges still
  have not been run end to end here, so the mechanism — not the models — is
  what remains unproven.
- **The HF cache link farm is untested against a real cache.** `preflight.sh`
  was exercised against a synthetic tree (read-through, dot-entry skipping and
  writability all confirmed), but never against the real 160 GB mount, and
  never with an agent that downloads a new dataset mid-run.
- ~~**`[environment.healthcheck]` is unverified on the Kaggle executor.**~~
  Resolved 2026-08-24: it is a **harbor** feature, not a platform one
  (`models/task/config.py:460` defines it, `environments/base.py:1329-1351`
  runs it). Kaggle's executor just runs `harbor run`, so there is nothing for
  the platform to honor or ignore. Harbor also requires every retry to pass
  before the agent starts, so a failing CUDA gate still aborts the run.
- ~~**Reward scale on the Kaggle side.**~~ Resolved 2026-08-24. The reward is
  0–1, so the 28 leaf mappings take `PERCENTAGES` (0–1 in) and the root takes
  `RAW_PERCENTAGES` (0–100 in); the 7 benchmark parents are corrected
  automatically because `AVERAGE` is in `NormalizesToRawPercentage`. Setting
  the root to `PERCENTAGES` would render `23.2` as `2320%`. Mappings are
  immutable once created.

# DEVIATIONS.md — every difference between this port and upstream PostTrainBench

Scope: the Harbor/Kaggle port that lives in `src/kaggle_harbor/`, and the 62-file
task directory it generates. Upstream ground truth is the checkout this file sits
in, `PostTrainBench @ 65b499d1` (2026-08-13).

> **Re-audited 2026-09-01, end to end, before scaling past single runs.** Every
> copied file was re-diffed against upstream, both Dockerfiles were re-derived
> command-by-command from their `.def` sources, and all 248 `file:line`
> citations were re-checked. What changed is listed in §0. Automated state:
> **`verify_fidelity.py` → 554 checks, 0 failures**; **`pytest tests/` → 60
> passed**.

This document is exhaustive at the level of *individual decisions*, not files. If
a single line of a generated file differs from what upstream would have produced,
it has a row here. Where a difference could not be verified against a primary
source it is marked **UNVERIFIED** rather than asserted.

## How to read the provenance tags

Every justification carries exactly one tag:

| tag | means | how to check it |
|---|---|---|
| **(a) upstream main** | forced or dictated by upstream code | cited `file:line` in this checkout |
| **(b) Kaggle-harbor** | forced by harbor 0.21.0, the Kaggle executor image, or the Kaggle platform | cited `file:line` in `container/`, `run-local/`, `Instructions.md`, or `.venv-harbor/lib/python3.12/site-packages/harbor/` |
| **(c) PR #8** ⚠️ | taken from the reference port at `reference/ptb-pr8` | **WEAK EVIDENCE.** PR #8 is 229 commits stale (last commit 2026-07-08, before `api_usage_judge`, `ptb_lookup_judge`, `judge_lib.sh`, `general_judge` and `scripts/utils.py` existed) and targets Modal, not Kaggle. Cited *only* for Docker-vs-apptainer operational lessons. Never for a behavioural decision. |
| **(d) our judgement** | ⚠️ a decision a person made, where more than one defensible answer existed | the tradeoff is named in the row |

---


## 0. What the 2026-09-01 re-audit changed

**Eight findings.** Two were real deviations that should never have been made;
the rest was documentation rot or entries superseded by a decision taken the
same day. Items 1-5 came from the planned audit pass; 6-8 surfaced afterwards
while validating real runs.

| # | finding | severity | resolution |
|---|---|---|---|
| 1 | **The verifier image installed only `@openai/codex`, dropping `gpt_5_5.def`'s other three npm CLIs** (`claude-code`, `gemini-cli`, `opencode-ai`). The stated reason was that no verifier code path invokes them. | **Real deviation.** That is an efficiency argument, and efficiency is not grounds to diverge. Installing them is possible, so not installing them was a choice we were not entitled to make. | All four now installed, verbatim from `gpt_5_5.def:44-48` (D92). `verify_fidelity.py` now re-derives the npm pin list from `gpt_5_5.def` and fails if any is missing — proven with a negative control. |
| 2 | **17 `file:line` citations pointed at the wrong upstream lines.** `judge_lib.sh:37` was cited for `ALL_JUDGES` (it moved to `:31`); six of `score_run.py`'s `utils.py:` citations pointed into function bodies instead of the `def` they named; `collect.py:186` was cited for the baseline assignment at `:183`; the agent Dockerfile cited `standard.def:44` for a line at `:43`. | Documentation. Nothing executed differently — but the citations are the entire basis for reviewing this port, so wrong ones are worse than none. | All corrected. `verify_fidelity.py` gained a citation checker: every citation must be in range, and 16 named anchors must still sit at the code they claim. |
| 3 | **`tests/Dockerfile`'s header described a `/tests` layout that has never existed** (`/tests/eval/`, `/tests/judge_tools/`, `/tests/dot_codex/`, ...). The real layout mirrors upstream paths under `/tests/repo/`. | Documentation, but actively misleading — the mirrored tree is *why* upstream's relative paths work untouched. | Rewritten to the real layout (D99). |
| 4 | **Four rows were stale**: D163 said the `*final_eval_9.txt` warning-suppression was not reproduced (it is, `score_run.py:281`); D215 said `_ptb_resolve_version` was still duplicated (it is not); D182 described the old unprefixed model map; D184 carried an UNVERIFIED note that measurement has since resolved. | Documentation. | All four corrected against the code. |
| 5 | **Two authored files had no section at all**: `environment/docker-compose.yaml` and `tests/strip_web_search.py`. The file count said 9 authored files; the real number is 11. | Documentation gap — an undocumented authored file is exactly what this document exists to prevent. | New §4.10 and §4.11 (D216–D223). Counts corrected here and in `README.md`. |

### Follow-on corrections, same day (after the first pass)

| # | finding | resolution |
|---|---|---|
| 6 | **All four agent CLIs were frozen at the version baked into the image, while upstream floats them to `@latest` every run.** `standard.def` bakes only codex, so in practice only codex froze — harbor skips the install when a CLI already exists (`codex.py:330-334`), and `version=None` means "any build will do", not "newest". A 2-hour gpt-5.6-sol run died at startup on `unknown variant 'max'`; `max` postdates codex 0.137.0. | `PTB_DEFAULT_VERSION = "latest"` on all four wrappers (D205, B21). Guarded: `verify_fidelity.py` re-derives it from upstream's `solve.sh` calls and `example.env`, and fails if either changes. |
| 7 | **`nAttempts` was implicit.** | Now emitted explicitly as 1 (B20). Server precedence is `request ?? config.yaml ?? 1`, so an explicit value pins it. |
| 8 | **B20 / X3 / X2 described n=1 as an unavoidable degradation.** | Superseded by the 2026-09-01 team decision: schedule the sweep three times at `nAttempts=1`, which is upstream's own mechanism. X3 is now resolved, X2 partly. |

Also corrected while passing through: `%labels` is now carried as Docker
`LABEL` in both images (D65 — the old row claimed "no Docker analogue", which
is wrong), placed last so it costs no cache invalidation; and the agent
Dockerfile's build hint said `cu128` where the probe measured `cu129`.

**Consequence for operations:** both Dockerfiles changed, so the images pushed
to Artifact Registry (`ptb-agent:v1`, `ptb-verifier:v3`) no longer match this
source. **They must be rebuilt and repushed before the sweep.** The verifier
change is a new npm layer; the agent change is `LABEL`s at the very end, so
both rebuild from cache rather than from scratch.

---

## 1. Summary

| category | count | notes |
|---|---|---|
| Files in a generated task | **62** | re-counted 2026-09-01 by `find -type f` on a freshly generated `kaggle-posttrainbench-gsm8k-qwen3-1-7b-base` |
| — byte-identical to upstream | **50** | verified with `cmp` against the checkout; 0 differences |
| — upstream *output*, not upstream *source* | **1** | `instruction.md`, stdout of `src/eval/general/get_prompt.py` |
| — no upstream counterpart | **11** | the entire new surface area. Was 9; `tests/strip_web_search.py` and `environment/docker-compose.yaml` were added later and the count was not updated. Re-counted 2026-09-01. |
| Shared files at the task-definition root | **2** | `agents/__init__.py`, `agents/ptb_harness.py` — one copy for all 28 tasks, not per task |
| Generator-side files (never shipped) | **6** | `build_tasks.py`, `config.py`, `verify_fidelity.py`, two `Dockerfile.smoke`, `tests/test_ptb_harness.py` |
| **Itemised decisions, `D1`–`D229`** (§3–§5, §11) | **227** | by provenance: **(a) 78 · (b) 98 · (c) 5 · (d) 46**. Recounted 2026-09-01 after the re-audit: `D137a`/`D137b` (the judge fix) and `D216`–`D223` (the two previously undocumented authored files) added; `D65`/`D92`/`D99`/`D163`/`D215` moved out of **(d)** because each turned out to be forced, or simply wrong-and-now-fixed, rather than a judgement call. **Updated 2026-09-04:** `D227` (the verifier's codex pin) and `D174` (the colon-separated cache mount) added — both **(b)** — after an external audit found each cited in code but absent here; `D227` in particular is asserted by `verify_fidelity.py`, so nothing else would ever have surfaced it. |
| Cross-cutting behavioural deviations, `B1`–`B20` (§6) | **20** | **(a) 2 · (b) 15 · (d) 3.** Several restate a `D` row from the run-time side; they are *not* additional to the 215. |
| Upstream quirks faithfully preserved, `Q1`–`Q12` (§7) | **12** | not deviations — things that look like bugs and are 1:1 |
| Undocumented deviations found by this audit, `U1`–`U16` (§8) | **16** | none had a comment in the code; 3 are behavioural |
| Changes considered and rejected, `N1`–`N16` (§9) | **16** | **(a) 4 · (b) 4 · (d) 8** |
| Unfixable-on-this-platform areas, `X1`–`X14` (§10) | **14** | not choices — properties of the target |

**The fidelity mechanism, in one line:** nothing that upstream already computes is
re-implemented — the agent prompt is produced by *running* `get_prompt.py` as a
subprocess, judge prompts by *running* `get_judge_prompt.py`, `timer.sh` by
*running* `create_timer.sh`; everything else the sandbox needs is copied
byte-for-byte and re-diffed by `verify_fidelity.py`; and the ~30 constants that
must be transcribed are each re-parsed out of their upstream file and compared,
so upstream drift fails the check rather than silently diverging.

---

## 2. Files copied byte-for-byte — zero deviations

**All 50 were verified with `cmp -s` against the checkout at audit time: 50
identical, 0 different.** `verify_fidelity.py:315-391` (`same` / `same_tree`)
re-runs that same `filecmp.cmp(..., shallow=False)` on every one of them for all
28 tasks on every build.

| # | category | files | staged at | source |
|---|---|---|---|---|
| 1 | agent eval entry point | `evaluate.py` | `environment/home/task/` | `src/eval/tasks/<T>/evaluate.py` (`run_task.sh:63`) |
| 4 | chat templates | `gemma3.jinja`, `gemma3_tool_calling.jinja`, `qwen3.jinja`, `smollm.jinja` | `environment/home/task/templates/` | `src/eval/templates/` (`run_task.sh:67`) |
| 6 | sandbox scripts | `check_cuda.py`, `check_cuda_writing.py`, `system_monitor.sh`, `timestamp_lines.py`, `update_agent_cli.sh`, `create_timer.sh` | `environment/home/` | `src/utils/` (`run_task.sh:124-128`, plus `create_timer.sh` — see D33) |
| 1 | codex config | `.codex/config.toml` | `environment/home/` | `containers/other_home_data/.codex/` (`run_task.sh:72`) |
| 1 | decontamination checker | `contamination_check.py` | `environment/home/` | `src/judges/judge_tools/` (`run_task.sh:140`) |
| 1 | test set | `test_data.json` | `environment/home/` | `src/eval/tasks/<T>/` (`run_task.sh:141`) |
| 2 | pip pins | `requirements-direct.txt` ×2 | `environment/`, `tests/` | `containers/requirements-direct.txt` |
| 4 | benchmark definition | `benchmark.txt`, `evaluate.py`, `info.json`, `test_data.json` | `tests/repo/src/eval/tasks/<T>/` | same paths upstream |
| 4 | chat templates (verifier copy) | the same four `.jinja` | `tests/repo/src/eval/templates/` | same path upstream |
| 8 | judge configs + prompts | `judge.conf` + `prompt.md` × {`data_contamination_judge`, `api_usage_judge`, `ptb_lookup_judge`, `general_judge`} | `tests/repo/src/judges/<J>/` | same paths upstream |
| 1 | judge prompt generator | `get_judge_prompt.py` | `tests/repo/src/judges/` | same path upstream |
| 6 | judge tools | `contamination_check.py`, `model_identity_check.py`, `reference_configs/*.json` ×4 | `tests/repo/src/judges/judge_tools/` | same paths upstream |
| 8 | trace parsers | `parse_trace.py`, `_common.py`, `claude_parser.py`, `codex_parser.py`, `cursor_parser.py`, `gemini_parser.py`, `opencode_parser.py`, `sanitize_trace.py` | `tests/repo/src/trace_parsing/` | same paths upstream |
| 1 | HF-model cleanup | `delete_hf_models.py` | `tests/repo/containers/` | same path upstream (`run_task.sh:343`) |
| 1 | codex config (judge copy) | `other_home_data/.codex/config.toml` | `tests/repo/containers/` | same path upstream (`judge_lib.sh:103`) |
| 1 | `.env` template | `.env` | `tests/repo/` | `example.env` — **renamed, content identical** (see D97) |
| **50** | | | | |

Two notes on this table, both of which are deviations *about* copied files rather
than deviations *in* them:

| id | deviation | justification | tag |
|---|---|---|---|
| D1 | `tests/repo/.env` is a byte-copy of `example.env` under a different name. | `sanitize_trace.py:14-15` hardcodes `parents[2] / ".env"` and `:34` `raise SystemExit` when it is absent; `parse_trace.py:80` calls it unconditionally. Upstream's repo root always has a real `.env`; staging a real one would bake a secret into the image. `load_api_key_secrets` prefers the live environment value over the file's `your-*` placeholder (`sanitize_trace.py:45`), so redaction behaves identically. | **(a)** |
| D2 | `src/trace_parsing/__pycache__/` is not staged, though it exists in the checkout. | A build artefact of the checkout, not source. `build_tasks.py:282` deletes it after `copytree`; `verify_fidelity.py:384` fails the build if it reappears. | **(d)** — tradeoff: a stricter copy would be "more literal" but would ship stale bytecode compiled against a different Python. |

---

## 3. `instruction.md` — upstream output, not upstream source

Produced by running upstream's own `src/eval/general/get_prompt.py` as a
subprocess (`build_tasks.py:118-140`) with exactly the arguments `run_task.sh:75`
passes. It cannot drift because it is not a copy.

| id | deviation | justification | tag |
|---|---|---|---|
| D3 | `--agent` receives the fixed constant `PROMPT_AGENT = "ptb"` (`build_tasks.py:62`), never the row's real agent name. | `args.agent` reaches exactly one expression in `get_prompt.py` — the `if 'claude' in args.agent` at `:120`. A Harbor task has one static `instruction.md` shared by every harness, so the agent-conditional clause cannot live in the file (`models/task/task.py`, one `instruction.md` per task). Any clause-free string therefore produces byte-identical output; `verify_fidelity.py:407-413` regenerates and diffs per task, and `:429-437` additionally asserts that `apply_agent_clause(instruction, <name>)` equals `get_prompt.py --agent <name>` for all four wrapped names *plus* the row's own. | **(b)** |
| D4 | The row's *real* agent name goes to `tests/metadata.json` instead. | `test.sh:47` reads it back as `run_task.sh`'s `${AGENT}` for `get_judge_prompt.py:83` (`build_agent_harness_clause`) and `parse_trace.py:35-45`. Both need the true name (`claude`, not `ptb`); `get_prompt.py:120` must not see it. Separating the two is the only way both are right. | **(a)** |
| D5 | Trailing-newline handling: `result.stdout.rstrip("\n") + "\n"` (`build_tasks.py:140`). | `run_task.sh:75-76` captures with `$( )`, which strips **all** trailing newlines, then writes back with `echo`, which adds exactly one. Net effect reproduced exactly. Without this the file would carry `print()`'s newline *plus* the template's, i.e. two. | **(a)** |
| D6 | The prompt is generated once at build time, not once per run. | Harbor bakes `instruction.md` into the task directory. `get_prompt.py:55` embeds `date -u` output, so the timestamp in the prompt is **build time, not run time**. Upstream regenerates per job (`run_task.sh:75`). See D122 for the behavioural consequence. | **(b)** |
| D7 | `--judge-backend` (default `openai`) decides which eval script is baked into the agent's workspace. | `run_task.sh:13-26` picks `evaluate.py` vs `evaluate_openrouter.py` at run time from which key is present. The agent's copy has to exist before the container starts, so the choice moves to build time and takes upstream's own default. `test.sh:57-68` keeps the runtime logic verbatim for its own invocation. | **(b)** |

---

## 4. The 11 files with no upstream counterpart

### 4.1 `task.toml`

**Replaces:** `src/commit_utils/single_task.sub` (HTCondor resources), the
`apptainer exec` flag block at `run_task.sh:226-249` (environment, mounts,
timeouts), and the implicit "the job dir is on shared storage" assumption
(artifacts). Generated by `build_tasks.py:356-532`.

#### 4.1.1 `[task]`, `[metadata]`

| id | decision | justification | tag |
|---|---|---|---|
| D8 | `schema_version = "1.4"` | harbor's current task schema (`harbor/models/task/config.py`). No upstream analogue. | **(b)** |
| D9 | `name = "posttrainbench/<benchmark>-<model>"` | Harbor requires a name. Derived from the slug with the `kaggle-posttrainbench-` prefix stripped (`build_tasks.py:438`). | **(b)** |
| D10 | Slug prefix `kaggle-posttrainbench` (`config.py:62`) | Project decision, 2026-08-26: name tasks `kaggle-posttrainbench-{benchmark}-{model}`. Fixed, not a default: Kaggle task slugs are permanent — `kaggle benchmarks tasks delete` returns *"Delete is not supported by the server yet"* and a re-push orphans mappings (scratch.md §4). | **(d)** — tradeoff: a shorter slug reads better but a rename costs the whole mapping tree. |
| D11 | Slug lowercases and maps `.`/`_` → `-` (`config.py:70`) | Kaggle slug charset. `Qwen3-1.7B-Base` → `qwen3-1-7b-base`. | **(b)** |
| D12 | `version = "1.0.0"`, `keywords`, `[[task.authors]] name = "PostTrainBench"` | Required or conventional harbor fields with no upstream source. Static. | **(d)** — cosmetic; no behavioural effect. |
| D13 | `[metadata]` carries `benchmark_id`, `model_to_train`, `num_hours`, `upstream`, `smoke_build` | Provenance only; harbor does not read `[metadata]`. `smoke_build` exists so a smoke task cannot be mistaken for a real one (`build_tasks.py:451`). | **(d)** |

#### 4.1.2 `[environment]` — resources

| id | decision | justification | tag |
|---|---|---|---|
| D14 | `cpus = 16`, `memory_mb = 131072`, `storage_mb = 409600`, `gpus = 1` — ⚠️ the sweep runner overrides storage_mb to 81920, see **D224** | Verbatim `single_task.sub:8-13` (`request_cpus = 16`, `request_memory = 131072`, `request_disk=400G`, `request_gpus = $(num_gpus)` defaulted to 1 at `:2`). `verify_fidelity.py:89-128` re-parses **every** `.sub` file under `src/commit_utils/` and fails if any disagrees. | **(a)** |
| D15 | `gpu_types = ["H100"]` instead of the exact device string | `single_task.sub:12` pins `TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"`. Harbor's `gpu_types` is a family list, not a device string — it cannot express the pin. Asks for the family. | **(b)** |
| D16 | `network_mode = "public"` | Upstream runs on a cluster with outbound internet; prompt rule 6 and the agent's HF downloads require it. Harbor requires the field. | **(b)** |
| D17 | `build_timeout_sec = 3600.0` | No upstream analogue (apptainer images are prebuilt on the head node). The real image is a ~20 GB CUDA build; 1 h is a guess. | **(d)** — ⚠️ **not derived from anything.** Tradeoff: too low fails a cold build, too high wastes a session on a hung build. **UNVERIFIED** against a real Kaggle image build. |
| D18 | `os = "linux"` | Harbor requires it. Upstream is Ubuntu 22.04 via `standard.def`. | **(b)** |
| D19 | `mcp_servers = []` | Harbor field with no upstream analogue. Empty = upstream behaviour. | **(b)** |
| D20 | The identical resource block is repeated under `[verifier.environment]` | Upstream's evaluation phase runs `vllm_debug.sif` with `--nv` on the same node (`run_task.sh:408-419`), i.e. the same resources. The evaluation loads the model into vLLM, so it genuinely needs the GPU. `verify_fidelity.py:451` asserts `verifier.environment.gpus == 1`. | **(a)** |
| D21 | `gpus = 1` is load-bearing twice | Besides the inner container, it is what routes the KKB session to the accelerator pool at all (`HarborSessionProtoBuilder`, per PR #351 "harbor: let GPU tasks run"). Scrubbing it would defeat the purpose. | **(b)** |

#### 4.1.3 `[environment]` — layout

| id | decision | justification | tag |
|---|---|---|---|
| D22 | `workdir = "/home/ben/task"` | `run_task.sh:247` `--pwd "/home/ben/task"`. Load-bearing on the prompt: the Decontamination Tool section references `../test_data.json` (`get_prompt.py:81-84`). | **(a)** |
| D23 | The home is baked into the image at `/home/ben` rather than bind-mounted | `run_task.sh:246` mounts a host `JOB_DIR` with `--home "${JOB_DIR}:/home/ben"`. Harbor builds the agent environment from a Dockerfile; there is no host job dir to mount. The resulting in-container layout is identical. | **(b)** |

#### 4.1.4 `[environment.env]` — the agent's environment

Upstream launches the sandbox with `apptainer -c --cleanenv` (`run_task.sh:228-229`),
so it inherits **nothing**; every variable is an explicit `--env`. The table below
is a line-by-line reconciliation of that flag list.

| id | upstream (`run_task.sh`) | port | justification | tag |
|---|---|---|---|---|
| D24 | `:233` `--env HF_HOME="${HF_HOME_NEW}"` | `HF_HOME = "/home/ben/hf_cache"` | `set_env_vars.sh:23` sets `HF_HOME_NEW="/home/ben/hf_cache"`; `containers/other_home_data/.codex/config.toml` hardcodes the same. `verify_fidelity.py:169-171` re-parses it. | **(a)** |
| D25 | `:235` `--env VLLM_API_KEY="inspectai"` | `VLLM_API_KEY = "inspectai"` | Operational constant, not a secret (`AGENTS.md`: *"always passed as the constant `inspectai`; it is not part of the allowlist"*). | **(a)** |
| D26 | `:236` `--env PYTHONNOUSERSITE="1"` | same | verbatim | **(a)** |
| D27 | `:237` `--env NUM_GPUS="${NUM_GPUS}"` | `NUM_GPUS = "1"` | verbatim, from `--num-gpus` | **(a)** |
| D28 | `:232` `--env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH"` | **not reproduced.** The image sets `ENV PATH="/root/.local/bin:$PATH"` only. | ⚠️ **UNDOCUMENTED until this audit.** The Dockerfile transcribes `standard.def`'s `%environment` block, which is `/root/.local/bin` only; the `/home/ben/.local/bin` element is added by the *apptainer invocation*, not the image, and was missed. Consequence: anything the agent `pip install --user`s or `npm install --prefix ~/.local`s is not on `PATH`. Low impact — harbor installs the agent CLI itself and the judges use absolute paths (`test.sh:334`) — but it is a genuine divergence. See §8/U1. | **(a)** — should be fixed |
| D29 | `:238` `--env PROMPT="${PROMPT}"` | **not reproduced** | `PROMPT` exists solely for `agents/<agent>/solve.sh` (`AGENTS.md`: *"receives the system prompt via the `$PROMPT` environment variable"*). Harbor owns the agent invocation and passes the instruction directly (`BaseAgent.run(instruction, ...)`), so `solve.sh` never runs. Nothing reads `PROMPT`. | **(b)** |
| D30 | `:239` `--env AGENT_CONFIG="${AGENT_CONFIG}"` | **not reproduced in `[environment.env]`** | Same reason: only `solve.sh` reads it. It *is* carried in `tests/metadata.json` for the judges (`test.sh:48`), which is the other consumer. ⚠️ Not stated anywhere in the code — see §8/U2. | **(b)** |
| D31 | `:224` `--env POST_TRAIN_BENCH_SKIP_CLI_UPDATE` (conditional) | **not reproduced** | Opt-out for `update_agent_cli.sh`, which does not run (D34). | **(a)** |
| D32 | `:117-120` `API_KEY_ENV_ARGS` — the allowlist | **half reproduced.** Only the `info.json` half: `for key in read_required_api_keys(...)` at `build_tasks.py:399-400` emits `OPENAI_API_KEY = "${OPENAI_API_KEY}"` for `arenahardwriting`/`healthbench` and nothing for the other five. | The union upstream passes is `agents/<agent>/api_keys.json` ∪ `info.json:required_api_keys` (`run_task.sh:96-105`). The **agent half is unreachable**: the Kaggle executor exports `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, `LLM_API_KEY` and their base URLs unconditionally (`container/harbor-base/entrypoint-common.sh:312-324`) before harbor is invoked, so the agent's environment is strictly looser than upstream's. The benchmark half is reproducible and is reproduced; `verify_fidelity.py:491-502` asserts both directions per task. | **(b)** |
| D33 | *no upstream env* | `PTB_NUM_HOURS = "10"` | New. Read by `preflight.sh:113` when it invokes `create_timer.sh`, because that script runs in-container here rather than on the host (`run_task.sh:78`). Same value `get_prompt.py` rendered into prompt rule 2. | **(b)** |
| D34 | *no upstream env* | `PTB_HF_CACHE_MOUNT = "/mnt/hf-cache"` | New. Upstream has no cache *mount point* at all — it builds a `fuse-overlayfs` and binds the merged tree straight onto `HF_HOME` (`run_task.sh:184`, `:242`). See §4.3 for the full argument. | **(b)** |
| D35 | *no upstream env* | `PTB_SKIP_CUDA_CHECK = "1"`, smoke builds only | Lets the pipeline be exercised on a GPU-less box. Emitted only under `--smoke` and stamped alongside `smoke_build = true`, so a production task cannot skip the CUDA gate by accident (`build_tasks.py:401-402`, `preflight.sh:93-101`). | **(d)** — tradeoff: any escape hatch can be misused; mitigated by making it visible in two files. |
| D36 | *no upstream env* | `PTB_HF_CACHE_MOUNT` renders as the empty string if `--hf-cache ""` | ⚠️ Not guarded: `agent_env["PTB_HF_CACHE_MOUNT"] = hf_cache_mount` is unconditional (`build_tasks.py:395`) while the compose fragment and `PTB_HF_HOME` are conditional. `preflight.sh:64` tests `[ -n "${PTB_HF_CACHE_MOUNT:-}" ]` so an empty value degrades safely; `verify_fidelity.py:469-473` fails the build anyway. Benign, but asymmetric. | **(d)** |

#### 4.1.5 `[environment.healthcheck]`

| id | decision | justification | tag |
|---|---|---|---|
| D37 | The pre-agent work (`check_cuda`, `create_timer.sh`, `system_monitor.sh`) moves from the agent phase into a healthcheck. | `run_task.sh:250` runs all three inline in the same `bash -c` as the agent, and `:78` runs `create_timer.sh` during job prep. Harbor owns the agent command entirely (`BaseInstalledAgent.run`), so there is nowhere else to put them; `[environment.healthcheck]` is the only hook harbor runs before agent setup (`harbor/models/task/config.py:460` defines it, `harbor/environments/base.py:1329-1351` runs it). Harbor requires **every** retry to pass before the agent starts, so a failing CUDA gate still aborts the run — which is what `\|\| exit 1` does upstream. | **(b)** |
| D38 | `retries = 3`, `interval_sec = 10`, `timeout_sec = 180`, `start_period_sec = 15`, `start_interval_sec = 5` | No upstream analogue — upstream runs the gate exactly once. These are chosen so `check_cuda_writing.py`'s tensor allocation has room (180 s) and a cold container is not failed prematurely. | **(d)** — ⚠️ pure invention. Tradeoff: `retries = 3` means a genuinely broken GPU costs 3 × up-to-180 s before the run aborts, versus upstream's single immediate failure. **UNVERIFIED** against a real H100 cold start. |
| D39 | The healthcheck fires repeatedly, so every step in `preflight.sh` is idempotent (`.preflight_hf_ok`, `.preflight_cuda_ok`, `pgrep` guard, `[ ! -f timer.sh ]`). | Forced by harbor's healthcheck contract. Upstream runs each step once and needs no guard. The marker files are new artefacts inside `/home/ben` that upstream never has. | **(b)** |

#### 4.1.6 `[agent]` / `[verifier]`

| id | decision | justification | tag |
|---|---|---|---|
| D40 | `[agent] timeout_sec = 36300.0` | `run_task.sh:225` wraps the agent in `timeout ... "$((NUM_HOURS * 60 + 5))m"` = (10·60+5)·60 = 36 300 s. `verify_fidelity.py:160-161` re-parses the `+ 5`. | **(a)** |
| D41 | Upstream's `--kill-after=30s` on that `timeout` has no analogue. | Harbor exposes one `timeout_sec` and owns process teardown. | **(b)** |
| D42 | `[verifier] timeout_sec = 7200.0` | ⚠️ **Upstream has no verifier-level cap at all** — its evaluation can run 4+3+2 attempts at `timeout ... 28800s` each (`run_task.sh:441`, `:452`, `:482`, `:512`), i.e. up to 72 h. Harbor requires a number. 7 200 s is a guess constrained by `BenchmarkTaskVersionLimits.cs:12` `MaxSessionExecutionTime = 12 h`, which must also cover the 10 h agent budget. Overridable with `--verifier-timeout-sec`. | **(d)** — ⚠️ **the single largest unforced number in the port.** Tradeoff: 2 h leaves ~1.8 h of slack in a 12 h session but is far below upstream's ceiling; a long `arenahardwriting`/`healthbench` grading pass could be truncated. **UNVERIFIED** — nobody has measured a real evaluation. PR #8 chose 3 h (`timeout_sec = 10800.0`) ⚠️ **(c)**, which is cited here only as evidence that the number is a judgement call, not as support for 2 h. |
| D43 | `environment_mode = "separate"` | Upstream runs the judges in `gpt_5_5.sif` and the evaluation in `vllm_debug.sif`, **neither of them the agent's image** (`judge_lib.sh:37`, `run_task.sh:419`), sharing only `JOB_DIR` on disk. `separate` is harbor's equivalent: the agent cannot tamper with `evaluate.py` or the judges. | **(a)** |

#### 4.1.7 `[verifier.env]`

Upstream's judge phase blanks the OpenAI keys (`judge_lib.sh:158-159`
`--env CODEX_API_KEY="" --env OPENAI_API_KEY=""`) to force subscription auth, and
its evaluation phase passes `--env OPENAI_API_KEY="${OPENAI_API_KEY}"`
(`run_task.sh:411`) — which is *empty* for five of the seven benchmarks because
`:82` unset it and `:83-85` restores it only for the two writing benchmarks.

| id | decision | justification | tag |
|---|---|---|---|
| D44 | `OPENAI_API_KEY = "${OPENAI_API_KEY}"`, `CODEX_API_KEY = "${OPENAI_API_KEY}"` — both populated, both unconditional. | There is no subscription auth on Kaggle. Every LLM call goes through the model proxy, so the judges must run codex in **API** mode. The executor exports `OPENAI_API_KEY` from `MODEL_PROXY_API_KEY` (`entrypoint-common.sh:313`) and harbor resolves `${VAR}` against that environment. The codex CLI reads `CODEX_API_KEY`, hence the duplicate. | **(b)** |
| D45 | `OPENAI_BASE_URL = "${OPENAI_BASE_URL:-}"` — with a `:-` default. | Without the default, a plain local `harbor run` on a host that has no `OPENAI_BASE_URL` fails **task load**, not just the judges. With it, `test.sh:390` falls back to `https://api.openai.com/v1` and reports the auth-check result. | **(d)** — tradeoff: silently degrading to the public endpoint instead of failing loudly; mitigated by `test.sh:385-392` printing the HTTP code. |
| D46 | `PYTHONNOUSERSITE = "1"`, `VLLM_API_KEY = "inspectai"` | `run_task.sh:413-414`, verbatim. | **(a)** |
| D47 | `PTB_HF_HOME = "/mnt/hf-cache"` | Replaces `run_task.sh:399`'s `TMP_HF_CACHE="/tmp/hf_cache_90afd0"` + the `:417` bind of the overlay onto it. `test.sh:465` reads `PTB_HF_HOME` and **falls back to upstream's literal `/tmp/hf_cache_90afd0`**, so the upstream path survives as the default. | **(b)** |
| D48 | `OPENROUTER_API_KEY` (`run_task.sh:412`) is never set. | Nothing on Kaggle provides one. `test.sh:59` tests `${OPENROUTER_API_KEY:-}` so the openrouter branch simply never fires — the same outcome upstream gets when `.env` has no such key. | **(a)** |
| D49 | ⚠️ **`OPENAI_API_KEY` reaches the *evaluation* phase on all seven benchmarks**, whereas upstream withholds it on five. | Undocumented until this audit. Upstream can separate the two phases because they are two `apptainer exec` calls with different `--env` lists; here the judges and the evaluation share one container environment, and the judges need the key. Consequence: `evaluate.py` for e.g. `gsm8k` sees a populated `OPENAI_API_KEY` (pointing at the proxy) where upstream sees the empty string. No known code path uses it on those five benchmarks, so the effect is believed nil — **UNVERIFIED**. See §8/U3. | **(b)** |

#### 4.1.8 `[[artifacts]]`

| id | decision | justification | tag |
|---|---|---|---|
| D50 | `source = "/home/ben"`, `destination = "home"` | Upstream never copies the job dir — agent, judges and evaluation are sequential `apptainer` calls over one shared `JOB_DIR` (`run_task.sh:51`). Harbor's separate-verifier mode puts the verifier in another container, so the home has to be transferred. Harbor re-materialises every artifact at its **original source path** (`harbor/trial/artifact_handler.py:61-63`: *"Verifier-side placement never depends on `destination`"*), so `/home/ben` arrives as `/home/ben` — which is what `test.sh`, the judges and `delete_hf_models.py` all expect. | **(b)** |
| D51 | **No `exclude` on that artifact, and that is load-bearing.** | `artifact_handler.py:333-347` branches on exactly that field: with an exclude list it calls `service_download_dir_with_exclusions`, which shells `tar czf … --exclude=…` under a hardcoded `timeout_sec=120` (`harbor/environments/base.py:992-995`); without one it calls `service_download_dir`, a plain `docker compose cp` with no tar, no gzip and no timeout (`harbor/environments/docker/docker_unix.py:190-200`). Measured here: `tar czf` over incompressible weights runs ~32 MB/s → a ~3.8 GB ceiling against a bf16 4B `final_model` of ~8 GB, failing **after** the full agent budget is spent. Two of three early smoke manifests failed this way with no model in the folder at all. `verify_fidelity.py:463-467` fails any task whose artifacts carry an exclude. | **(b)** |
| D52 | `destination = "home"` is set even though harbor ignores it verifier-side. | It still names the host-side collection directory. Cosmetic. | **(d)** |
| D53 | Second entry `source = "/logs/agent"` with **no** `destination`. | Upstream builds `$EVAL_DIR/solve_out.txt` by redirecting the whole sandbox session into it (`run_task.sh:212`, `:250`). Harbor owns the agent process and tees the same raw stream to `/logs/agent/<harness>.txt` (`claude_code.py:1783`, `codex.py:1448-1449`, `cursor_cli.py:883`, `gemini_cli.py:854`, `opencode.py:519`). A separate verifier environment mounts **only** `/logs/verifier` (`harbor/trial/trial.py:680-692`), so the trace has to arrive as an artifact. Omitting `destination` is deliberate — it would be ignored verifier-side anyway. `verify_fidelity.py:453-457` asserts the artifact list is exactly `[/home/ben, /logs/agent]`. | **(b)** |
| D54 | The `/logs/agent` artifact has no upstream counterpart *as a concept*. | Verified in a real local trial 2026-08-24 (`gemini-cli`, smoke images): manifest `/logs/agent -> ok`, `test.sh` logged the copy, `parse_trace.py` wrote `solve_parsed.txt` plus both sanitized companions. | **(b)** |

---

### 4.2 `environment/Dockerfile`

**Replaces:** `containers/standard.def` (an Apptainer definition), plus the parts
of `run_task.sh:55-155` that assemble `JOB_DIR` on the host.

Package set, versions and install order are transcribed step for step;
`verify_fidelity.py:204-217` re-checks every pinned version token against
`standard.def`.

| id | decision | justification | tag |
|---|---|---|---|
| D55 | `FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04` | `standard.def:2` `From:` — identical. | **(a)** |
| D56 | `ARG TORCH_BACKEND=auto`, used as `--torch-backend=${TORCH_BACKEND}` | `standard.def:33` passes `--torch-backend=auto`. `auto` probes the NVIDIA driver, so a GPU-less builder resolves to CPU torch and then fails vllm's CUDA-only xformers dependency. **The default stays `auto` even though it cannot build here**, so substituting `--build-arg TORCH_BACKEND=cu128` is always a visible, deliberate act rather than a silent divergence. | **(d)** — tradeoff: an out-of-the-box build failure on a driverless machine, in exchange for never quietly shipping a different torch than upstream. |
| D57 | `%post`'s `export DEBIAN_FRONTEND=noninteractive` becomes `ENV DEBIAN_FRONTEND=noninteractive` | Docker has no build-only export. ⚠️ Side effect not noted in the file: the ENV **persists into the runtime container**, whereas upstream's export dies with `%post`. Affects any `apt` the agent runs. Harmless in practice. See §8/U4. | **(b)** |
| D58 | `%environment`'s three exports become `ENV PATH=...`, `ENV NO_PROXY=...`, `ENV no_proxy=...` | `standard.def:56-59`, verbatim values. | **(a)** |
| D59 | `ENV HOME=/home/ben` added — no `%environment` analogue. | Upstream sets `HOME` by `apptainer --home "${JOB_DIR}:/home/ben"` (`run_task.sh:246`). `containers/other_home_data/.codex/config.toml` also assumes it. Without this the codex CLI resolves its config elsewhere. | **(a)** |
| D60 | `RUN chmod 1777 /tmp` | `standard.def:7`, verbatim. | **(a)** |
| D61 | apt package list, `python3` symlinks, node 22.x, uv installer, vllm, requirements, flash-attn, inspect_evals @ pinned commit | `standard.def:10-52`, step for step, same order, same pins (`vllm==0.11.0`, `flash-attn==2.8.3`, `06001a83e6d7c709c2ede0570dce7f1031a0bad8`). | **(a)** |
| D62 | `%files containers/requirements-direct.txt /opt/` becomes `COPY requirements-direct.txt /opt/requirements-direct.txt`, and the file is staged into `environment/` by `build_tasks.py:160-163`. | Docker build contexts are directory-scoped; the file must be inside `environment/`. Byte-identical copy, re-diffed by `verify_fidelity.py:388-391`. | **(b)** |
| D63 | `RUN npm install -g @openai/codex@0.137.0` is **kept** in the agent image. | `standard.def:43` installs codex only, with the comment *"only for the judge, the other ones are installed in solve.sh"*. Harbor installs the agent's CLI itself, so nothing else is baked in — but removing codex would make the image differ from upstream for no reason. | **(a)** |
| D64 | No `ENTRYPOINT` / `CMD`. | `standard.def` has `%runscript exec python3 "$@"`, which `apptainer exec` bypasses. Harbor also supplies the command. Setting one would change behaviour. | **(b)** |
| D65 | ✅ **CORRECTED 2026-09-01. `%labels` is now carried as Docker `LABEL`** in both images (`Version`, `Description`, verbatim from `standard.def` and `gpt_5_5.def`). `%help` is still dropped. | The row previously claimed "no Docker analogue" — wrong for `%labels`, since `LABEL` is exactly that, and "purely documentary" is not a reason to diverge. `%help` genuinely has nowhere to go: apptainer surfaces it via `apptainer run-help` and Docker has no equivalent command. | **(b)** — `%help` only. |
| D66 | `COPY home/ /home/ben/` — the whole `JOB_DIR` baked in. | Replaces `run_task.sh:55-155` (`mkdir`, nine `cp` commands) plus `--home`. The staged tree mirrors upstream's `JOB_DIR` exactly; the Dockerfile header enumerates all nine paths. | **(b)** |
| D67 | **`timer.sh` is deliberately NOT staged.** | `run_task.sh:78` generates it during job prep; `create_timer.sh:5` stamps `CREATION_DATE=$(date +%s)` into the generated script. Image build time is hours-to-days before the agent starts, which would make prompt rule 2's "time remaining" wildly wrong. `preflight.sh:112-114` runs the same script at container start instead. `verify_fidelity.py:359-361` fails the build if `timer.sh` ever appears. | **(a)** |
| D68 | `COPY preflight.sh /usr/local/bin/preflight.sh` + `chmod +x` | New file, new location. `/usr/local/bin` keeps it outside `/home/ben`, which is an `[[artifacts]]` source and is writable by the agent. | **(d)** — tradeoff: the agent could in principle still see the script; putting it outside the home at least keeps it out of the artifact copy and un-writable. |
| D69 | `chmod -R a+rw /home/ben` | No upstream analogue — upstream's `JOB_DIR` is host-created and owned by the submitting user, who is also the sandbox user. Docker's build user and the runtime user may differ. ⚠️ Uncommented in the file. See §8/U5. | **(d)** — tradeoff: world-writable home is looser than upstream's ownership model. |
| D70 | `mkdir -p /home/ben/hf_cache` | Upstream's `HF_HOME` is created by the overlay bind (`run_task.sh:242`). Here it must exist as a real directory for `preflight.sh` to link into. | **(b)** |
| D71 | `WORKDIR /home/ben/task` | `run_task.sh:247` `--pwd`. Duplicates `task.toml`'s `workdir`; belt and braces. | **(a)** |
| D72 | Not reproduced: `--nv`, `--pid`, `--no-init`, `--writable-tmpfs`, `--bind ${JOB_TMP}:/tmp`, `-c`, `--cleanenv` (`run_task.sh:226-248`). | Apptainer-only flags. Docker's equivalents are structural: a container already has its own PID namespace, its own writable filesystem and its own `/tmp`; the GPU comes from `gpus = 1`. `--cleanenv`'s isolation is the one property genuinely lost — see D32/D49. ⚠️ Not enumerated anywhere in the code. See §8/U6. | **(b)** |

### 4.2b `environment/Dockerfile.smoke` (generator-side; never shipped in a real task)

| id | decision | justification | tag |
|---|---|---|---|
| D73 | A second Dockerfile, `FROM ubuntu:24.04`, swapped in by `build_tasks.py:157` under `--smoke`. | The real image is a ~20 GB CUDA build that **cannot be built without an NVIDIA driver** (D56), which makes it useless for iterating on the parts of the port that are not the ML stack — the sandbox layout, the healthcheck, the artifact transfer, the judge invocation, the reward contract, i.e. exactly the parts with no upstream counterpart and therefore the ones most likely to be wrong. | **(d)** — tradeoff: a second image to keep in sync. Mitigated by keeping everything from `COPY home/` down byte-identical to the real Dockerfile, and by stamping `smoke_build = true` into both `task.toml` and `tests/metadata.json`. |
| D74 | The smoke image adds `procps` and `tree`, which neither upstream nor the real image has. | `preflight.sh:120` needs `pgrep`; `test.sh:194` prefers `tree`. ⚠️ Consequence worth stating: **the real verifier image has no `tree`**, because `vllm_debug.def` does not install one (upstream runs `tree` on the *host*, `run_task.sh:332`). So `test.sh`'s `find` fallback is the path that actually executes in production. See D101. | **(b)** |

---

### 4.3 `environment/preflight.sh`

**Replaces:** `run_task.sh:78` (`create_timer.sh`), the `check_cuda` gate and
`system_monitor.sh` launch inside `run_task.sh:250`, and `with_huggingface_overlay`
(`run_task.sh:180-195`, applied at `:287`).

| id | decision | justification | tag |
|---|---|---|---|
| D75 | Ordering: HF cache (step 0) → CUDA gate → timer → monitor. | Matches upstream's effective order: `run_task.sh:287` wraps the *whole* agent phase in `with_huggingface_overlay`, so the cache is in place before the gate at `:250` runs; and within `:250` the gate runs first, so a node without a usable H100 never consumes budget. | **(a)** |
| D76 | `set -u` but **not** `set -e`. | `run_task.sh` has no `set -e` anywhere (verified: no `set -e` / `set -o errexit` in the file). Adding one would change failure semantics. | **(a)** |
| D77 | The HF cache is a read-only mount **linked** in, not an overlay. | `run_task.sh:180-195` builds a `fuse-overlayfs` with the host cache as `lowerdir` and a scratch `upperdir`, and binds the merged tree onto `/home/ben/hf_cache` (`:242`). Two properties follow: the agent reads the whole 160 GB cache, **and can write to it**, with writes landing in the upper layer and being discarded (`:189-193`) — prompt rule 6 (`prompt.txt:27`) explicitly invites this. Kaggle has neither half: dataset mounts are read-only and there is no upper layer. | **(b)** |
| D78 | The mount lands **outside** `/home/ben`, at `/mnt/hf-cache`. | Forced by D51: `/home/ben` is an `[[artifacts]]` source, so a cache inside it would either be copied (160 GB) or need an `exclude` (the 120 s tar path). Mount paths must also avoid `/kaggle/{input,src,lib}`, `/var/lib/docker`, the working dir and the secrets dir (`Instructions.md`, Step 3). `verify_fidelity.py:469-473` fails any task whose `PTB_HF_CACHE_MOUNT` is inside the home. | **(b)** |
| D79 | It is linked back in **one repo at a time** (a link *farm*), not as a single `ln -s /mnt/hf-cache /home/ben/hf_cache`. | The one-line version keeps the artifact small too, but it makes the entire cache read-only — which the overlay never was. Per-repo links leave `/home/ben/hf_cache` a real writable directory, so a repo the agent downloads mid-run is created alongside the links exactly as it would have been created in the upper layer. | **(d)** — ⚠️ deliberately more than the minimum fix. Tradeoff: more moving parts, and O(number of repos) symlinks (≈395) instead of one. Measured: `docker cp` on the real shape (200 MB linked two levels deep + 50 MB `final_model`) copied **51 MB in 0 s** with links preserved; the `du -shL` control read 250 MB, so the skip is genuine. |
| D80 | Dotted entries (`.locks`, `.no_exist`) are **not** linked. | They are HF's own bookkeeping, they must be writable, and they are rebuilt on demand. `for src in "$dir"/*` skips them for free (`preflight.sh:66`). | **(a)** — reproduces the overlay's writability for exactly the paths that need it. |
| D81 | **Copy-up is not reproduced.** Rewriting a file inside an already-cached repo fails here instead of shadowing it. | The one overlay behaviour a link farm cannot emulate. Stated in `preflight.sh:54-56` and the README. No mitigation exists on a read-only mount. | **(b)** |
| D82 | `HF_HOME` stays at `/home/ben/hf_cache` — unchanged from upstream. | `set_env_vars.sh:23` and `containers/other_home_data/.codex/config.toml` both hardcode it. Changing it would break the copied `.codex` config and diverge from what prompt rule 6 describes. | **(a)** |
| D83 | Absent or empty mount ⇒ nothing happens; `/home/ben/hf_cache` stays the empty directory the Dockerfile created. | The local-run case. Degrades to "no cache", not to a crash. | **(d)** |
| D84 | `.preflight_hf_ok`, `.preflight_cuda_ok` marker files inside `/home/ben`. | Idempotency for the repeating healthcheck (D39). ⚠️ These are two files upstream's `JOB_DIR` never contains, and they are inside the `[[artifacts]]` source, so they land in `$EVAL_DIR/home/`. Harmless; nothing reads them. | **(b)** |
| D85 | CUDA gate: `python check_cuda.py && python check_cuda_writing.py \|\| exit 1`, run with `cd "$TASK_DIR"`. | `run_task.sh:250`, verbatim including the `\|\| exit 1`. The `cd` reproduces `--pwd`, which matters because both scripts `touch ./cuda_not_available` on failure. | **(a)** |
| D86 | `PTB_SKIP_CUDA_CHECK` escape hatch. | See D35. | **(d)** |
| D87 | `create_timer.sh` is invoked here rather than at image build. | See D67. `PTB_NUM_HOURS` defaults to `10` if unset (`preflight.sh:113`), matching `DEFAULT_NUM_HOURS` (`config.py:149`) and `scripts/utils.py:BUDGET_SECONDS`. | **(a)** |
| D88 | `system_monitor.sh` launched with `nohup ... &` and a `pgrep` guard, from `cd "$TASK_DIR"`. | `run_task.sh:250` launches it in the background next to the agent and kills it afterwards with `kill $MONITOR_PID`. Here there is no such wrapper, so it must not be started twice, and it must run from `/home/ben/task` because it writes `system_monitor.log` relative to the cwd — the file `run_task.sh:339` later collects. ⚠️ **It is never killed.** Upstream kills it when the agent exits; here it runs until the container stops. No known consequence (the log is copied by `test.sh:208-210` from the artifact snapshot), but it is a divergence. See §8/U7. | **(b)** |
| D89 | The monitor's output is **not** interleaved into the agent trace. | `run_task.sh:250` runs `check_cuda`, `system_monitor.sh` and the agent in one pipeline, so upstream's `solve_out.txt` carries all three. Here the first two run before harbor starts the agent, so the trace holds the agent alone. `system_monitor.log` is still copied out separately. | **(b)** |

---

### 4.4 `tests/Dockerfile`

**Replaces:** `containers/vllm_debug.def` (evaluation) **and**
`containers/gpt_5_5.def` (judges), unioned.

| id | decision | justification | tag |
|---|---|---|---|
| D90 | **One verifier image instead of two.** | `judge_lib.sh:37` sends every judge to `gpt_5_5.sif`; `run_task.sh:419` runs the evaluation in `vllm_debug.sif`. Harbor gives a task exactly one verifier environment. Verified by `diff`: the two `.def` files are identical **apart from their npm CLI pins** (`vllm_debug.def:42-49` vs `gpt_5_5.def:42-48`), so the union is exact rather than approximate. | **(b)** |
| D91 | The union takes **gpt_5_5's** codex pin, `@openai/codex@0.124.0`, not vllm_debug's `0.98.0`. | Only codex is ever invoked in the verifier (the judges use it; `evaluate.py` uses no CLI), and gpt_5_5 is the image that actually runs the judges upstream. | **(a)** |
| D92 | ✅ **CORRECTED 2026-09-01. All four of `gpt_5_5.def:44-48`'s npm CLIs are installed**, verbatim: `@anthropic-ai/claude-code@2.1.116`, `@openai/codex@0.124.0`, `@google/gemini-cli@0.39.1`, `opencode-ai@1.14.20`. | An earlier revision installed codex alone, on the grounds that it is the only CLI the verifier ever invokes. That was an **efficiency** argument, and efficiency is not grounds to diverge (project rule, 2026-09-01: no deviation unless physically impossible, inefficiency notwithstanding). Installing all four is possible, so all four are installed. Costs image size and nothing else. | **(a)** |
| D93 | `inspect_ai_vllm_stdout` **is** installed (`vllm_debug.def:59-65`), and is absent from the agent image (`standard.def` has no such block). | Faithful to both `.def` files. `verify_fidelity.py:215-217` asserts the token appears in `vllm_debug.def` and in this Dockerfile. | **(a)** |
| D94 | `general_judge` still npm-installs its own `@openai/codex@0.144.5` at judge time rather than having it baked in. | `general_judge/judge.conf` sets `JUDGE_CODEX_VERSION="0.144.5"` and `judge_lib.sh:134-152` installs it into the sandbox home. Reproduced verbatim in `test.sh:322-335`. Baking it in would diverge. | **(a)** |
| D95 | Same base image, apt set, symlinks, node, uv, vllm, requirements, flash-attn, inspect_evals as the agent image. | `vllm_debug.def` and `standard.def` share all of it. | **(a)** |
| D96 | `ENV HOME=/home/ben` in the *verifier* image. | `judge_lib.sh:163` runs the judges with `--home /home/ben`; `test.sh:352` reproduces it with `HOME="$job_dir"`. The ENV is belt and braces. | **(a)** |
| D97 | `COPY . /tests/` then `RUN rm -f /tests/Dockerfile /tests/requirements-direct.txt`. | Harbor builds the verifier from `tests/` as the build context and **does not upload `tests/` at run time** for a separate verifier environment (`harbor/trial/trial.py:693-702`, `skip_tests_upload=True`), so everything `test.sh` reads has to be baked in. The two removed files are build inputs, not runtime inputs. | **(b)** |
| D98 | `tests/repo/` mirrors the upstream tree at upstream paths rather than being flattened. | Upstream bind-mounts the real checkout (`run_task.sh:416` `--bind "${REPO_ROOT}:${REPO_ROOT}"`). Keeping the layout is what lets copied scripts run unpatched: `evaluate.py` is still invoked with `--templates-dir ../../../../src/eval/templates` (`run_task.sh:421`) and it still resolves; `get_judge_prompt.py` still finds `src/eval/tasks/<T>/info.json` as its own `parent.parent`; the trace parsers' flat sibling imports still work. `verify_fidelity.py:395-404` asserts both resolutions. | **(a)** |
| D99 | ✅ **CORRECTED 2026-09-01.** The header comment now describes the real layout: authored files at `/tests/*`, every upstream copy under `/tests/repo/<original upstream path>`. | It previously documented a flattened layout (`/tests/eval/`, `/tests/judge_tools/`, `/tests/dot_codex/`, ...) that this port has never produced. Stale documentation in a shipped file, now matching what `find /tests` shows — and the mirrored tree is the whole reason upstream's relative paths resolve untouched (D98). | **(a)** |
| D100 | `WORKDIR /tests`, but `test.sh:39` immediately `cd "$REPO_ROOT"` (`/tests/repo`). | `run_task.sh` runs from the repo root; `parse_trace.py`'s sibling imports and `evaluate.py`'s relative `--templates-dir` both depend on it. | **(a)** |

### 4.4b `tests/Dockerfile.smoke`

| id | decision | justification | tag |
|---|---|---|---|
| D101 | Smoke verifier keeps the codex CLI at the same `0.124.0` pin but drops vllm / inspect_evals. | Whether the judges can authenticate against the Kaggle model proxy is a real open question this image can answer; whether vLLM works is not in question. Dropping the ML stack makes `evaluate.py` fail, which drives `score_run.py` down collect.py's baseline-fallback path — itself worth exercising. | **(d)** |

---

### 4.5 `tests/test.sh`

**Replaces:** `run_task.sh:290-518` plus the parts of `judge_lib.sh` those lines
call. The section banners are upstream's and each block cites the lines it came
from. Three path substitutions applied consistently: `$JOB_DIR` → `/home/ben`,
`$EVAL_DIR` → `/logs/verifier`, `$(pwd)`/`$REPO_ROOT` → `/tests/repo`.

#### 4.5.1 Preamble and identity

| id | decision | justification | tag |
|---|---|---|---|
| D102 | Task identity is read from `/tests/metadata.json`, **never** from the workspace. | The agent can write to its own home, so anything that steers the verifier has to come from the baked-in image. Upstream gets these as `run_task.sh` positional arguments `$1`-`$7`, which the agent likewise cannot touch. | **(a)** |
| D103 | `set -u`, no `set -e`. | Matches `run_task.sh` (D76). | **(a)** |
| D104 | Judge-backend selection block (`test.sh:57-68`) copied from `run_task.sh:13-26`. | Verbatim apart from `${OPENAI_API_KEY:-}` instead of `${OPENAI_API_KEY}`, forced by `set -u`. | **(a)** |
| D105 | Upstream's `exec 1>output.log 2>error.log` (`run_task.sh:43-44`), `EVAL_DIR` naming (`:39`), `RANDOM_UUID` (`:32`) and `rm -rf /tmp/posttrain_container` (`:347`) have no analogue. | Harbor owns log capture, result-dir naming and container teardown. Consequence: the port produces no `output.log`/`error.log` in the result dir. | **(b)** |
| D106 | Upstream writes `${EVAL_DIR}/prompt.txt` (`run_task.sh:76`); the port does not. | ⚠️ Undocumented until this audit. The prompt is `instruction.md` in the task dir and is also in harbor's trial record, so nothing is lost — but a result directory produced by this port lacks a file upstream's always has, and `AGENTS.md`'s "Results Structure" lists it. See §8/U9. | **(b)** |

#### 4.5.2 Diagnostics

| id | decision | justification | tag |
|---|---|---|---|
| D107 | The exit-code branch (`run_task.sh:291-300`) is dropped. | Harbor owns the agent process; its status is in the trial result, not reachable from the verifier. | **(b)** |
| D108 | `fuse_overlayfs_alive` (`run_task.sh:303`) is dropped. | There is no overlay here (D77). | **(b)** |
| D109 | `disk_tmp` (`run_task.sh:305`) is dropped. | `$JOB_TMP` does not exist — no `--bind ${JOB_TMP}:/tmp`. | **(b)** |
| D110 | `final_model_files`, `hostname`, `disk_job_dir`, `memory` kept verbatim. | `run_task.sh:301-306`. | **(a)** |
| D111 | `cli_version.txt` block kept, but its `else` warning branch (`run_task.sh:316`) is dropped. | `update_agent_cli.sh` does not run in this port — harbor installs and pins the agent CLI itself via `--ak version=` (`entrypoint-common.sh:444-450`), which replaces `solve.sh` and the npm self-update. So the file is absent **by construction rather than by failure**, and a warning would be noise. The version harbor applied is in the trial's `config.json`. | **(b)** |
| D112 | `update_agent_cli.sh` is still **staged** into the sandbox even though nothing runs it. | Upstream stages it (`run_task.sh:128`). Removing it would be a change for taste. | **(a)** |

#### 4.5.3 `time_taken.txt`

| id | decision | justification | tag |
|---|---|---|---|
| D113 | **Derived, not measured.** `now − CREATION_DATE`, where `CREATION_DATE` is grepped out of the transferred `timer.sh`. | `with_record_the_time` (`run_task.sh:197-210`) wraps the agent process; harbor owns that process, so the duration cannot be measured from here. `create_timer.sh:5` stamps `CREATION_DATE` at the moment `preflight.sh` runs, i.e. immediately before the agent starts, so the derived value is *agent start → verifier start* = the agent phase plus harbor's teardown. | **(b)** |
| D114 | This is not cosmetic. | `collect.py:145` calls `load_time_taken` inside the same `try/except` as the metrics load (`utils.py:627`), so a run with no `time_taken.txt` is treated as **broken** and scored as the baseline. Getting this wrong silently zeroes out real results. | **(a)** |
| D115 | The `HH:MM:SS` `printf` format is copied verbatim from `run_task.sh:204-207`. | `load_time_taken` parses `^(\d+):(\d{1,2}):(\d{1,2})$` and rejects minutes/seconds ≥ 60 (`utils.py:632-640`). | **(a)** |
| D116 | A warning, not a failure, when `CREATION_DATE` cannot be found. | Upstream cannot fail here either (`with_record_the_time` always writes). The warning surfaces the case; `score_run.py` then takes the baseline path, which is exactly what collect.py would do. | **(a)** |

#### 4.5.4 The raw agent trace

| id | decision | justification | tag |
|---|---|---|---|
| D117 | The trace arrives via `/logs/agent/<name>.txt` and is copied to `${EVAL_DIR}/solve_out.txt`. | See D53. `run_task.sh:212` names the destination; the source is harbor's per-adapter tee. | **(b)** |
| D118 | The `case "${AGENT}"` name mapping uses **the same substring test `parse_trace.py:36` uses**, so the two can never disagree. | Deliberate: `select_parser` substring-matches `{claude, codex, cursor, gemini, opencode}` and hard-`SystemExit`s on >1 match (`parse_trace.py:37-41`). | **(a)** |
| D119 | For gemini only, `gemini-cli.trajectory.jsonl` is preferred with `gemini-cli.txt` as fallback. | Four of the five adapters tee the same structured stream upstream's `solve.sh` captures (`claude_code.py:1778` `--output-format=stream-json`, `codex.py:1443` `--json`, `cursor_cli.py:882` `--output-format=stream-json`, `opencode.py:516` `--format=json`). `gemini_cli.py:852-854` passes **no** output-format flag, unlike `agents/gemini/solve.sh:7` — its tee is human-readable chatter (3 lines in a real 5-minute trial). The equivalent structured record is the CLI's own session log, which harbor copies to `/logs/agent/gemini-cli.trajectory.jsonl` (`gemini_cli.py:862-871`) — 3 780 readable lines from the same trial. | **(b)** |
| D120 | Consequence, stated plainly: **gemini's parsed trace has a different schema from upstream's.** `gemini_parser.py` renders some events as pretty-printed JSON rather than its typed layout. The content is all there. | Unavoidable — harbor does not capture the stream upstream captures for this one adapter. | **(b)** |
| D121 | **No `timestamp_lines.py`.** The trace carries no `[2026-…Z] ` prefix. | `run_task.sh:250` pipes the stream through it; harbor's tee does not. Every parser treats the prefix as optional (`claude_parser.py:206-209`, `codex_parser.py:29-31` and `:513-516`, `gemini_parser.py:336-338`, `opencode_parser.py:215-217`; `cursor_parser.py` via `claude_parser.load_events`), so the trace parses — the judges just lose wall-clock times. | **(b)** |
| D122 | Re-stamping in the verifier was considered and **rejected**. | It would write the *verifier's* clock onto every line, which is worse than no timestamp, and harbor reads `claude-code.txt` back itself (`claude_code.py:862-865`), so mutating it in place is unsafe. | **(d)** — tradeoff: the judges lose timing evidence permanently. |
| D123 | A missing trace produces a `WARNING` and the script continues. | See D131. | **(a)** |
| D124 | `parse_trace.py` invocation (`test.sh:185-186`) is `run_task.sh:323-325` verbatim, including the `cp` of `solve_parsed.txt` back into `$JOB_DIR`. | The judges' prompts name `../solve_parsed.txt` as their primary evidence, with `../solve_out.txt` as fallback; the contamination judge's phase 1 is *"Reconstruct from the trace first"*. | **(a)** |

#### 4.5.5 Cleanup

| id | decision | justification | tag |
|---|---|---|---|
| D125 | `tree ${JOB_DIR}/task` gains a `find -maxdepth 3 \| sort` fallback. | `run_task.sh:332` runs `tree` on the **host**, which has it. The verifier image does not install `tree` (neither `vllm_debug.def` nor `gpt_5_5.def` does), so in production the fallback is the branch that runs (D74). Output differs from upstream's — different format, and `-maxdepth 3` truncates. Diagnostic only; nothing parses it. ⚠️ Uncommented in the file. See §8/U10. | **(b)** |
| D126 | Order preserved: `final_model` and `system_monitor.log` copied out **before** `delete_hf_models.py`. | `run_task.sh:335-343`. Load-bearing: the judges run after the delete, so they see a `task/` dir with the weights already stripped — which is why `prepare_judge_sandbox` hands them `final_model/config.json` separately (`judge_lib.sh:78-80`). | **(a)** |
| D127 | `delete_hf_models.py`, `cp -r task` kept verbatim. | `run_task.sh:343`, `:345`. | **(a)** |

#### 4.5.6 The judges

| id | decision | justification | tag |
|---|---|---|---|
| D128 | **The judges authenticate in API mode against the model proxy, not with a ChatGPT subscription.** | Upstream runs all four on `agents/codex_non_api/auth.json`, bind-mounted at `judge_lib.sh:162` so the rotated single-use refresh token survives the job, with `CODEX_API_KEY` and `OPENAI_API_KEY` blanked (`:158-159`) to force it. Kaggle has no subscription auth; every LLM call goes through the proxy. So the blanking is inverted: the keys must stay populated (D44). | **(b)** |
| D129 | `setup_judge_codex_auth` becomes `setup_judge_codex_config` — the `.codex` reset is kept, the auth.json placeholder (`judge_lib.sh:105`) and bind are dropped. | The reset is what stops agent-specific codex settings such as `model_reasoning_effort` leaking into the judges (`judge_lib.sh:88-90`); it has nothing to do with auth and must stay. | **(a)** |
| D130 | ⚠️ `judge_lib.sh:106-108`'s `forced_login_method = "chatgpt"` append is **also** dropped, and that is only implicitly covered by "minus the auth.json handling". | Correct behaviour — forcing ChatGPT login in API mode would break the judges outright. But the omission is not called out. See §8/U11. | **(a)** |
| D131 | The OAuth precheck (`run_task.sh:253-281`, a `curl` to `chatgpt.com/backend-api/codex/models`) becomes a reachability check against `${OPENAI_BASE_URL}/models`. | Same purpose — fail fast on broken judge credentials rather than after 10 h. ⚠️ **Weaker**: upstream `exit 1`s on any non-200 (`run_task.sh:274-280`); the port only **prints** the HTTP code and continues (`test.sh:391`). Deliberate: on Kaggle a transient proxy blip should not discard a finished agent run, and there is no rerun pipeline to recover it. | **(d)** — tradeoff: a genuinely broken proxy now costs a full evaluation before the missing verdicts surface at scoring time (`score_run.py` exit 1). |
| D132 | `judge_model_map.json` remaps upstream's bare model names onto proxy slugs, applied **after** sourcing `judge.conf` so the conf stays byte-identical. | `judge_lib.sh:35` `JUDGE_DEFAULT_MODEL="gpt-5.4"` and `general_judge/judge.conf`'s `JUDGE_MODEL="gpt-5.6-terra"` are bare upstream names; the Kaggle proxy serves dated slugs. An identity map reproduces the upstream strings verbatim. | **(b)** |
| D133 | **One extra codex flag: `-c openai_base_url="…"`** (`test.sh:347-348`), applied only when `OPENAI_BASE_URL` is non-empty. | The codex CLI ignores `OPENAI_BASE_URL`. Harbor's own adapter says so at `agents/installed/codex.py:1243-1252` ("codex only honors `openai_base_url` from config.toml"), and a smoke run confirmed it: codex dialled `wss://api.openai.com/v1/responses` and 401'd with the proxy key. `-c` is the same mechanism upstream already uses for `model_reasoning_effort` (`judge_lib.sh:167`), so this **adds** a flag rather than changing one. Upstream needs no such flag — its judges talk to the real OpenAI endpoint. | **(b)** |
| D134 | `apptainer exec --containall … --home … --pwd …` becomes a direct call in a subshell with `cd "$job_dir/task"` and `HOME="$job_dir"`. | This container **is** the judge's container. Every flag with an analogue is kept; every flag without one is an apptainer-ism (D72). | **(b)** |
| D135 | `--env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH"` (`judge_lib.sh:157`) not reproduced. | Same gap as D28. Harmless here because the pinned codex is invoked by absolute path (`test.sh:334`). | **(a)** — should be fixed |
| D136 | The pinned-codex npm install uses `--prefix "${job_dir}/${pin_prefix}"` and an absolute `codex_bin`, where upstream uses `/home/ben/…`. | `job_dir` **is** `/home/ben`, so the strings are identical at run time. Written as a variable because `test.sh` parameterises `JOB_DIR`. | **(a)** |
| D137 | `--search`, `-a never`, `exec --json`, `-c model_reasoning_summary=detailed`, `-c model_reasoning_effort=…`, `--skip-git-repo-check`, `--yolo`, `--model`, `2>&1 \| tee` — all kept verbatim from `judge_lib.sh:167`. | `--search` in particular is required, not optional (2026-08-24): 1:1 means it stays regardless of whether a judge would have used it. Verified 2026-08-26 that `gpt-5.4` and `gpt-5.6-terra` both answer on `/openapi/responses` **and** run a real `web_search_call` — **but that was from a workstation token; the platform-minted run token 403s the `web_search` tool. See D137a/D137b.** | **(a)** |
| D137a | ⚠️ **SUPERSEDED BY D227 (2026-09-03).** The four-flag block is gone: bumping the verifier's codex to 0.146.1, which falls back to HTTPS on the 307, removed the need for it. `test.sh:373-376` now uses the stock one-flag `-c model_providers.openai.base_url=…`. The diagnosis below is still the reason D227 exists — kept in full because it is the measurement that explains the version pin. ~~**`-c openai_base_url=` replaced by a four-flag custom `model_provider` (`kproxy`).**~~ | Not a preference — the one-flag form does not work. It redirects codex's **built-in `openai`** provider, which prefers a websocket transport and dials `wss://$OPENAI_BASE_URL/responses`. The proxy registers that route only **with a trailing slash**: slashless → **307**, with slash → **401** (route exists). codex's ws client does not follow redirects, so it burns five reconnects and fails the turn — which is exactly why run 768637 died with `RewardFileNotFoundError` after a 7-minute verifier phase. Measured in prod (run 865446 `verifier/test-stdout.txt`), reproduced locally with codex 0.142.5, redirect confirmed with a raw handshake. A **custom** provider skips the websocket path entirely; `model_providers.openai` cannot be overridden ("reserved built-in provider IDs … cannot be overridden"), so a new id is the only expression of the fix. Verified in prod, run 866264: no websocket error on any custom-provider attempt. ⚠️ Harbor's own codex adapter (`agents/installed/codex.py:1399-1406`) sets `openai_base_url` on the built-in provider, so **every** codex agent on Kaggle takes the broken path. | **(b)** — forced. Four `-c` flags replace one; `-c` is the same mechanism upstream already uses for `model_reasoning_effort`. |
| D137b | **`PTB_DISABLE_WEB_SEARCH=1` starts `tests/strip_web_search.py`, a loopback shim that deletes `web_search` from `tools[]`. OFF by default.** | With D137a in place the judges hit a second, independent wall: `403 Forbidden: Google Search Grounding is not enabled for your account`. Raw HTTP from inside the verifier isolates it (run 866861): no tools → 200, one **function** tool → 200, `tools:[]` → 200, `stream:true` → 200, `chat/completions` + function tool → 200, **`tools:[{"type":"web_search"}]` → 403**. Only the provider-*native* tool is refused. codex sends it on every request and no config removes it — `--search`/no `--search`, `tools.web_search=false`, `tools.web_search_request=false`, `tools.enabled_tools=[]`, `experimental_supported_tools=[]` all yield the same 11-tool list (captured against a recording HTTP server; confirmed in prod runs 866646/866861). The real fix is the feature flag **`MODEL_PROXY_NATIVE_TOOL_ACCESS` (4831)**, which sets `allow_model_native_tools` on the minted token (`TokenUtil.cs:151-154`, called from `BenchmarkContainerEnvVarBuilder.cs:214-221`) and is evaluated against the account launching the run — self-serve, same shape as `BENCHMARKS_ALLOW_USE_ACCELERATOR`. The shim exists only to answer "does everything *else* work", which is the test requested on 2026-08-31: strip web search from the judges and models, and see whether the rest of the path works. Prod run **867385: both judge models return `turn.completed` through the shim; the no-shim control still 403s.** ✅ **Flag 4831 was granted on 2026-09-01 and the shim became unnecessary**: prod run **868182** runs the FULLY FAITHFUL invocation (`--search` kept, `web_search` advertised, no shim) on `openai/gpt-5.4-2026-03-05` and `openai/gpt-5.6-terra` and gets `turn.completed` on all four attempts, including a multi-turn tool-heavy one with **2 real `web_search` calls each**. The shim is retained, inert and off by default, as the diagnostic that isolated the cause. | **(c)** — a run with this set is **NOT 1:1**. Default off, and nothing ships with it on; the judge path is byte-for-byte D137 unless the env var is set. |
| D138 | `collect_judge_output` keeps `missing_fatal = 0` semantics. | `run_task.sh:390` passes `0`; `judge_lib.sh:188-196` warns and returns 0. A judge that produces no verdict must never cost a finished agent run its evaluation. | **(a)** |
| D139 | ✅ **CLOSED 2026-09-01.** The `parse_trace.py --agent codex` call is back, and prod run 871144's bundle carries `judge_output_{gpt5_4,api,ptb_lookup,general}.txt` alongside the `.json`. Historical note follows. ⚠️ **`collect_judge_output` drops `judge_lib.sh:186`'s `parse_trace.py --agent codex` call.** The per-judge human-readable transcript `judge_output_<id>.txt` is therefore **never produced**. | Undocumented until this audit. `AGENTS.md`'s "Results Structure" lists `judge_output_*.{json,txt}`; this port produces only the `.json`. Nothing downstream reads the `.txt` (`collect.py`/`utils.py` never open it), so scoring is unaffected — but the artifact upstream ships for human review is missing, and the port's own warning message at `test.sh:375` even points at the `.json` instead of the `.txt` upstream points at. See §8/U12. | **(a)** — should be fixed |
| D140 | `JUDGE_EXTRA_APPTAINER_ARGS` (`run_task.sh:366-371`: `--nv`, `HF_HOME`, `VLLM_API_KEY`, the overlay bind) has no analogue. | The verifier container already has the GPU (`gpus = 1`), `VLLM_API_KEY` is in `[verifier.env]`, and the cache arrives via the compose fragment. `HF_HOME` is **not** set for the judges here, where upstream sets it — no judge tool reads it (`contamination_check.py` and `model_identity_check.py` are pure-Python file checks), so the effect is believed nil. **UNVERIFIED.** | **(b)** |
| D141 | `with_huggingface_overlay run_judge_exec` (`run_task.sh:385`) becomes a plain call. | No overlay (D77). | **(b)** |
| D142 | The judge loop itself (`FIRST_JUDGE`, the `rm -f judgement.json` between judges, `load_judge_conf \|\| exit 1`) is `run_task.sh:373-391` verbatim. | | **(a)** |
| D143 | **No trace guard, deliberately.** | `src/judges/run_judges.sh:82-92` has one, but that is the standalone re-run tool (`src/judges/rerun/`); the sweep path is `run_task.sh`, which sources `judge_lib.sh` and runs the loop inline at `:373-391` with no guard of any kind and no `set -e`. Upstream with no trace: `parse_trace.py` fails nonfatally, the judges find neither `../solve_parsed.txt` nor `../solve_out.txt`, and `collect_judge_output` runs with `missing_fatal=0`, which warns and returns 0. **The run still evaluates and still scores.** An earlier pass of this port *did* add the guard; it was wrong and was removed. | **(a)** |

#### 4.5.7 Evaluation

| id | decision | justification | tag |
|---|---|---|---|
| D144 | `apptainer exec … vllm_debug.sif python "${EVAL_SCRIPT}"` becomes a direct call in a subshell with `cd "$REPO_ROOT/src/eval/tasks/${EVALUATION_TASK}"`. | This is the evaluation container. **Every argument to `evaluate.py` is unchanged**, including `--templates-dir ../../../../src/eval/templates`, which resolves identically because `/tests/repo` mirrors the upstream tree (D98, asserted at `verify_fidelity.py:395-399`). | **(b)** |
| D145 | **PID-1 guard on the GPU kill.** `run_task.sh:406` is `nvidia-smi --query-compute-apps=pid … \| xargs -r kill -9`; `test.sh:442-446` skips pid ≤ 1. | Upstream's apptainer sandbox has no init to kill. In a Docker container a reparented vLLM child can land on PID 1, and killing it takes the container down. Adopted from PR #8's `template/tests/test.sh:166-176` ⚠️ **(c)** — a Docker-vs-apptainer operational lesson, which is the one category PR #8 is cited for. Independently justified by the mechanism, not by PR #8's authority. | **(c)** |
| D146 | ⚠️ `run_evaluation`'s output redirect gains `2>&1`. Upstream (`run_task.sh:424`) redirects **stdout only**; stderr goes to `run_task.sh`'s `error.log`. | Undocumented until this audit. Here there is no `error.log` (D105), so without `2>&1` the evaluation's stderr — which is where inspect-ai's tracebacks land — would vanish into the container log. `final_eval_N.txt` therefore contains strictly more than upstream's. Nothing parses it. See §8/U13. | **(b)** |
| D147 | `TMP_HF_CACHE` becomes `${PTB_HF_HOME:-/tmp/hf_cache_90afd0}`. | `run_task.sh:399` sets the literal and `:417` binds the overlay onto it. Here the cache is at the mount path; the fallback preserves upstream's literal for a run without one. | **(b)** |
| D148 | `declare -f run_evaluation` instead of `declare -f run_evaluation with_huggingface_overlay` (`run_task.sh:441`). | The overlay function does not exist. | **(b)** |
| D149 | The retry schedule — `4`, then `3`, then `2`, with the two `case` token tables — is `run_task.sh:452-512` verbatim, **including** the comment at `:454` that says "up to 2 attempts" while the code passes 3. | `verify_fidelity.py:133-156` re-parses the retry counts, the 28 800 s attempt timeout and **both** token tables out of `run_task.sh` and fails on any mismatch. | **(a)** |
| D150 | The pre-loop `sleep 5`, the early return when `metrics.json` already exists, and the 28 800 s per-attempt `timeout --signal=TERM --kill-after=60s` are kept verbatim. | `run_task.sh:427-449`. | **(a)** |
| D151 | ⚠️ `echo $(cat "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt")` (`run_task.sh:514`) becomes a guarded `cat`. | Two differences: the `[ -f ]` guard (upstream errors if no attempt ever ran) and `cat` vs `echo $(cat …)` (the latter collapses all whitespace onto one line through word splitting). The port's output is strictly more readable and strictly more faithful to the file's actual bytes — but it is not what upstream prints. Diagnostic only. Uncommented. See §8/U14. | **(d)** |

---

### 4.6 `tests/score_run.py`

**Replaces:** nothing upstream runs at task time. `run_task.sh` stops at
`metrics.json`; `scripts/collect.py` turns a run directory into a score days
later, with the whole sweep in hand. Harbor's contract is one scalar per task, so
collect.py's per-cell rules have to be applied here.

Every function is a transcription with a `utils.py` / `collect.py` citation.

| id | decision | justification | tag |
|---|---|---|---|
| D152 | The whole file exists. | Harbor requires a reward. `Trial` reads it from the verifier. No upstream analogue at all. | **(b)** |
| D153 | `load_metrics` reads **`accuracy` and nothing else**, raising `KeyError` otherwise. | `utils.py:379-400` does exactly that. There is deliberately **no** fallback chain to `pass@1` / `score` / `exact_match` / first-numeric. PR #8 has such a chain ⚠️ **(c)** — cited here only as a counter-example of what not to do; it is a divergence from main, not upstream behaviour. | **(a)** |
| D154 | It returns `float`, where `utils.py:400` returns `str(accuracy)`. | `aggregate.py:119` immediately does `float(...)`, so the value is identical. A float is what harbor's reward file wants. | **(d)** — cosmetic. |
| D155 | Rerun-verdict preference (`judgement_gpt5_4_rerun.json` before `judgement_gpt5_4.json`, and the same for `_api` / `_ptb_lookup`) is reproduced. | `utils.py:410-430`, `:487-515`. There is no rerun pipeline here, so the `_rerun` branch can never fire — but reproducing it costs nothing and keeps the transcription literal. | **(a)** |
| D156 | Every type check is reproduced: top-level must be an object, fields must be present, booleans must be `bool` (`isinstance(..., bool)`), accuracy must be numeric-and-not-bool. | `utils.py:432-462`, `:494-515`, `:395-400`. These raise the exact exception types collect.py's `except` clause catches, which is what makes the baseline fallback fire correctly. | **(a)** |
| D157 | `missing_required_judgements` drops upstream's `run_id >= NEWER_JUDGES_MIN_RUN_ID` gate (`utils.py:553`, `:565-568`) and requires the PTB-lookup verdict **unconditionally**. | `NEWER_JUDGES_MIN_RUN_ID = 17400000` was chosen above every run id that existed when the judge was added (2026-07-16, max was 17 397 666). Every run this port produces is newer than that, so the gate is always true. Reproducing it would require inventing a run id. | **(a)** |
| D158 | **A firing PTB-lookup judge produces no reward, a `PTB_LOOKUP_JUDGE_FIRED` marker file, and exit 1.** | `collect.py:136-141` raises `RuntimeError`, which is deliberately **not** in the `except` tuple at `:147`, so it propagates and collect.py produces nothing for the whole sweep. The nearest local equivalent is refusing to score the trial. | **(d)** — tradeoff: upstream kills the whole aggregation; a per-task port can only kill the task. Naming the marker file makes the reason visible in the artifact dir. |
| D159 | **A missing required verdict produces no reward, a `MISSING_JUDGEMENTS` marker, and exit 1.** | `collect.py:126-130` puts the run on `judgements_missing`, and `:351` makes main() skip that whole method — warning, no CSVs — rather than aggregate an unchecked score. Upstream can defer because the rerun pipeline supplies the verdict later; there is none here, so the consequence surfaces now. | **(d)** — tradeoff: a proxy outage during the judge phase discards an otherwise-complete 10 h run. This is the strictest reading of collect.py and is deliberate: an unchecked score is worse than no score. |
| D160 | ⚠️ **The order of the checks differs from collect.py, with a behavioural consequence.** `score_run.py` runs the PTB-lookup check and the missing-verdict check **before** `load_metrics`; `collect.py:116-141` runs `load_metrics` **first**, inside the guarded block. | Undocumented until this audit — and `score_run.py`'s own module docstring (lines 13-26) records collect.py's order correctly, so the implementation contradicts its own documentation. Consequence: a broken run with **no `metrics.json`** and **no judge verdicts** exits 1 with no reward here, whereas upstream takes the `FileNotFoundError` at `:116`, jumps to the `except` at `:147`, and **scores the baseline** — collect.py's own comment at `:120-125` says the ordering is deliberate *"so collect.py stays usable mid-sweep"*. Same inversion applies to the PTB-lookup `RuntimeError`. See §8/U15. | **(a)** — should be fixed |
| D161 | `PTB_ALLOW_MISSING_JUDGEMENTS=1` bypasses D159. | Smoke builds only; set from `task.toml` `[verifier.env]` under `--smoke` (`build_tasks.py:426-427`). | **(d)** |
| D162 | The general judge is ignored entirely — never required, never scored, never an error. | `collect.py:19-22` and `utils.py:562-563`, `:592`, explicitly. ⚠️ Note that `src/judges/general_judge/judge.conf`'s own header comment contradicts this, claiming collect.py *"raises an error listing every flagged run"*. **The code wins**: `grep general scripts/collect.py scripts/utils.py` shows the verdict is never loaded. That stale comment is copied byte-for-byte into the task (§2) and is upstream's problem, not the port's. | **(a)** |
| D163 | ✅ **STALE ROW, CORRECTED 2026-09-01.** Upstream's "skip the warning when a `*final_eval_9.txt` exists" refinement (`collect.py:153-156`) **is** reproduced, at `score_run.py:281`, and the glob deliberately also matches the rerun naming `z_new_<id>_final_eval_9.txt`. | The row claimed it was not reproduced; that stopped being true when the warning-suppression branch was added. Verified against the file. | **(a)** |
| D164 | `load_time_taken` returns `int` where `utils.py:627` returns `tuple[str, int]`. | Only the seconds are used, and only for its exceptions. | **(d)** — cosmetic. |
| D165 | The reward is written as a bare decimal to `reward.txt`, and the process exit code is `test.sh`'s exit code. | Harbor's contract. Verified end to end: a smoke run returned exactly `0.12661106899166036`, the `baselines.json` zero-shot value for `Qwen3-1.7B-Base`/`gsm8k`. | **(b)** |
| D166 | The reward scale is **0–1**. | `baselines.json` is all `0.xxx`; inspect-ai's `accuracy` is a fraction. On the Kaggle side this means the 28 leaf mappings take `PERCENTAGES` (0–1 in) and the root takes `RAW_PERCENTAGES` (0–100 in); the 7 benchmark parents are corrected automatically because `AVERAGE` is in `NormalizesToRawPercentage`. Setting the root to `PERCENTAGES` would render `23.2` as `2320%`. | **(b)** |

---

### 4.7 `tests/docker-compose.yaml`

**Replaces:** `run_task.sh:410` + `:417` — upstream binds the same overlay into
the evaluation phase at `/tmp/hf_cache_90afd0`.

| id | decision | justification | tag |
|---|---|---|---|
| D167 | The file exists at all. | On harbor the normal route is closed **three ways**: `--mounts` (how a Kaggle dataset mount arrives) is appended to the **agent** environment only (`harbor/trial/trial.py:1299` = `base + list(self.config.environment.mounts or [])`); the separate verifier env is built with a single fixed bind of `/logs/verifier` (`trial.py:680-692`); and `extra_docker_compose` is explicitly blanked for it (`trial.py:648-650`). `task.toml` cannot help either: `EnvironmentConfig` has **no `mounts` field** (verified: `grep mounts harbor/models/task/config.py` → 0 hits) and, because pydantic `extra` is left at `"ignore"`, writing one is **silently dropped rather than rejected**. Probe output, constructing both environments for real: agent targets `['/logs/agent', '/logs/artifacts', '/logs/verifier', '/mnt/hf-cache']`, verifier targets `['/logs/verifier']`. | **(b)** |
| D168 | The one seam that *is* honoured: the verifier's build context is the task's `tests/` directory (`trial.py:693-702`) and `DockerEnvironment._docker_compose_paths` (`harbor/environments/docker/docker.py:363-364`) layers in `<environment_dir>/docker-compose.yaml` when it exists. | Verified by constructing the real verifier environment from a generated task: its compose stack came back `['docker-compose-build.yaml', 'docker-compose.yaml']`, the second being ours. | **(b)** |
| D169 | Service name must be `main`. | Harbor's compose convention. | **(b)** |
| D170 | The mount path is written **literally** (`/mnt/hf-cache:/mnt/hf-cache:ro`), not as `${PTB_HF_CACHE_MOUNT}`. | Interpolation resolves against the *verifier's* environment, where that variable need not be set, and an empty expansion yields a malformed bind. | **(b)** |
| D171 | A missing source degrades safely. | Docker creates an empty directory, so the evaluation behaves as it did before this file existed. | **(d)** — tradeoff: a silent degradation instead of a loud failure. ⚠️ This matters: **`gpqamain` uses the gated `Idavidrein/gpqa`** and the verifier carries no `HF_TOKEN`, so a silently-empty cache fails that one benchmark and only that one. Watch it first. |
| D172 | `:ro`. | Kaggle dataset mounts are read-only anyway; making it explicit means the verifier cannot corrupt the shared cache. Upstream's evaluation gets the writable overlay. | **(d)** — tradeoff: an evaluation that wanted to download something new cannot. It never does. |
| D173 | The route is **verified structurally, never on a real Kaggle run.** | The compose fragment is proven to be layered into the verifier's stack (D168), but no real run has confirmed the bind resolves on the Kaggle executor. Stated as an open item, not a claim. **UNVERIFIED** end to end. | **(b)** |
| D174 | **`PTB_HF_CACHE_MOUNT` is a COLON-SEPARATED LIST of mount paths, PATH-style — not the single path D47/D147/D167–D173 describe.** Both consumers iterate it: `preflight.sh:98-118` splits on `:` and link-farms each shard into `HF_HOME`; `test.sh:567-588` splits and links each into the writable `TMP_HF_CACHE`. `build_tasks.py` emits it as `":".join(hf_cache_mounts)`. | The ~160 GB cache spans ~396 HuggingFace repos, which unpacks to well over the 1,000 files a single Kaggle dataset allows without `ExtraDatasetsQuota` (`DatasetConstants.cs:42`), so it ships as several datasets, each mounted separately. Merging is free: every shard carries the same top-level `$HF_HOME` layout (`hub/`, `datasets/`), the `mkdir -p` is idempotent, order does not matter, and a repo lives in exactly one shard so there is no dedup. A single-element list is the degenerate case, which is why the rows above stayed readable while being wrong. | **(b)** — forced by the per-dataset file cap. |

---

### 4.8 `tests/metadata.json`

**Replaces:** `run_task.sh`'s positional arguments `$1`-`$7`, plus lookups into
`scripts/baselines.json` and `scripts/factors.json` that upstream only does at
aggregation time.

| id | field | justification | tag |
|---|---|---|---|
| D174 | `benchmark_id`, `model_to_train`, `num_hours`, `num_gpus`, `agent`, `agent_config` | `run_task.sh:3-9`'s arguments. A Harbor verifier has no argv, so they have to be baked into a file the agent cannot write. | **(b)** |
| D175 | `benchmark_name` | `get_prompt.py:17` reads it from `benchmark.txt`. Recorded so the task description can use it without re-reading. | **(a)** |
| D176 | `model_key` | The bare name after the organisation slash — the key `baselines.json`, `utils.py:EXPECTED_MODELS` and collect.py's `walk_latest_runs` all use. | **(a)** |
| D177 | `eval_script` | Which of `evaluate.py` / `evaluate_openrouter.py` was baked in (D7). Read back by `verify_fidelity.py:335`. | **(b)** |
| D178 | `zeroshot_baseline` | `score_run.py`'s fallback value, from `scripts/baselines.json["zeroshot"][model][benchmark]` — the exact table `collect.py:183` uses. Resolved at build time because the verifier has no copy of `baselines.json`. `verify_fidelity.py:485-487` re-derives it. | **(a)** |
| D179 | `benchmark_weight` | `scripts/factors.json` — the weight this benchmark carries in `aggregate.py:107-122`. **Not used by the verifier at all**; recorded so the Kaggle parent mapping can be built from the tasks rather than from a second copy of the table. `verify_fidelity.py:222-225` asserts the weights sum to 1.0, which is what makes `(ΣWᵢXᵢ)/(ΣWᵢ)` on the Kaggle side agree exactly with upstream's un-divided sum. | **(d)** — tradeoff: carrying a value the verifier ignores. The alternative is a second hand-maintained weights table. |
| D180 | `smoke_build` | See D13. | **(d)** |

---

### 4.9 `tests/judge_model_map.json`

| id | decision | justification | tag |
|---|---|---|---|
| D181 | The file exists. | `judge_lib.sh:35`'s `JUDGE_DEFAULT_MODEL="gpt-5.4"` and `general_judge/judge.conf`'s `JUDGE_MODEL="gpt-5.6-terra"` are bare upstream names. The Kaggle model proxy serves dated slugs. Remapping in a separate file lets `judge.conf` stay byte-identical to upstream (§2) — the alternative was editing a copied file, which the port never does. | **(d)** — tradeoff: one more generated file. Chosen over editing a copy. |
| D182 | Current contents: `gpt-5.4 → openai/gpt-5.4-2026-03-05`, `gpt-5.6-terra → openai/gpt-5.6-terra`. **Updated 2026-09-01** — was previously the unprefixed `gpt-5.4-2026-03-05` / identity `gpt-5.6-terra`. | Both targets must be RESPONSES-wired, because the proxy picks its backend from the slug rather than the request shape (`entrypoint-common.sh:509-518`) and codex speaks only Responses. Verified against the live 145-model catalog on 2026-09-01: `openai/gpt-5.4-responses` does NOT serve (**503**, configured with no backend), and `gpt-5.6-terra` is already the Responses variant — the proxy carries `gpt-5.6-terra-cc` as its Chat-Completions sibling. **gpt-5.4 stays gpt-5.4**; a `gpt-5.5-responses` substitution was written here and reverted (the project permits gpt-5.4 and gpt-5.6 only, not gpt-5.5), and prod run 868182 shows gpt-5.4 completing multi-turn tool-and-search judge-shaped work over Responses with no contamination. | **(b)** |
| D183 | Lookup is `m.get(name, name)` (`test.sh:270-274`) — an unmapped name passes through unchanged. | A judge added upstream without a map entry keeps working with its upstream name. | **(d)** |
| D184 | `build_tasks.DEFAULT_JUDGE_MODEL_MAP` is a module constant with no CLI flag. | Changing it is a code edit, which is visible in review. The old **UNVERIFIED** note on whether `gpt-5.4-2026-03-05` is still the right slug is resolved: it was re-checked against the live catalog on 2026-09-01 and exercised end to end by prod runs 868182 and 871144. | **(d)** |

---

### 4.10 `environment/docker-compose.yaml`

**Replaces:** nothing upstream. It exists only because two things cannot be
said in `task.toml`.

| id | decision | justification | tag |
|---|---|---|---|
| D216 | The file exists at all, on the **agent** side as well as the verifier side (§4.7). | Harbor layers `<environment_dir>/docker-compose.yaml` over its own stack for *both* environments (`environments/docker/docker.py:363-364`). It is the only seam for **build args**: harbor's build file is `build: {context: ${CONTEXT_DIR}}` with no `args:` (`environments/docker/docker-compose-build.yaml`) and nothing on the Kaggle path passes `--build-arg`, so without this file the `ARG TORCH_BACKEND=auto` default would be the only reachable value and a GPU-less builder would resolve torch to CPU and fail vllm's CUDA-only `xformers` dependency. | **(b)** |
| D217 | The agent-side copy carries **only** `build.args`, never a `volumes:` block; the verifier-side copy carries both. | The agent environment already receives Kaggle dataset mounts through `--mounts` (`harbor/trial/trial.py:1299`); only the verifier is cut off from them. Emitting an agent-side bind would duplicate a mount harbor already makes. | **(b)** |
| D218 | The file is **deleted rather than written empty** when it would carry nothing (prebuilt image, no binds). | An empty `services.main` is not valid compose input; and with `docker_image` set there is no build step, so `build.args` would be misleading noise pointing at a build that never happens. | **(d)** — tradeoff: the file's presence varies by configuration, so "is it there?" is not a fixed expectation. |
| D219 | ⚠️ In the shipped 3-hour and 5-minute tasks this file carries `TORCH_BACKEND: auto`, which is **inert**, because both tasks set `docker_image`/`verifier_docker_image` and harbor therefore pulls instead of building (`environments/definition.py:26-36`, and `"force_build": false` in every run's `lock.json`). | Stated so nobody reads `auto` here and concludes the shipped images were built with `auto`. They were built locally with `--build-arg TORCH_BACKEND=cu129`, which is the wheel Kaggle's `auto` resolves to on an A100 (probe run 767447 returned `129`). | **(b)** |

---

### 4.11 `tests/strip_web_search.py`

**Replaces:** nothing upstream. Diagnostic only, and **inert unless
`PTB_DISABLE_WEB_SEARCH=1`**, which nothing shipped sets.

| id | decision | justification | tag |
|---|---|---|---|
| D220 | The file is staged into every task even though no shipped task enables it. | Turning the diagnostic on becomes an env-var change rather than an image rebuild, which matters because a rebuild-and-repush is the slowest loop in this project. It contributes ~6 KB and no behaviour. | **(d)** — tradeoff: dead code in a shipped image. Chosen over a 25 GB rebuild to debug a judge. |
| D221 | What it does: a loopback HTTP proxy that deletes any `tools[]` entry whose `type` starts with `web_search`, and forwards everything else verbatim, streaming the response back unmodified. | It was the only way to answer "does anything *else* in the judge path work" while the proxy 403'd the native tool — exactly the test requested on 2026-08-31. See D137b for the measurement chain. | **(c)** |
| D222 | It is **not** needed any more and is not used. | `MODEL_PROXY_NATIVE_TOOL_ACCESS` (4831) was granted 2026-09-01; prod run 868182 then ran the fully faithful invocation, `--search` and all, on both judge models. Kept as the reproduction for a failure mode that will recur the moment that flag is lost. | **(a)** |
| D223 | ⚠️ A run with `PTB_DISABLE_WEB_SEARCH=1` is **NOT 1:1** and its reward is a pipeline proof, not a benchmark number. Run **867724** (reward 0.172100) is such a run. | Recorded here so that number is never quoted as a benchmark result. | **(c)** |

---

## 5. The two shared files at the task-definition root

`agents/` is staged **once**, beside the 28 task directories, by
`build_tasks.py:708-713`. It is not part of any task.

### 5.1 `agents/__init__.py`

| id | decision | justification | tag |
|---|---|---|---|
| D185 | The package exists at the `--output` root rather than inside a task. | `harbor_apply_custom_import_pythonpath` (`container/harbor-base/entrypoint-common.sh:531-541`) prepends `KAGGLE_TASK_DEFINITION_ROOT` — the task-definition mount root, i.e. the output directory — to `PYTHONPATH` whenever `$AGENT` contains a `:`. So `--agent agents.ptb_harness:…` resolves only if the package sits there. Same layout as `kaggle-benchmark-harbor-starter-template/agents/`. | **(b)** |
| D186 | Docstring only; no code. | Nothing to do. | **(d)** |
| D187 | `__pycache__` is removed after staging (`build_tasks.py:712`); `verify_fidelity.py:297-298` fails if it reappears. | Same reasoning as D2. | **(d)** |

### 5.2 `agents/ptb_harness.py`

**Replaces:** `agents/<name>/solve.sh` (partly), `get_prompt.py:120-123`'s
run-time half, and the per-agent `case` branches in the shared executor
entrypoint that a custom import path does not match.

#### 5.2.1 Why it exists

| id | decision | justification | tag |
|---|---|---|---|
| D188 | Four thin subclasses of harbor's real adapters — `PtbHarness_{claude,codex,gemini,opencode}` — each overriding `run()` only. | `get_prompt.py:120` makes upstream's prompt agent-dependent; a Harbor task has one static `instruction.md`. The clause has to be re-applied at run time, per harness. Each class calls `super().run(...)` so **every line of the real harness still executes**; nothing is replaced or reimplemented. | **(b)** |
| D189 | `apply_agent_clause` reproduces `get_prompt.py:120-123` by round-tripping the capture: `rstrip("\n")` → append the literal → `rstrip("\n") + "\n"`. | Undoes `run_task.sh:76`'s `$( )` capture to recover `get_prompt.py`'s `result`, applies the clause, re-applies the capture (D5). `verify_fidelity.py:255-258` asserts `_NON_INTERACTIVE_CLAUSE` is byte-identical to the literal in `get_prompt.py`'s own source, extracted by string index rather than transcribed. | **(a)** |
| D190 | **The class names are load-bearing data, not style.** | The same string feeds two upstream substring dispatchers: `get_prompt.py:120` (`'claude' in args.agent`) and `parse_trace.py:36-45` (`select_parser`, which hard-`SystemExit`s on >1 match and silently copies the raw trace on 0). Each slug contains exactly one parser key and contains `claude` iff the upstream agent name does. `opencode` does not contain `codex`; `agents`, `ptb_harness` and `PtbHarness` contain no key. `verify_fidelity.py:266-286` asserts all of it against upstream's real `PARSERS` dict, plus that the slug resolves through harbor's own `import_class`. | **(a)** |
| D191 | Registered on Kaggle as a **HarnessVersion's `HarborSlug`**, scheduled against the Agent that references it. | `BenchmarkContainerEnvVarBuilder.cs:361` emits it as `KAGGLE_AGENT_HARNESS`; `harbor_resolve_agent` exports it as `AGENT` (`entrypoint-common.sh:247-264`); `harbor_exec_run` passes it as `--agent` (`:564-565`). | **(b)** |
| D192 | ⚠️ **NOT** via a task version's `OverrideHarnessVersionCustomSlug`. | That also reaches `--agent` (`:246-251`), but `CreateBenchmarkTaskFromHarborKaggleDatasetsHandler.cs:190-195` forces `CandidateType = ModelVersions` whenever it is set — pinning one harness across all 28 tasks, which a sweep whose rows vary both model and harness cannot survive. This overturned an earlier recommendation. | **(b)** |

#### 5.2.2 The executor `case`-branch gap

The shared executor entrypoint switches on `$AGENT` by **exact name** in three
places, and a custom import path matches none of them — exactly as
`harbor.agents.oneshot.*` did before it got its own branches at
`entrypoint-common.sh:346` and `:390`. Editing `container/` is off limits to this
port, so the wrapper closes the gap itself, in the shape the official
starter template already uses (`kaggle-benchmark-harbor-starter-template/agents/antigravity_agent.py:112-128`).

| id | gap, and the entrypoint branch it replaces | what the wrapper does, and why | tag |
|---|---|---|---|
| D193 | **gemini credentials** — `entrypoint-common.sh:335-345` | Fills `GEMINI_API_KEY` and `GOOGLE_GEMINI_BASE_URL` (= proxy + `/genai`, matching `:344`). Without them the CLI dies with *"you must specify the GEMINI_API_KEY environment variable"* (observed) or, with only the key, talks to `generativelanguage.googleapis.com` instead of the proxy. | **(b)** |
| D194 | **reasoning effort** — `entrypoint-common.sh:365-396` + `harbor_append_reasoning_effort:404-420` | Injects the harness's own kwarg from `KAGGLE_AGENT_LLM_REASONING_EFFORT` — `reasoning_effort` for claude/codex/gemini, `variant` for opencode (`:377-383`). | **(b)** |
| D195 | **codex model slug** — `harbor_map_responses_slug`, `entrypoint-common.sh:512-518` | Rewrites `gpt-5.5` / `openai/gpt-5.5` → `openai/gpt-5.5-responses`. In scope because `commit.sh:52` and `:65` both run `agent_config=gpt-5.5` against codex. **One `if` for one model, not a slug map** — copied exactly. | **(b)** |
| D196 | Nothing is overwritten: entrypoint value first, `MODEL_PROXY_*` only as a fallback, `--ae` (`extra_env`) ahead of `os.environ`. | `BaseInstalledAgent._env_sources` (`harbor/agents/installed/base.py:584-590`) ranks `extra_env` above `os.environ`. If the entrypoint ever grows an `agents.ptb_harness:*` branch, the wrapper becomes a no-op. | **(d)** — tradeoff: duplicated logic that can drift. Mitigated by D197 and by logging every decision at construction. |
| D197 | The credential env **names** are read off each harness's own `MODEL_CONNECTION` (`harbor/agents/model_connection.py:133-147`), not written as literals, with an `assert` if the spec ever stops exposing `api_key_envs`/`base_url_envs`. | A harbor rename is followed automatically instead of the wrapper silently setting a name nothing reads. No literal names remain: the one harness with no spec, `CursorCli`, is not wrapped. | **(d)** |
| D198 | `opencode` needs no credentials. | `ModelConnectionSpec(passthrough=True)` (`opencode.py:53`) declares no env names, and the unconditional block at `entrypoint-common.sh:312-324` already covers it. | **(a)** |
| D199 | `claude_code.py:89-95`'s `env_fallback="CLAUDE_CODE_EFFORT_LEVEL"` does **not** make claude a special case. | The platform emits `KAGGLE_AGENT_LLM_REASONING_EFFORT` (`BenchmarkContainerEnvVarBuilder.cs:365`), a different name, so the fallback does not cover us and the kwarg is still required. Asserted in the unit suite. | **(b)** |
| D200 | No validation of the effort value. | `entrypoint-common.sh:401-403`: *"We do no validation in the executor and assume that configuration is correct for the particular harness."* A bad value must reach the adapter and be rejected there. | **(b)** |
| D201 | An unset `PTB_EFFORT_KWARG` logs and drops the value rather than failing. | `entrypoint-common.sh:413-416`, the empty-kwarg branch: *"not fatal … the agents without the knob shouldn't abort the sweep."* | **(b)** |

#### 5.2.3 The version pin

| id | decision | justification | tag |
|---|---|---|---|
| D202 | A three-source fallback chain exists at all: `--ak version=` kwarg → `PTB_HARNESS_VERSION` env → `PTB_DEFAULT_VERSION` class attribute → unpinned. | Upstream installs exact CLI builds, several per harness — 8 `@anthropic-ai/claude-code@`, 8 `@openai/codex@`, 4 `@google/gemini-cli@` and 1 `CURSOR_VERSION` across `containers/*.def` — so a leaderboard row is not fully specified without one. On the Agent route the platform supplies it (`BenchmarkContainerEnvVarBuilder.cs:362` → `KAGGLE_AGENT_HARNESS_VERSION` → `harbor_append_agent_version`, `entrypoint-common.sh:444-450` → `--ak version=`) and the wrapper is a pass-through. The chain fires only where nothing supplies one: a plain local `harbor run`, or the custom-slug route, whose branch at `:246-251` **blanks** the variable. | **(b)** |
| D203 | The env var is deliberately named `PTB_HARNESS_VERSION`, **not** `KAGGLE_AGENT_HARNESS_VERSION`. Together with `--ak version=` it overrides the `"latest"` default (D205) and is this port's equivalent of upstream's `POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1`. | Avoids colliding with the platform's own variable. ⚠️ Using it makes a run reproducible at the cost of no longer matching upstream's floating CLI — the same tradeoff upstream's own opt-out carries. | **(d)** |
| D204 | A non-string `--ak version=` value is stringified **with a warning**, not silently. | Harbor's `parse_kwargs` runs `json.loads` on every `--ak` value (`harbor/cli/utils.py:65-90`), so an unquoted `version=2.1` arrives as the float `2.1` and `version=1` as int `1`. `str()` cannot recover what `json.loads` discarded (`2.10` → `2.1`), so the coercion is flagged rather than pinning the wrong build silently. | **(b)** |
| D205 | ✅ **CORRECTED 2026-09-01. Unset now means `"latest"`, not "no pin"** — `_PtbHarness.PTB_DEFAULT_VERSION = "latest"` for all four wrappers. | The row previously said unset meant "the adapter installs its default (newest) build". **That was factually wrong**, and the mistake cost a 2-hour run. Harbor with `version=None` does not install anything: `_installed_codex_satisfies_version` (`codex.py:330-334`) only checks whether the CLI *exists* and skips the install when it does — so the version baked by `standard.def` becomes permanent. Upstream does the opposite: all four `agents/<a>/solve.sh` open with `update_agent_cli.sh <cli>`, which runs `npm install -g --prefix $HOME/.local <pkg>@latest` (`update_agent_cli.sh:56`) into a prefix `run_task.sh` puts AHEAD of the baked copy, and the opt-out `POST_TRAIN_BENCH_SKIP_CLI_UPDATE` is **commented out** at `example.env:31`. So upstream's agent CLI is *latest at run time* and `standard.def`'s pins are the offline fallback only ("falling back to pinned", `:60`). Symptom that exposed it: run 920879, gpt-5.6-sol, died at startup with `unknown variant 'max'` because `max` postdates codex 0.137.0. `verify_fidelity.py` now re-derives all of this from upstream and fails if either the solve.sh calls or the example.env opt-out change. | **(a)** |
| D206 | ⚠️ `cursor-cli` cannot be pinned at all. | `harbor/agents/installed/cursor_cli.py:339-348` curls `cursor.com/install` and never reads `self._version`. Moot — see D207. | **(b)** |

#### 5.2.4 cursor-cli

| id | decision | justification | tag |
|---|---|---|---|
| D207 | **Four wrappers, not five.** cursor-cli is not wrapped. | Confirmed 2026-08-24: cursor-cli is not supported by the model proxy, so take it out please?"* Checked rather than assumed: there is no hook to point it at a proxy even if it were — `CursorCli` declares no `MODEL_CONNECTION` (verified: `grep -c MODEL_CONNECTION cursor_cli.py` → **0**), so it inherits `BaseAgent.MODEL_CONNECTION = None` (`harbor/agents/base.py:66`) and has no `base_url_envs`; its run command (`:879-883`) passes no endpoint flag. The harness is a **candidate**, not a task property, so this changes none of the 28 tasks — we simply never register a cursor candidate. | **(d)** — tradeoff: upstream's `cursor_cli` rows are unreproducible. Accepted by scope decision. |
| D208 | The unit suite keeps a one-line tripwire asserting `CursorCli.MODEL_CONNECTION is None`. | So a harbor upgrade that adds a spec is caught rather than silently invalidating D207's reasoning. | **(d)** |
| D209 | `verify_fidelity.py:422-428` derives the wrapper roll-call from the module (`vars(ptb_harness)`) rather than listing it. | A new wrapper cannot be added without being checked. | **(d)** |

#### 5.2.5 Structural

| id | decision | justification | tag |
|---|---|---|---|
| D210 | `_PtbHarness` is a mixin placed **before** the real adapter in the MRO. | So `__init__` can mutate `kwargs` (version, effort kwarg, model slug) before `BaseInstalledAgent.__init__` consumes them — `CLI_FLAGS` kwargs are auto-extracted at `harbor/agents/installed/base.py:167-170` and resolved at `:570`, both inside that `super()` call. | **(b)** |
| D211 | All logging happens **after** `super().__init__`. | `self.logger` is set by `BaseAgent.__init__` (`harbor/agents/base.py:82`). | **(b)** |
| D212 | `_ptb_set` writes to **both** `self._extra_env` and `os.environ` (via `setdefault`, never overwriting). | `_extra_env` is what harbor's adapters resolve through; `os.environ` is what a CLI the adapter spawns inherits. Belt and braces. | **(d)** |
| D213 | `run()`'s signature is copied from `harbor/agents/base.py:205-210`. | Matched by all four adapters (`claude_code.py:1601`, `codex.py:1333`, `gemini_cli.py:780`, opencode's `run`). | **(b)** |
| D214 | Every decision (version + source, model rewrite, effort, each credential) is logged at construction, and the clause application is logged at `run()`. | Residual-drift mitigation: if someone adds e.g. `gpt-5.7-responses` to `harbor_map_responses_slug` upstream, our copy will not know — but the mismatch is visible in every trial log. | **(d)** |
| D215 | ✅ **FIXED 2026-09-01. `_ptb_resolve_version` is now defined exactly once** (`ptb_harness.py:239`); the byte-identical duplicate has been removed. | The row previously recorded the duplicate as outstanding. Re-verified by `grep -n 'def _ptb_resolve_version'`, which returns a single hit. | **(a)** |

---

## 6. Behavioural deviations — things that differ at run time regardless of file

These are the deviations you cannot see by reading any one file.

| id | behaviour | upstream | port | tag |
|---|---|---|---|---|
| B1 | **API-key isolation** | `apptainer -c --cleanenv` + an allowlist that is `agents/<agent>/api_keys.json` ∪ `info.json:required_api_keys` (`run_task.sh:96-121`). `OPENAI_API_KEY` reaches the agent only on `arenahardwriting`/`healthbench`. Eight of the seventeen agent variants in `commit.sh` are `_non_api` and list `[]`. | Only the benchmark half is reproduced. The Kaggle executor exports `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, `LLM_API_KEY` and their base URLs unconditionally (`entrypoint-common.sh:312-324`). **The agent's environment is strictly looser than upstream's.** | **(b)** |
| B2 | **The agent trace has no timestamps** | Piped through `src/utils/timestamp_lines.py` (`run_task.sh:250`). | Harbor's tee does not stamp. Parsers tolerate it; the judges lose wall-clock times. | **(b)** |
| B3 | **gemini reads a different file** | `solve.sh:7` passes `--output-format stream-json`, so `solve_out.txt` is the structured stream. | `gemini_cli.py:852-854` passes no output-format flag; the tee is 3 lines of chatter. `test.sh` prefers `gemini-cli.trajectory.jsonl` (3 780 parsed lines from the same trial) — **a different schema**, rendered partly as pretty-printed JSON by `gemini_parser.py`. | **(b)** |
| B4 | **The trace holds the agent alone** | `run_task.sh:250` runs `check_cuda`, `system_monitor.sh` and the agent in one pipeline, so all three are interleaved in `solve_out.txt`. | The first two run in `preflight.sh` before harbor starts the agent. `system_monitor.log` is copied out separately. | **(b)** |
| B5 | **`time_taken.txt` is derived, not measured** | `with_record_the_time` wraps the agent process. | `now − CREATION_DATE`. Agent start → verifier start, including harbor teardown. Not cosmetic: a missing/malformed value scores the baseline. | **(b)** |
| B6 | **One verifier image instead of two** | `gpt_5_5.sif` for the judges, `vllm_debug.sif` for the evaluation. | The union, with gpt_5_5's codex pin and three unused CLIs dropped. | **(b)** |
| B7 | **No copy-up on the HF cache** | `fuse-overlayfs` with a scratch upper layer: the agent can rewrite anything in the cache and the write is discarded afterwards. | A link farm over a read-only mount. New repos work; **rewriting a file inside an already-cached repo fails.** | **(b)** |
| B8 | **A verifier timeout exists** | None. Up to 9 attempts × 28 800 s. | 7 200 s, `--verifier-timeout-sec` to change. | **(d)** |
| B9 | **The judge auth failure is not fatal** | `run_task.sh:274-280` `exit 1` before the agent even starts. | `test.sh:388-391` prints the HTTP code and continues; the consequence surfaces at scoring (exit 1, no reward). | **(d)** |
| B10 | **`judge_output_<id>.txt` is never written** | `judge_lib.sh:186` runs `parse_trace.py --agent codex` on every judge's raw JSON. | Omitted. Scoring unaffected; the human-review artefact is missing. §8/U12. | **(a)** |
| B11 | **`OPENAI_API_KEY` reaches the evaluation on all 7 benchmarks** | Empty on 5 of 7 (`run_task.sh:82` unsets, `:83-85` restores for two). | One shared verifier environment; the judges need the key. §8/U3. | **(b)** |
| B12 | **The prompt's `{datetime}` is build time** | Regenerated per job (`get_prompt.py:55`). | Baked once. A task pushed in August and run in October tells the agent it is August. | **(b)** |
| B13 | **`update_agent_cli.sh` never runs; `cli_version.txt` never exists** | Runs from `solve.sh` and records the npm build. | Harbor installs and pins the CLI (`--ak version=`). The applied version is in the trial's `config.json`. | **(b)** |
| B14 | **`solve.sh` never runs at all** | The agent entry point; receives `$PROMPT`/`$AGENT_CONFIG`, does per-agent setup, launches the CLI. | Harbor's adapter does all of it. Per-agent logic that lived in `solve.sh` (e.g. `glm5`/`qwen3max` remapping their provider key into `ANTHROPIC_API_KEY`) has no analogue — but neither does any of those agents, which are out of scope. | **(b)** |
| B15 | **`system_monitor.sh` is never killed** | `kill $MONITOR_PID` when the agent exits (`run_task.sh:250`). | Runs until the container stops. §8/U7. | **(b)** |
| B16 | **The result directory is missing `output.log`, `error.log`, `prompt.txt`, `cli_version.txt`, `judge_output_*.txt`** | All five present (`AGENTS.md`, "Results Structure"). | Harbor owns the first two, the third is `instruction.md`, the fourth is B13, the fifth is B10. | **(b)** |
| B17 | **A run with no `metrics.json` *and* no judge verdicts scores nothing instead of the baseline** | `collect.py` scores the baseline (the missing-verdict check is unreachable without metrics). | `score_run.py` exits 1 first. §8/U15. | **(a)** |
| B18 | **Marker files appear in the result dir on the failure paths** | No analogue. | `PTB_LOOKUP_JUDGE_FIRED`, `MISSING_JUDGEMENTS`. Deliberate: make the reason visible in the artifact dir when there is no reward to explain it. | **(d)** |
| B19 | **`.preflight_hf_ok` / `.preflight_cuda_ok` appear in the sandbox home** | No analogue. | Healthcheck idempotency (D84). They land in `$EVAL_DIR/home/` via the artifact. | **(b)** |
| B20 | ✅ **SUPERSEDED 2026-09-01. `nAttempts = 1` per run, and the sweep is SCHEDULED THREE TIMES.** | Every published cell is a mean over n=3 (`scripts/utils.py:HARDCODED_AGENT_MAP`, `aggregate.py:84-90`). | This now **matches upstream's mechanism rather than degrading it.** Upstream's three trials are three INDEPENDENT sweeps: `commit.sh` has no attempt loop, it is re-run under a new `EXPERIMENT_NAME`, and `utils.py` groups the resulting directories so `aggregate.py` can take mean and sample stddev per `(model, benchmark)` cell. Kaggle's `nAttempts` is serial inside one session (3 x 10 h = 30 h, past both the 24 h requirement and the session cap) and would measure within-machine variance where upstream measures across-machine, across-time variance. So: **1 attempt per run, three scheduled runs, aggregate afterwards** — decided by the team leads 2026-09-01. The value is now emitted explicitly in the push body rather than left implicit (`push_tasks.py:build_body`), because server precedence is `request.NAttemptsNullable ?? config.yaml ?? 1` and an explicit value is final. ⚠️ Upstream is not always exactly 3: of its 23 grouped methods, 14 have three sweeps and 9 have two. | **(a)** |

---

| B21 | **The agent CLI floats to `@latest` on every run, for all four harnesses.** | Upstream: `agents/<a>/solve.sh` -> `update_agent_cli.sh` -> `<pkg>@latest`, opt-out commented out in `example.env:31`. | Reproduced via `PTB_DEFAULT_VERSION = "latest"`. An earlier revision left it unpinned, which harbor interprets as "keep whatever is installed" and which therefore froze every harness at its `standard.def` fallback. ⚠️ Consequence, inherited from upstream and NOT introduced here: **runs are not byte-reproducible** — two runs days apart can use different CLI builds. | **(a)** |

## 7. Faithfully reproduced upstream quirks — deliberate, not bugs

These all look wrong. They are 1:1 and must stay that way.

| id | quirk | evidence | why it is preserved |
|---|---|---|---|
| Q1 | **Escaped backticks in the prompt.** `prompt.txt:1` writes `` \`{model}\` `` and nothing ever interprets the backslash, so the agent literally reads `` \`Qwen/Qwen3-1.7B-Base\` ``. Also in the Decontamination Tool section (`get_prompt.py:81-88`) and the eval-API note (`:105`). | `prompt.txt:1`, `:8`, `:16-17`, `:29`; `get_prompt.py:81-95`, `:105` | The prompt is produced by *running* `get_prompt.py`, so the port could not "fix" it without editing upstream. `verify_fidelity.py:407-413` diffs the output. |
| Q2 | **The judges see no model weights.** `delete_hf_models.py` runs on `$JOB_DIR/task` **before** the judges (`run_task.sh:343` vs `:373`), deleting `final_model` along with everything else. | `run_task.sh:335-345`, `judge_lib.sh:75-80` | That is exactly why `prepare_judge_sandbox` hands them `final_model/config.json` separately — the architecture-identity check needs only the config. `test.sh:204-219` preserves the order. |
| Q3 | **`run_task.sh:454`'s comment says "up to 2 attempts" and the code passes 3** (`:482`). | `run_task.sh:454`, `:482` | The code wins. `test.sh:499-512` reproduces both, with the discrepancy called out in a comment so no future reader "fixes" it. `verify_fidelity.py:133-136` re-parses `[4, 3, 2]`. |
| Q4 | **Two different sources for the benchmark name.** `get_prompt.py:17-22` reads `benchmark.txt`; `get_judge_prompt.py` reads `info.json["benchmark"]`. | `get_prompt.py:19`, `get_judge_prompt.py` | Both files are staged, both at upstream paths, so both lookups resolve. Unifying them would be a behavioural change. |
| Q5 | **`resources.json` prefetches 14 models for a 4-model sweep** (~160 GB, of which ~134 GB is 9 models referenced by no code at all). | `containers/download_hf_cache/resources.json` | The cache is **deliberately agent-facing** — prompt rule 6 (`prompt.txt:27`) invites the agent to use it, and the prompt does not enumerate the contents, so the agent discovers them by listing the hub dir. Decision: ship the full cache; agents had access to it, so 1:1 fidelity requires it. |
| Q6 | **`general_judge/judge.conf`'s header comment contradicts `collect.py`.** It claims collect.py *"raises an error listing every flagged run"*; collect.py never loads the verdict. | `general_judge/judge.conf` header vs `collect.py:19-22`, `utils.py:562-563` | The file is copied byte-for-byte (§2) and `score_run.py` follows the **code**, not the comment. Upstream's inconsistency, faithfully carried. |
| Q7 | **The rerun-verdict preference is reproduced even though there is no rerun pipeline here.** | `utils.py:410-430` | Literal transcription costs nothing and keeps `score_run.py` diffable against `utils.py`. |
| Q8 | **`codex` is installed in the agent image even though nothing in that image uses it.** | `standard.def:43`, comment *"only for the judge, the other ones are installed in solve.sh"* | Removing it would make the image differ from upstream for no reason. |
| Q9 | **`update_agent_cli.sh` is staged into a sandbox that never runs it.** | `run_task.sh:128` | Upstream stages it. Not staging it would be a change for taste. |
| Q10 | **`--search` is passed to all four judges** including ones that would never use it. | `judge_lib.sh:167` | Confirmed 2026-08-24: required, not optional. 1:1 means it stays. |
| Q11 | **`gpt_5_5.def` names the image after GPT-5.5 but pins codex 0.124.0 and is used for GPT-5.4/5.6 judges.** | `judge_lib.sh:37`, `gpt_5_5.def:45` | The port takes the pin, not the name. |
| Q12 | **The `-c --cleanenv` sandbox still gets `PATH` handed to it explicitly** rather than inheriting the image's. | `run_task.sh:232` | Reproduced only partially — see D28/§8/U1. |

---

## 8. Undocumented deviations found by this audit

**This section is the audit result: code that diverges from upstream with no
comment saying so.** Every item below was found by reading the port against
upstream line by line, not by reading the port's own comments. Ordered by
consequence.

| id | what | where | consequence | recommended |
|---|---|---|---|---|
| **U15** | `score_run.py` checks the PTB-lookup verdict and the missing-verdict set **before** `load_metrics`; `collect.py:116-141` checks them **after**, inside the guarded block, with an explicit comment (`:120-125`) saying the ordering is deliberate. `score_run.py`'s own docstring (lines 13-26) records collect.py's order correctly, so the implementation contradicts its own documentation. | `score_run.py:206-259` | **Behavioural.** A run with no `metrics.json` and no judge verdicts (agent crashed, judges then had nothing to look at) exits 1 with **no reward** here; upstream scores the **baseline**. Same inversion for a firing PTB-lookup judge on a metrics-less run. | Move both checks inside the `try` after `load_metrics`, or state the inversion explicitly and justify it. |
| **U12** | `collect_judge_output` omits `judge_lib.sh:186`'s `parse_trace.py --agent codex "$out_dir/judge_output_<id>.json" -o "$out_dir/judge_output_<id>.txt"`. | `test.sh:367-377` | **Artefact loss.** `judge_output_*.txt` — listed in `AGENTS.md`'s "Results Structure" — is never produced. Scoring is unaffected (nothing downstream reads it). The port's own warning text at `test.sh:375` even points readers at the `.json`, where upstream points at the `.txt`. | Add the one line; everything it needs (`parse_trace.py`, the repo root cwd) is already present. |
| **U3** | `[verifier.env] OPENAI_API_KEY` is unconditional, so it reaches `evaluate.py` on all seven benchmarks. Upstream (`run_task.sh:82-85`, `:411`) supplies it to the evaluation on only two. | `build_tasks.py:408` | Unknown. No code path on the other five is known to read it — **UNVERIFIED**. If any does, it would now reach the model proxy rather than nothing. | Either document it as forced (one shared verifier env), or gate it: the judges could read `CODEX_API_KEY` alone and `OPENAI_API_KEY` could be emitted only for the two benchmarks. |
| **U1** | The agent image's `PATH` omits `/home/ben/.local/bin`, which `run_task.sh:232` and `judge_lib.sh:157` both add. | `template/environment/Dockerfile:38`, `template/tests/Dockerfile:40` | Anything the agent installs into `~/.local/bin` is not on `PATH`. Low impact today. | Add the element to both `ENV PATH` lines. It is upstream's value, not an invention. |
| **U16** | ~~`_ptb_resolve_version` is defined **twice** with byte-identical bodies; the second shadows the first.~~ ✅ **CLOSED** — the duplicate was removed 2026-09-01 (D215). Verified 2026-09-04: exactly one definition, `ptb_harness.py:291`. This row is stale and kept only so the number is not reused. | `ptb_harness.py:291` | None. | Done. |
| **U8** | `tests/Dockerfile`'s header block (lines 89-111) documents a `/tests/eval/`, `/tests/judges/<name>/`, `/tests/judge_tools/`, `/tests/eval_task_info/`, `/tests/dot_codex/` layout that does not exist. The real layout is `/tests/repo/src/…`. | `template/tests/Dockerfile:89-111` | Documentation only, but it is the file a reader opens to learn the layout. | Rewrite the block to match `stage_tests()`. |
| **U7** | `system_monitor.sh` is started by `preflight.sh` and never killed. `run_task.sh:250` kills it when the agent exits. | `preflight.sh:120-123` | The monitor keeps sampling through the agent phase *and* whatever follows in the same container, growing `system_monitor.log`. The copied log may therefore cover a longer window than upstream's. | Note it, or have the healthcheck stop writing after the budget. |
| **U13** | `run_evaluation` adds `2>&1` to the `final_eval_N.txt` redirect. `run_task.sh:424` redirects stdout only. | `test.sh:459` | `final_eval_N.txt` contains strictly more than upstream's. Nothing parses it. Arguably necessary here (D105). | One-line comment. |
| **U9** | `${EVAL_DIR}/prompt.txt` (`run_task.sh:76`) is never written. | `test.sh` | A result dir produced by this port lacks a file upstream's always has and `AGENTS.md` documents. | Copy `instruction.md` into `$EVAL_DIR/prompt.txt`, or state the omission. |
| **U10** | `tree` gains a `find -maxdepth 3 \| sort` fallback with no comment — and because the verifier image installs no `tree`, **the fallback is the production path**, producing different output from upstream's. | `test.sh:194-198` | Diagnostic only. | Comment it, or `apt-get install tree` in the verifier image so upstream's output is reproduced. |
| **U2** | `AGENT_CONFIG` (`run_task.sh:239`) is not in `[environment.env]`. | `build_tasks.py:383-396` | None known — only `solve.sh` read it, and it *is* carried in `metadata.json` for the judges. | One-line comment next to the `PROMPT` reasoning. |
| **U11** | `judge_lib.sh:106-108`'s `forced_login_method = "chatgpt"` append is dropped, covered only by the blanket "minus the auth.json handling". | `test.sh:297-301` | Correct behaviour (it would break API mode) but not called out. | Add it to the DIVERGENCE block above the function. |
| **U14** | `echo $(cat …)` becomes a guarded `cat`, changing both whitespace handling and the no-attempt case. | `test.sh:528-530` | Diagnostic only. | One-line comment. |
| **U4** | `DEBIAN_FRONTEND=noninteractive` becomes a persistent runtime `ENV`, where `standard.def:8` exports it only during `%post`. | `template/environment/Dockerfile:35`, `template/tests/Dockerfile:39` | Affects any `apt` the agent or judge runs. Harmless. | One-line comment, or move it to per-`RUN` scope. |
| **U5** | `chmod -R a+rw /home/ben` has no upstream analogue and no comment. | `template/environment/Dockerfile:115` | The sandbox home is world-writable, which upstream's host-owned `JOB_DIR` is not. | Comment it. |
| **U6** | The apptainer flag block (`-c`, `--cleanenv`, `--pid`, `--no-init`, `--writable-tmpfs`, `--bind ${JOB_TMP}:/tmp`, `--nv`) is nowhere enumerated as dropped-with-a-Docker-equivalent. | `template/environment/Dockerfile` | Documentation. The one property genuinely lost (`--cleanenv`) *is* documented, under divergence 10. | Add the reconciliation table (§4.2 D72 above is one). |

### 8b. Citation drift (documentation defect, not behaviour)

Found while verifying every `file:line` reference in this port. **The
`judge_lib.sh` citations are systematically wrong by +6 to +31 lines**, and
several `utils.py` / `collect.py` / `set_env_vars.sh` / `get_judge_prompt.py`
references are off by 3-20. The checkout has not moved since the port was
written (`judge_lib.sh` last touched at `e53de0d`, HEAD is `65b499d1`, 2026-08-13),
so these are transcription errors, not drift.

| cited as | actually at | cited in |
|---|---|---|
| `judge_lib.sh:37` ALL_JUDGES | `:31` | `config.py:111`, `test.sh:245` |
| `judge_lib.sh:35-36` defaults | `:35-36` | `config.py:120`, `test.sh:241` |
| `judge_lib.sh:37` JUDGE_CONTAINER | `:37` | `config.py:206` |
| `judge_lib.sh:42-66` load_judge_conf | `:42-63` | `test.sh:248` |
| `judge_lib.sh:68-92` prepare_judge_sandbox | `:68-85` | `test.sh:277` |
| `judge_lib.sh:94-112` setup_judge_codex_auth | `:94-109` | `test.sh:294` |
| `judge_lib.sh:114-128` build_judge_prompt | `:114-120` | `test.sh:303` |
| `judge_lib.sh:130-179` run_judge_exec | `:130-168` | `test.sh:312` |
| `judge_lib.sh:181-197` collect_judge_output | `:181-197` | `test.sh:362` |
| `judge_lib.sh:188-196` warn branch | `:191-196` | README, `test.sh:400` |
| `set_env_vars.sh:24` HF_HOME_NEW | `:23` | `config.py:168`, `build_tasks.py:384` |
| `get_judge_prompt.py:57` API_JUDGE_EXCEPTION_BENCHMARKS | `:54` | `config.py:124` |
| `run_task.sh:236-237` VLLM_API_KEY | `:235` | `build_tasks.py:387` |
| `run_task.sh:410-411` overlay bind into the eval | `:410` (HF_HOME) + `:417` (bind) | README, `build_tasks.py:328` |
| `utils.py:398` load_metrics | `:379` | `score_run.py:15`, `:111` |
| `utils.py:414/417/437` judgement helpers | `:407/410/432` | `score_run.py:41`, `:48`, `:62` |
| `utils.py:472/490/520` optional-flag helpers | `:~478/487/517` | `score_run.py:81`, `:92`, `:19` |
| `utils.py:551` missing_required_judgements | `:556` | `score_run.py:17`, `:164` |
| `utils.py:568` judgement_to_cell | `:581` | `score_run.py:21`, `:151` |
| `utils.py:611` load_time_taken | `:627` | `score_run.py:22`, `:133` |
| `collect.py:152` the `except` | `:147` | `score_run.py:24` |
| `collect.py:186` baseline substitution | `:183` | `score_run.py:26` |

`verify_fidelity.py` re-derives these values by **regex**, not by line number, so
none of this is caught by the existing checks and none of it affects behaviour.
It does undermine the port's central claim of citation-level rigour.

### 8c. Other staleness in the port's own docs

| what | where | correct value |
|---|---|---|
| *"Only six files have no upstream counterpart"* | `README.md:44-48` | **Eleven** per task: `task.toml`, both `Dockerfile`s, both `docker-compose.yaml`s, `preflight.sh`, `test.sh`, `score_run.py`, `strip_web_search.py`, `metadata.json`, `judge_model_map.json` — plus the two shared `agents/*`. Recounted 2026-09-01; the README and the §1 table have been corrected. The two `requirements-direct.txt` copies are byte-identical to `containers/requirements-direct.txt` and count as upstream copies, not authored files. |
| *"~4450 checks"* | `README.md:50` | scratch.md §6 records 4 535 after the HF-cache change. |
| `build_tasks.py` docstring: *"the only new code is … task.toml, the two Dockerfiles, preflight.sh, test.sh and score_run.py"* | `build_tasks.py:13-15` | Same omission. |
| `stage_tests()` takes `model_to_train`, `agent`, `agent_config` and uses none of them | `build_tasks.py:238-248` | Dead parameters (the values reach it inside `metadata`). Harmless. |
| `resources(indent_header)` — the parameter is a TOML section header, not an indent | `build_tasks.py:370` | Misnomer. |

---

## 9. Deviations we did NOT make, and why

Each of these was considered, is defensible, and was rejected.

| id | rejected change | why rejected | tag |
|---|---|---|---|
| N1 | **Raise harbor's 120 s artifact timeout.** | It would need ~250 s for 8 GB plus headroom, and the `120` lives inside the pinned `harbor==0.21.0` package (`harbor/environments/base.py:995`) — patching a third-party dependency, the same class of problem as editing the shared image. Moving the cache mount out of `/home/ben` removes the timer instead of betting on a bigger one, because an artifact with no `exclude` takes the uncapped `docker compose cp` path entirely (D51). | **(d)** |
| N2 | **A bare `ln -s /mnt/hf-cache /home/ben/hf_cache`.** | Shrinks the artifact just as well, but makes the whole cache read-only — which the overlay never was. Prompt rule 6 invites the agent to use the cache and anything it downloads mid-run has to land somewhere. Link farm instead (D79). | **(d)** |
| N3 | **Reach the wrapper via `--custom-harness` / `OverrideHarnessVersionCustomSlug`.** | It does reach `--agent` (`BenchmarkContainerEnvVarBuilder.cs:246-251`), but it is a **task-version** property and `CreateBenchmarkTaskFromHarborKaggleDatasetsHandler.cs:190-195` forces `CandidateType = ModelVersions` whenever it is set — pinning one harness across all 28 tasks. PTB rows vary both model and harness, so it would have destroyed the benchmark. **This overturned an earlier recommendation.** | **(b)** |
| N4 | **Add four `agents.ptb_harness:*` glob branches to `container/harbor-base/entrypoint-common.sh`.** | This is what `oneshot`/`tool_loop` did (`:346`, `:390`) and would be the cleanest fix. **Ruled out by project policy**: no edits or PRs to shared files; `container/` is off limits and everything must be self-contained. The wrapper closes the gap instead, deferring to any value the entrypoint already set so the branch, if it ever lands, makes the wrapper a no-op. | **(d)** |
| N5 | **Port `run_judges.sh:82-92`'s "no trace file → `exit 1`" guard.** | An earlier pass *did* add it. It was wrong: `run_judges.sh` is the standalone re-run tool; the sweep path is `run_task.sh`, which has no guard and no `set -e`. Upstream with no trace still evaluates and still scores. Removed. | **(a)** |
| N6 | **Re-stamp the trace with `timestamp_lines.py` in the verifier.** | It would write the *verifier's* clock onto every line — worse than no timestamp — and harbor reads `claude-code.txt` back itself (`claude_code.py:862-865`), so mutating it is unsafe. | **(d)** |
| N7 | **Declare a `mean` metric in `task.toml` before using `n_attempts > 1`.** | Planned, then retracted 2026-08-26: the shipped feature supplies metrics to harbor's `JobConfig.Metrics` **from the push request at invoke time**, not from `task.toml`, and defaults to mean when omitted. `task.toml` stays untouched — one fewer deviation from upstream. | **(b)** |
| N8 | **Use `merge_results_from_runfiles` to emit a ± confidence interval.** | It works (`kaggle_benchmarks/kaggle/serialization.py:571-580` routes an `(mean, radius)` tuple into `{"numericResult": {"value", "confidenceInterval"}}`), but `ComputeAggregatedRuns` (`GetUnifiedBenchmarkLeaderboardHandler.cs:404-409`) builds `new NumericResult { Value = … }` — value only, `grep ConfidenceInterval` → zero hits. So the ± would appear only on the 28 leaves, which upstream never publishes as columns, and be missing on the 7 benchmark parents and the root, which is exactly where upstream shows one. Backwards from what we need. The platform's intended shape is `n_attempts`. | **(b)** |
| N9 | **Ask for a session with N GPUs so `n_attempts=3` runs in parallel.** | The wrong thing to ask for: `SelectPoolNameAsync` is a binary accelerator-pool route, not a count, and the platform's own follow-up is one VM per attempt. Explicitly: do not raise it. | **(b)** |
| N10 | **Add a `pass@1` / `score` / `exact_match` fallback chain to `load_metrics`.** | PR #8 has one ⚠️ **(c)**. `utils.py:393-395` raises `KeyError` on anything but `accuracy`, and the caller turns that into the baseline. The fallback is a divergence from main, not upstream behaviour. | **(a)** |
| N11 | **Reproduce upstream's `run_id >= NEWER_JUDGES_MIN_RUN_ID` gate on the PTB-lookup requirement.** | The threshold (17 400 000) was chosen above every run id that existed on 2026-07-16. Every run this port produces is newer, so the gate is always true; reproducing it would mean inventing a run id. | **(a)** |
| N12 | **Edit `judge.conf` to hold the proxy slug directly.** | It would keep `test.sh` simpler, but `judge.conf` is a copied file and the port never edits a copy. `judge_model_map.json` + a post-`source` remap keeps the conf byte-identical (D181). | **(d)** |
| N13 | **Drop `--search` from the judges.** | Confirmed 2026-08-24: required, not optional. Upstream passes it to all four (`judge_lib.sh:167`). | **(a)** |
| N14 | **Set `TORCH_BACKEND=cu128` as the default so the image builds anywhere.** | The default stays `auto` — upstream's value — so a substitution is always a visible, deliberate act. | **(d)** |
| N15 | **Install `tree` in the verifier image so `test.sh:195` matches upstream's output.** | Not done; the `find` fallback is used instead. In hindsight this is the cheaper fidelity win — see §8/U10. | **(d)** |
| N16 | **Register a `cursor-cli` candidate.** | The model proxy does not support it, and `CursorCli` has no `MODEL_CONNECTION` to point at one (D207). | **(d)** |

---

## 10. Known unfaithful areas that cannot be fixed on this platform

These are not deviations we chose. They are properties of the target that make an
exact reproduction impossible. They must be stated wherever the numbers are.

| id | area | upstream | here | can it be fixed? |
|---|---|---|---|---|
| X1 | **The `_non_api` subscription-auth rows.** Of the 17 agent variants in `commit.sh`, **8 are `_non_api`**, including every codex row but the base one. They authenticate with OAuth tokens bind-mounted at run time (`auth.json`, `oauth_token`, `cursor_auth.json`, `grok_auth.json`; `run_task.sh:143-177`) and declare `api_keys.json = []`. | Subscription/OAuth. Upstream clearly thought the distinction mattered — it maintains separate agent directories for it. | Kaggle routes **everything** through the model proxy, an API path. Subscription-auth rows cannot be reproduced at all. Unknown whether it changes behaviour (rate limits, effort defaults, quota truncation) — **UNVERIFIED**. | **No.** There is no subscription-auth path on Kaggle. |
| X2 | **The `±` on aggregated cells.** Every published number is `mean ± stddev` over n=3. | `scripts/utils.py:HARDCODED_AGENT_MAP` maps each row to three run dirs; `aggregate.py:84-90` computes mean and sample stddev. | **Partly resolved 2026-09-01.** With three scheduled sweeps (X3) the three per-cell samples now EXIST, so mean and stddev are computable offline exactly as `aggregate.py` does it. What is still missing is the *display*: the 7 benchmark parents and the root drop any confidence interval on aggregation (`GetUnifiedBenchmarkLeaderboardHandler.cs:404-409` builds value only; `grep ConfidenceInterval` -> 0 hits). Deferred by project decision, 2026-09-01: the score display is to be fixed later. So the number is recoverable; the leaderboard rendering is the open item, and it is a platform change, not a port change. | **Data yes, display no** — tracked with the team. |
| X3 | ✅ **RESOLVED 2026-09-01. n=3, as three separately scheduled sweeps.** | Three runs per cell. | Three. The earlier entry recorded n=1 as forced (*"running it multiple times isn't an option"*, 2026-08-26) and warned that our number was **one sample from the distribution upstream reports the mean of**. The team leads reversed that on 2026-09-01: press run three times at `nAttempts=1`. That is upstream's own mechanism (independent sweeps, not in-session attempts), so the statistic is now the same one upstream reports. Cost: 28 x 3 = 84 runs, and the `ScheduledTasksLimit = 20` cap (`BenchmarkBatchScheduler.cs:91`, enforced as tasks x candidates) means each press must be chunked or carry ops flag `BYPASS_BATCH_SCHEDULE_BENCHMARK_TASK_RUNS_LIMIT` (26767). | **Yes — done.** |
| X4 | **Mapping weights may be dropped.** `factors.json`'s weights go on the leaf→parent mapping (`benchmark_types.proto:1107-1108 optional double weight = 10`). | `aggregate.py:107-122` multiplies by the factor and sums; weights sum to 1.0. | ⚠️ **Unverified that `weight` persists through `CreateBenchmarkTaskVersionMappings`** — the mock sent 2.0/1.0 and the response did not echo them back. If it silently does not stick, `WeightNullable ?? 1` turns the root into a plain average and every weight is ignored, **with no error**. Mappings are **immutable** (re-sending returns `400 already mapped`, delete 404s), so this must be proven before the real push. | **Unknown.** Must be tested, and can only be tested once. |
| X5 | **`aggregation_type` is not ours to set.** | n/a | `CreateBenchmarkHandler.cs:240-245` hardcodes it from `BenchmarkType` and never reads the request, so we always get `Unspecified`. With `Unspecified` the root looks for its own runs (finds none), the 7 parents never average, and the root emits no value — **28 leaf numbers, 7 empty parents, 1 empty root, no error raised**. The ask is: set `AVERAGE` on the 7 parents and `WEIGHTED_AVERAGE` on the root. | **No, not by us.** Requires the Benchmarks team. `Instructions.md:583` says coordinating here is the documented path, not an escalation. |
| X6 | **L4, not H100 80GB HBM3.** | `single_task.sub:12` pins `TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"`. | `HarborSessionProtoBuilder.cs:32` hardcodes one accelerator pool, `production-exeunit-no-fs-l4-1`. `container/harbor-base/kaggleenv/README.md`: *"there is no multi-GPU pool, and KKB staging has no accelerator pool at all, so GPU runs can only be validated in prod."* A 10 h budget on an L4 is not a 10 h budget on an H100 — **the agent gets materially less compute for the same wall clock**, which changes the result, not just the runtime. | **No, not by us.** Top blocker. `task.toml`'s `gpu_types = ["H100"]` records the requirement; the platform ignores it today. |
| X7 | **CPU / RAM / disk are capped by the outer KKB session.** | 16 cores, 128 GB, 400 GB. | `HarborSessionProtoBuilder.cs:76-81` hardcodes 4 cores / 32 GB / 80 GB. `task.toml`'s values configure the **inner** container and are capped by the outer session — requesting 16 cores inside a 4-core session yields 4. Gaps: **4× CPU, 4× RAM, 5× disk.** | **No, not by us.** Platform-side, same ask as X6. |
| X8 | **Wall clock.** | 10 h agent + up to 72 h of evaluation retries. | `BenchmarkTaskVersionLimits.cs:12` `MaxSessionExecutionTime = 12 h`, which must cover both. That is what forces D42's 7 200 s verifier cap. | **No, not by us.** |
| X9 | **The row → CLI build mapping is still open.** | Upstream pins exact builds per row: 8 `claude-code`, 8 `codex`, 4 `gemini-cli`, 1 cursor. A published row is not fully specified without one. | The mechanism exists (D202) but the table does not. Some rows may be unrecoverable: rows whose run dirs date to Dec 2025 – Jan 2026 predate pinning entirely (`standard.def:35-37` had no versions at all at `@0aaa86b`, `@774716d`), so the build is whatever npm `latest` was on image-build day. | **Partly.** The recoverable rows can be mapped; the rest cannot. |
| X10 | **`aggregated_std_*.csv` — upstream's own measured per-cell spread — has not been pulled.** | Shipped upstream. | Would quantify how far a single run can plausibly land from the published mean, per benchmark, which is exactly the caveat X3 needs. Offered, not requested. | **Yes, but not done.** |
| X11 | **The verifier's HF cache route is verified structurally, never on a real Kaggle run.** | Direct bind (`run_task.sh:417`). | Compose fragment (D167-D173). If it silently does not resolve, `evaluate.py` re-downloads its dataset — fine for six benchmarks, but **`gpqamain` uses the gated `Idavidrein/gpqa`** and the verifier has no `HF_TOKEN`. | **Yes**, by running it. Watch `gpqamain` first. |
| X12 | **The link farm has never met a real 160 GB cache.** | n/a | `preflight.sh` was exercised against a synthetic tree (read-through, dot-entry skipping and writability all confirmed) but never against the real mount, and never with an agent that downloads a new dataset mid-run. | **Yes**, by running it. |
| X13 | ✅ **CLOSED 2026-09-01 by prod run 871144.** All four judges ran end to end against the real prompts, `judge.conf` files and `judge_lib.sh` transcription, and wrote four well-formed `judgement_*.json` with trace-grounded justifications. Zero `Google Search Grounding` / `responses_websocket` / `307` / `turn.failed` in the verifier stdout. | Four verdicts per run — delivered. | The question about the prod token was the right one and the answer was NO: the platform-minted token lacked native-tool access, so `--search` 403'd while a workstation token worked. Fixed by granting `MODEL_PROXY_NATIVE_TOOL_ACCESS` (4831) — see D137b. The other half was the websocket 307 — see D137a. | **Answered.** Both halves measured, not inferred. |
| X14 | **`tests/repo/.env` is easy to lose in transit.** | n/a | `sanitize_trace.py:14` hardcodes the name and `PostTrainBench/.gitignore:173` is `.env`. The dataset flow copies files directly and is unaffected, but a `harbor-git-v1` push that commits the generated tasks would silently drop it and `parse_trace.py` would exit 1 after writing `solve_parsed.txt`. `verify_fidelity.py` checks the generated tree, not the uploaded one. | **Yes**, by checking after any packaging change. |

---

## 11. Appendix — generator-side files

Not shipped in a task; listed for completeness.

| file | what it is | deviation status |
|---|---|---|
| `build_tasks.py` | The generator. Every "new file" above is written by it. | Not a deviation itself; all of its decisions are itemised above. Docstring staleness noted in §8c. |
| `config.py` | The ~30 constants that must be transcribed: the matrix, resources, retry counts, token tables, judge list, container pins, sandbox paths, baselines/factors loaders. | Each block carries a provenance comment naming the upstream file. `verify_fidelity.py:68-233` re-derives every one from its source. Two off-by-one citations noted in §8b. |
| `verify_fidelity.py` | The check. Three kinds: (A) constants re-derived from upstream, (B) every copied file byte-compared, (C) `instruction.md` regenerated and diffed — plus `task.toml` parsed through harbor's own `Task` model so a schema mistake is caught here and not on the platform. Currently ~4 500 checks across the 28 tasks. | No upstream counterpart. It re-derives by **regex**, not by line number, so §8b's citation drift is invisible to it. |
| `template/environment/Dockerfile.smoke` | D73-D74. | Never shipped in a real task; stamped `smoke_build = true` in two files. |
| `template/tests/Dockerfile.smoke` | D101. | Same. |
| `tests/test_ptb_harness.py` | 33 unit tests covering the clause literal, parser dispatch, import resolution, the version chain, credential filling, effort injection, the codex slug rewrite, and the `CursorCli.MODEL_CONNECTION is None` tripwire. | No upstream counterpart. |

### D224 — sweep `storage_mb = 81920` instead of upstream's 409600

> ⛔ **RETIRED 2026-09-09 — see D236.** This deviation caused the loss of
> a full opus-5 wave (8/8 runs). `storage_mb` is now upstream's 409600.
> The reasoning below is kept as the record of why it was taken and how
> its estimate failed.

**Approved by the user, 2026-09-03.**

| | |
| :--- | :--- |
| **Upstream** | `single_task.sub:11` — `request_disk = 400G` (409600 MB). |
| **Port** | `run_posttrainbench.STORAGE_MB = 81920` (80 GB), overriding D14 for the sweep only. `verify_fidelity:519` flags this on all 28 tasks; the flag is expected. |
| **Class** | **(b)** — platform constraint, deliberately accepted. |

**Why.** 400 GB exceeds the default H100 machine's 88 GB
(`HarborSessionProtoBuilder.cs:91-100`) and routes the run to the scarcer
1 TB pool. The sweep needs **14 concurrent** H100s; one task on the large-disk
pool is routine, fourteen is a capacity gamble against a grant we were given
specifically for this.

**Why it is believed safe.** `storage_mb` is scratch disk, not VRAM and not
RAM — both of those already match the original exactly (H100 80 GB HBM3, and
`memory_mb = 131072`). The 346 GiB HF cache does not land on this disk, for
two independent reasons: the benchmark pools mount the dataset with **gcsfuse**
rather than copying it (landed 2026-09-03), and `preflight.sh` then **symlinks**
the cache into place rather than copying it again. What remains on the scratch
disk is checkpoints and working files for a model of at most 4B parameters — a
full-precision 4B checkpoint is ~16 GB, so several fit inside 80 GB. The 3 h
e2e run 872756 completed at this setting.

> ⚠️ **This paragraph previously carried a false claim, and that claim cost
> real time.** It read: *"Kaggle dataset mounts are read-only NFS and do not
> consume `storage_mb`."* That is true of Filestore-backed pools. Harbor's
> benchmark pools are `production-exeunit-no-fs-*` — they have no Filestore, so
> before gcsfuse a mount performed a **full GCS prefix copy onto the exeunit's
> local disk**, which `storage_mb` sizes. Under that behaviour this deviation
> was not merely inefficient, it was the direct cause of the dataset-mount
> failures: every mount whose contents exceeded the declared disk failed, and
> every one that fit succeeded (7/7). The "ceiling between 45 and 89 GB" that
> was chased for hours was simply `storage_mb`. The original's
> `request_disk = 400G` was never padding — it is sized to hold the cache.
> The claim is true again today, but only because gcsfuse removed the copy;
> **if gcsfuse is ever rolled back, this deviation breaks every run.**

**The risk being taken.** Upstream asked for 400 GB and we do not know what
for. If an agent retains every intermediate checkpoint, or writes a large
synthetic dataset, 80 GB could fill. A run that dies on a full disk at hour 8
costs a full GPU-day. This is unmeasured; the first pass should be checked for
disk-pressure failures before passes 2 and 3 commit to the same setting.

### D225 — the original's `unset` of other providers' API keys is not reproduced

**Approved by the user, 2026-09-03.**

| | |
| :--- | :--- |
| **Original** | `agents/claude/solve.sh:2-3` — `unset GEMINI_API_KEY`, `unset CODEX_API_KEY` before launching the CLI. Each `agents/*/solve.sh` strips the other providers' credentials. |
| **Port** | Not reproduced. `entrypoint-common.sh:312-324` exports `OPENAI_*`, `ANTHROPIC_*` and the Google keys unconditionally, and we do not unset any of them. |
| **Class** | **(b)** — the mechanism the original protects with this does not exist here. |

**Why it does not carry over.** The original's keys are real vendor
credentials, so unsetting them physically removes the agent's ability to reach
OpenAI or Gemini. Under the model proxy every one of those variables points at
the *same* metered endpoint, so unsetting them removes no capability -- the
proxy is the boundary, not the key.

**What is left unprotected, and what covers it.** The agent could still ask the
proxy for a model other than the one it was scheduled with. That is exactly
what the API-usage judge detects, and `collect.py:167-183` (transcribed in
`score_run.py:24-26`) drops the cell to the zero-shot baseline when it fires.
So the original's `unset` is belt-and-braces on top of a check we already run.

**Residual risk.** Our agent runs with more providers reachable than the
original intended. If the API-usage judge is ever removed or weakened, this
deviation stops being covered and should be revisited.

### D226 — `verify_fidelity` did not audit the original's `solve.sh`

**Not a deviation in the artefact — a gap in the verification, recorded so the
class of bug is visible.**

`verify_fidelity.py:326-329` reads each `agents/<harness>/solve.sh` but asserts
exactly one property of it: that the file still contains `update_agent_cli.sh`.
Nothing enumerates the rest. `claude/solve.sh` is five lines and only one was
covered, which is how `BASH_MAX_TIMEOUT_MS=36000000` went unnoticed through
4,847 checks.

The structural cause: the suite compares artefacts we GENERATE against the
original. `solve.sh` is a file we deliberately do not reproduce -- Harbor's
adapter launches the CLI instead -- so nothing in the design required
accounting for what it does. Files we replace wholesale are a blind spot.

Resolved for the two items found (`BASH_MAX_TIMEOUT_MS` added to `agent_env`;
`unset` waived as D225). **The other three harnesses' `solve.sh` files have not
been audited line by line.**

### D227 — the verifier's codex CLI is 0.146.1, not `gpt_5_5.def`'s 0.124.0

**A behavioural deviation: a different CLI build runs all four judges.**

| | |
| :--- | :--- |
| **Original** | `containers/gpt_5_5.def:44-48` pins `@openai/codex@0.124.0`. That is the image `judge_lib.sh:37` (`JUDGE_CONTAINER="gpt_5_5.sif"`) runs every judge in. |
| **Ours** | `template/tests/Dockerfile:96-100` pins `@openai/codex@0.146.1`. The other three npm pins are `gpt_5_5.def`'s, verbatim. |
| **Class** | **(b)** — forced. 0.124.0 cannot reach the model proxy at all. |

**Why it is forced.** codex's built-in `openai` provider prefers a websocket
transport and dials `wss://$OPENAI_BASE_URL/responses`. The proxy registers
that route only with a trailing slash: slashless → **307**, with slash → 401.
codex 0.124.0 has no HTTPS fallback and does not follow the redirect, so it
burns five reconnects and fails the turn — the failure D137a diagnosed. codex
**≥ 0.146** falls back to HTTPS when the wss upgrade redirects, which is what
makes the judges reachable.

0.146.1 is not an arbitrary newer build: it is the version Kaggle registers as
its canonical codex harness (`HARNESS-VERSIONS.md`, ver id 2,
`codex-0.146.1`). The platform only permits approved harness versions, so the
set of builds available to us is that list, and 0.146.1 is the only member of
it that clears the 307.

**What it supersedes.** D137a's four-flag custom `model_providers.kproxy`
block was the fix for this on 0.124.0 — a custom provider skips the websocket
path entirely. The version bump removes the need for it, so `test.sh:373-376`
now uses the stock single flag:

```bash
BASE_URL_ARG=()
if [ -n "${JUDGE_BASE_URL:-}" ]; then
    BASE_URL_ARG=(-c "model_providers.openai.base_url=\"${JUDGE_BASE_URL}\"")
fi
```

**⚠️ `verify_fidelity.py` asserts the deviation rather than catching it.**
`:213` documents it, `:220` asserts the pin *is* 0.146.1, and `:234-238`
exempts codex from the pin-matching sweep while warning if `gpt_5_5.def`'s own
pin ever moves off 0.124.0. This is deliberate — an unexempted sweep would
fail on every run — but it means the fidelity suite will never re-surface this
gap. This section is the only record.

**Residual risk.** Judge behaviour is not guaranteed identical across a
22-minor-version span of the codex CLI. The judges' prompts, `judge.conf`
files, flag line and output parsing are all byte-faithful (D137, D139); only
the binary interpreting them differs. Prod run 871144 (X13) shows all four
judges producing well-formed, trace-grounded verdicts on this build.

### D228 — `KAGGLE_HARBOR_AGENT_OVERRIDE` used to DISPATCH, not to pin

**⚠️ Read this before changing how tasks are pushed or scheduled.**

| | |
| :--- | :--- |
| **Documented use** | `Instructions.md:427` — *"Pin this whenever Step 1b told you the benchmark is reported against a specific harness."* One harness, fixed for the task. |
| **Our use** | Point it at `agents.ptb_harness:PtbHarness`, a class that reads `KAGGLE_AGENT_HARNESS` at run time and *becomes* the harness that was actually scheduled. The pinning mechanism, used to avoid pinning. |
| **Class** | **(d)** — the field names are canonical, the pattern is ours. |

**Why it exists.** Three requirements together: one task set serving every
(harness, model, effort) combo; the original's non-interactive clause applied
iff the harness is Claude (`get_prompt.py:61`); and no HarnessVersion
registration. The clause needs our code running at run time, and without
registration the override is the only way to get our code to run -- but the
override also fixes the harness, which breaks requirement one. The dispatcher
undoes that side effect.

**What it depends on**, none of it promised by the docs:

1. `entrypoint-common.sh:254` resolves
   `AGENT="${KAGGLE_HARBOR_AGENT_OVERRIDE:-${KAGGLE_AGENT_HARNESS:-...}}"`, so
   our class loads.
2. The platform *still emits* `KAGGLE_AGENT_HARNESS` from the scheduled Agent
   even though the override supersedes it (`:241-243`). **This is the fragile
   one.** It would be entirely reasonable for the platform to stop emitting a
   value it knows is being overridden -- and the day it does, the dispatcher
   has nothing to read and every run fails.
3. Harbor's factory calls `agent_class(...)` with a `cast` and no isinstance
   check (`factory.py:17-21, :130-131`), so `__new__` may return an instance of
   a different class.

**The trap for a reader.** The task definition says the harness is pinned to
`PtbHarness`. It is not pinned to anything -- the scheduled agent slug decides.
Anyone auditing the definition will draw the wrong conclusion.

**Verified 2026-09-03.** `candidateType = agents` (zz-hroute-env); model wires
through from the slug (run 1066402 recorded `claude-opus-5-default`); the
dispatcher runs end to end (run 1066920 completed).

**The documented alternative, if this ever breaks.** Register each wrapper as a
HarnessVersion whose `HarborSlug` is its import path, push with NO harness
override, and schedule agent slugs normally. The scheduled slug then selects
the class directly and every dependency above disappears. Needs the
`agentic-candidates-creation` flag (granted 2026-09-03). Precedent: `one-shot`,
`tool-loop-browse` and `aisi-inspect` all ship as custom import paths this way.
Runbook: `container/harbor-base/oneshot/README.md`, "Ship on Kaggle".

### D229 — `CUDA_VISIBLE_DEVICES = "0"` pins the agent to one GPU

**Approved by the user, 2026-09-04.**

| | |
| :--- | :--- |
| **Original** | Sets nothing. `single_task.sub` requests one `"NVIDIA H100 80GB HBM3"` and HTCondor hands the job a machine with exactly that one GPU, so the variable is unnecessary. |
| **Port** | `build_tasks.agent_env["CUDA_VISIBLE_DEVICES"] = "0"`, emitted into `[environment.env]` of all 28 tasks. |
| **Class** | **(b)** — platform constraint, deliberately accepted. |

**Why.** Kaggle's exeunit exposes *every* GPU on the node to the container —
`/dev/nvidia0` through `/dev/nvidia7` — even though the task declares
`gpus = 1`. `check_cuda.py:34` fails its first branch
(`device_count != expected_gpus`, i.e. 8 != 1), touches `cuda_not_available`
and exits 1. Because `preflight.sh` is wired as the `[environment.healthcheck]`
command, Harbor retries it three times and aborts the trial with
`HealthcheckError`.

Measured, 2026-09-04 sanity sweep: **8 of 8 runs died at t≈86 s**, identically,
across claude-code, codex and opencode. The run artifacts are unambiguous —
`artifacts/home/task/cuda_not_available` present (CUDA gate failed) and
`artifacts/home/.preflight_hf_ok` present (the HF cache mounted and linked
fine, so this is not a cache problem). Representative run: **1118002**.

This is a regression on the platform side, not a change on ours. Run **872756**
logged `GPU devices detected: /dev/nvidia0` — a single device — under the same
`gpus = 1` declaration, and completed with a real reward of 0.7498. The
8-device allocation appeared with the pool migration that brought gcsfuse.

**Why this form.** Setting the variable makes `torch.cuda.device_count()`
return 1, so **`check_cuda.py` passes completely unmodified**. The deviation is
one environment variable rather than a patch to the original's CUDA gate,
which is the smallest edit that resolves the failure.

**Why `"0"` is future-proof.** It does not mean "physical GPU 0". The NVIDIA
container runtime renumbers devices per container, so index 0 always denotes
the first GPU the container was given. If the platform later schedules one task
per GPU, each task's assigned device is its own index 0 and this value stays
correct — no task edit and no re-push (a re-push would mint a new `versionId`
and orphan the 28 leaf→parent leaderboard mappings; see `scratch.md:580`).

**⚠️ The alternative that was rejected.** Relaxing `check_cuda.py` to accept
`>= 1`, or deleting the check, is *not* equivalent and is actively unsafe. The
check is an assertion, not a mechanism: removing it leaves all 8 devices
visible, and HuggingFace `Trainer` defaults to `DataParallel` across every
visible GPU. Training would silently run on 8 GPUs at 8× the effective batch,
producing a score that looks valid but cannot be compared with the published
leaderboard. Failing loudly is strictly better than that.

**Why contention is not a risk today.** Scheduling is per *cluster*, not per
GPU: one cluster serves exactly one task, and a cluster cannot currently be
shared by two. So the task is the only tenant of its node and index 0 is
uncontended — there is nothing to race against. (The other seven GPUs on the
node are stranded, which is understood and accepted: the grant is counted in
clusters, 14 of the team's 26.) This matches the entrypoint's behaviour, which
sets up its own cgroup hierarchy and starts its own dockerd per run.

**The one condition that would change this.** If Kaggle later lets a single
cluster serve several tasks *without* per-container device renumbering, every
task on that node would pin to the same physical GPU. Renumbering is the
NVIDIA container runtime's default, so the expected outcome is that each task
still sees its own device as index 0 and this value stays correct — but if
that assumption ever breaks, this is the line to revisit.

### D230 — port bug (fixed): the dispatcher dropped opencode's reasoning effort

**Not a deviation from the original — a defect in our own code, found by audit
2026-09-04 and fixed the same day.** Recorded here because it silently
mis-scored a whole harness and the failure mode leaves no trace.

| | |
| :--- | :--- |
| **Symptom** | A run scheduled as `opencode-1.18.14-<model>-<effort>-reasoning` executed at opencode's **default** effort, while its agent slug advertised the requested one. |
| **Blast radius** | All 11 opencode slugs in `agent_options.md` carry a reasoning effort, so there was no way to schedule opencode and avoid it. claude-code, codex and gemini-cli were unaffected. |
| **Class** | port bug, not (a)–(d). |

**The defect.** `PtbHarness.__new__` composed the dispatched class from a
literal dict that restated the per-harness facts:

```python
{"PTB_AGENT_NAME": agent_name,
 "PTB_DEFAULT_VERSION": pinned,
 "PTB_PROXY_SUFFIX": "/genai" if agent_name == "gemini" else ""}
```

`PTB_EFFORT_KWARG` is absent, so every dispatched class inherited the base
default `"reasoning_effort"`. That is right for claude-code
(`claude_code.py:90`), codex (`codex.py:66`) and gemini-cli
(`gemini_cli.py:97`), and **wrong for opencode**, which declares
`CliFlag("variant", cli="--variant")` (`opencode.py:65`) — the platform agrees,
giving opencode its own arm at `entrypoint-common.sh:492-498`.

`_ptb_inject_effort` therefore set `reasoning_effort=<effort>`, no CLI
descriptor matched it, and `BaseAgent.__init__`'s `**kwargs` swallowed it. No
crash, no warning, no log line — the run simply scored at the default effort.

**Root cause: one fact in two places.** `PtbHarness_opencode` already carried
`PTB_EFFORT_KWARG = "variant"`; the dispatcher bypassed that class and
restated a subset of its attributes by hand. The fix removes the restatement —
the dispatched class now copies `_PTB_DISPATCHED_ATTRS` from the matching
per-harness class via `_PTB_BY_AGENT_NAME`, so the classes stay the single
source of truth and the two cannot drift again.

**Two further inconsistencies the same edit removed**, both latent rather than
observed:

* `PTB_PROXY_SUFFIX` was `""` for non-gemini where the base class documents
  `None` as "nothing to do" (`:254`, tested at `:422`). Inert only because the
  `_ptb_lookup` short-circuit at `:426` writes nothing — inert by accident.
* `PTB_DEFAULT_VERSION` was `pinned`, i.e. `None` for gemini, overriding the
  base `"latest"` that `PtbHarness_gemini` itself inherits. A dispatched gemini
  and a direct `PtbHarness_gemini` resolved different versions.

**Why the existing tests missed it.**
`test_effort_reaches_the_opencode_variant_flag` builds `PtbHarness_opencode`
directly, which was always correct; only the composed class was wrong. The
regression tests added with this fix drive the **dispatcher** —
`KAGGLE_AGENT_HARNESS` set, class built by `PtbHarness.__new__` — because that
is the code path production uses.

**No fidelity implication.** This restores the behaviour the platform
entrypoint would have supplied for a built-in agent name; it does not diverge
from the original, which never had a dispatcher at all.

### D232 — ship an empty `environment/.env` and `tests/.env` in every task

**Approved by the user, 2026-09-05, after the Benchmarks team confirmed the
mechanism from GCS artifacts and fleet telemetry.**

| | |
| :--- | :--- |
| **Original** | No such file. HTCondor/apptainer has no docker compose, so nothing looks for one. |
| **Port** | `build_tasks._compose_dotenv()` writes a 0-byte `.env` into both directories harbor uses as a compose `--project-directory`. |
| **Class** | **(b)** — platform constraint. Adds a file the original has no concept of; changes no behaviour. |

**The fault.** `docker compose` stats `.env` in its `--project-directory` on
every invocation, and harbor points that at the task's own directory
(`environments/docker/docker.py:620`), which on Kaggle is a gcsfuse mount.
Nothing in the port references the file — compose looks for it unconditionally.

With no file there it is a **negative** lookup, and the mount runs gcsfuse with
`--implicit-dirs` (`mount.go:147`). gcsfuse cannot answer `ENOENT` from the 404
alone: it must issue an `objects.list` with prefix `.env/` to rule out an
implicit directory. When that list call hits a transient 503 or timeout the
layer returns **`EIO` instead of `ENOENT`**. Compose silently ignores `ENOENT`
and treats `EIO` as fatal, exiting 1 — which under `set -euo pipefail` kills
the command before it has run.

That is how run **1120769** died installing claude-code on a path the install
command never mentions:

```
Healthcheck passed
Running command: ... npm install -g @anthropic-ai/claude-code@2.1.223 ...
Command failed (exit 1):
stdout: stat /kaggle/input/ptb-sweep-defs/ptb-gpqamain-qwen3-1-7b-base/environment/.env:
        input/output error
```

**Not a PostTrainBench bug.** Benchmarks queried
`kaggle_prod.BenchmarkRuns` across all Harbor runs: `NonZeroAgentExitCodeError`
at **0.15%–5.84% per day** fleet-wide (10,254 runs → 15 on 09-01; 1,198 → 70 on
09-02; 2,942 → 83 on 09-04). Our own sweep saw 1 of 14 (~7%), consistent with
the high end.

**Why an empty file fixes it.** The lookup becomes positive: GCS answers 200,
gcsfuse caches the metadata locally, and no `objects.list` is issued for that
path again. Compose parses an empty `.env` to zero variables, so nothing about
the environment changes.

**What it does NOT do.** It does not fix the gcsfuse fault, which still affects
every other negative lookup on the mount and every other Harbor benchmark. It
removes the one path compose touches on every exec — the path that was actually
killing us.

**Both directories, deliberately.** The agent env's project directory is
`environment/`; the separate verifier env's is `tests/`. Only the first was
observed failing, but the mechanism is identical and a second 0-byte file costs
nothing.

---

### D233 — copy `datasets--*` repos into the verifier's HF_HOME instead of linking them

`template/tests/test.sh`, the merge loop that builds `$TMP_HF_CACHE`.

**Symptom.** Every bfcl run scored its zero-shot baseline. For three of the four
models that baseline is `0.0`, so a crashed verifier and a model that genuinely
scored nothing produce the *same number* — the failure was invisible to a
reward-vs-baseline check. Observed on all 4 bfcl cells of the v4 sonnet-5
validation cycle (runs 1331139, 1331154, 1331159, and 1331164), and in at least
one case the agent had trained a working model first (~94% on its own eval).

```
OSError: [Errno 30] Read-only file system: '/tmp/hf_cache_90afd0/hub/
  datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/snapshots/
  1bf8bbc3c0e35d04d00339c223a3fd653aa195ac'
... nine retries ...
reward = 0.0  (baseline fallback: no usable metrics)
```

**Cause — a gap in D224's approximation, not in the original.** The original
bind-mounts a fuse-overlayfs *merged* tree at `$HF_HOME` (`run_task.sh:410-411`),
which is writable at every depth. Kaggle mounts input datasets read-only and
offers no overlay, so D224 approximates it with a symlink farm: a real
`$TMP_HF_CACHE`, real dirs one level down, and a symlink per repo into the `:ro`
mount. That makes *new top-level repos* writable, which is all any model needs —
and all four other benchmarks need. It does not make the inside of an existing
repo writable.

bfcl is the only benchmark that needs it. At the `inspect_evals` commit the
original pins in all eleven `containers/*.def`
(`06001a83e6d7c709c2ede0570dce7f1031a0bad8`), `bfcl.py:41-48` is the only one of
the five task definitions that passes a `revision=`:

```python
ds = dataset.hf_dataset(
    DATASET_PATH, split="train", sample_fields=record_to_sample,
    # main branch does not load cleanly into an HF dataset so we use a PR branch
    revision="1bf8bbc3c0e35d04d00339c223a3fd653aa195ac",
    name="exec_simple",
)
```

gsm8k, humaneval and aime2025 take the cached default revision; gpqa uses
`csv_dataset` over a URL and never touches the hub cache. So the failure needs
two conditions at once — repo present in the cache (hence a symlink) but the
requested revision absent — and only bfcl creates them. The `revision=` pin is
the original's, inherited through its own dependency pin; the crash is ours.

**Fix.** Copy `datasets--*` repos instead of linking them; keep the symlink for
everything else. `-L` resolves the `snapshots/ -> ../../blobs/` relative links
into real files, and `chmod -R u+w` clears the mount's mode bits. Datasets in
this cache are megabytes against 346 GiB of models, so the copy is seconds and
the size argument that forces symlinks for models does not apply.

**Verified** against a synthetic read-only mount: the model repo stays a
symlink, the dataset repo becomes a real dir whose files still read through to
the blobs, `mkdir snapshots/<rev>` succeeds inside the dataset copy, and the
same `mkdir` inside a symlinked repo still fails — i.e. both the bug and its
cure reproduce.

**Scope.** bfcl's 4 tasks only. The other 16 are unaffected and were not
re-pushed.

---

### D234 — resolve the agent trace by discovery, not from `metadata.json`

`template/tests/test.sh`, the trace-copy block and its two consumers.

**Symptom.** `solve_out.txt` and `solve_parsed.txt` were absent from **all 20
runs** of the first v5 validation cycle (0/20, confirmed against the session
archives). Every judge in every run opened its justification with the same
complaint, e.g. *"The authoritative solve trace is missing: neither
`../solve_parsed.txt` nor `../solve_out.txt` exists"*, and
`judgement_general.json` flagged `general_anomaly: true` on the whole sweep.

**Cause.** `test.sh` picked the trace filename with
`case "${AGENT}" in *claude*|*codex*|…`, where `${AGENT}` is
`metadata.json`'s `agent`. For this port that value is the `PROMPT_AGENT`
placeholder `"ptb"` (`build_tasks.py:62`), because one task is deliberately
shared by every candidate (D228) and so cannot name the harness that ran.
`"ptb"` matches no branch, `HARBOR_TRACE` stayed empty, and the copy was
skipped silently.

`build_tasks.py:55-61` had anticipated exactly this and documented the
separation — the placeholder for the prompt, the row's real name in
`metadata.json` for `parse_trace.py` and `get_judge_prompt.py`. The sweep
driver then passed `--agent ptb`, collapsing the two.

**Three consumers were affected, not one:**

| consumer | effect |
|---|---|
| trace copy | `solve_out.txt` never written |
| `parse_trace.py --agent` | `select_parser("ptb")` → `None`, so no `solve_parsed.txt` |
| `get_judge_prompt.build_agent_harness_clause` | fell to the generic branch, so judges were told "the **ptb** harness" instead of the claude/codex/gemini-specific hints that identify `claude-haiku-*` helper calls as the harness rather than agent-made API calls |

The third is the reason this is not merely lost auditing: an api-usage judge
that cannot recognise harness traffic can flag a clean run, and a flagged run
is scored at the baseline (`collect.py`). Not observed — `judgement_api`
passed on the runs inspected — but it was live risk.

**Fix.** Harbor writes exactly one trace, named for the adapter that ran, so
the filesystem is the reliable signal. If the metadata-derived name is empty
or absent, probe the known trace names in a fixed order (not a glob, so a
stray file cannot be mistaken for a trace), then derive the real upstream
agent name from whichever was found and use it for the copy, for
`parse_trace.py` and for the judge prompt.

**Restores fidelity rather than diverging from it:** upstream's `${AGENT}` is
the real agent name, and this makes the port's effective value the real agent
name too.

**Verified** by exercising the block against a synthetic `/logs/agent` for all
five harnesses plus the no-trace case: 6/6 resolve the correct agent and copy
(or correctly warn and fall back to the metadata value). Scoring is untouched
— the reward path is `evaluate.py → metrics.json → score_run.py` and never
reads the trace.

---

### D235 — the judges are told the model the platform scheduled, not the baked one

`build_tasks.py` (`[verifier.env]`) and `template/tests/test.sh`.

**Same class as D234, found by auditing for it.** `metadata.json`'s
`agent_config` is baked at build time, and one task is shared by every
candidate (D228), so it is correct only for the candidate it was built for.
The sweep bakes `claude-opus-5`; wave 2 runs gpt-5.5 and wave 3 gemini, so for
those the value is simply wrong.

**Why it matters.** `get_judge_prompt.build_agent_harness_clause(agent, model)`
uses it to tell the judges which banners, processes and token/cost records
belong to the harness rather than to an agent-made API call. Told the wrong
model, the api-usage judge can read legitimate harness traffic as a
third-party call — and a flagged run is scored at the baseline
(`collect.py`). Unlike D234 this had not yet had a chance to bite: waves 2 and
3 have not run on v5.

**Fix.** Export the platform's own `KAGGLE_AGENT_LLM` into the verifier
environment and prefer it, falling back to the baked value when the platform
does not set it (the verifier is a separate environment, so its presence is
not guaranteed). A mismatch is logged rather than silently corrected.

**Verified:** with `KAGGLE_AGENT_LLM=gpt-5.5` the verifier resolves `gpt-5.5`;
with it unset it falls back to the baked `claude-opus-5`. Scoring is untouched.

---

### D236 — D224 retired: `storage_mb` and the agent grace period restored to upstream

**2026-09-09. Two deviations removed, not added. The sweep now runs at
upstream's exact resource values and `verify_fidelity` passes with NO waivers.**

| field | was | now | upstream |
|---|---|---|---|
| `storage_mb` | 81920 (80 GB) | **409600** | `single_task.sub:11` `request_disk=400G` |
| `[agent] timeout_sec` | 36000 | **36300** | `run_task.sh:225` `$((NUM_HOURS*60+5))m` |

**What went wrong.** D224 cut the disk to 80 GB to stay inside the default
H100 machine's 88 GB, justified by an estimate: *"a full-precision 4B
checkpoint is ~16 GB, so several fit inside 80 GB."* opus-5 at max effort
wrote roughly six checkpoints plus AdamW optimizer states and vLLM compile
caches. The disk filled, ext4 went read-only, overlay2 entered a permanent EIO
state, and **8 of 8 completed runs died** — surfacing as
`failed to solve: mmap allocate error: input/output error` in the *verifier
build*, which is why it read as a Docker fault rather than a disk fault.

The agent timeout was the same shape: `build_tasks.py` already defaulted to
the faithful 36300 (D40), and the sweep driver overrode it with
`NUM_HOURS * 3600` = 36000, silently dropping the five-minute grace that lets
an agent finish writing `final_model` after its own timer fires.

**The pattern, which is the real lesson.** In both cases the port's own
config was faithful and the *driver* overrode it with a tighter number. A full
audit of every value the driver overrides (`--num-hours`,
`--agent-timeout-sec`, `--cpus`, `--memory-mb`, `--storage-mb`, `--gpu-types`,
gpus) now shows **0 mismatches**.

**Why not the alternative fix.** Coaching the agent to prune checkpoints
(`save_total_limit=1`, LoRA) was rejected: the original's `prompt.txt` says
nothing about disk, storage or checkpoints, so adding that would change agent
behaviour and bias our numbers against the published leaderboard invisibly.
Giving the agent the disk the original gives it does not.

**Affordable** because a full machine is dedicated per run (confirmed
2026-09-09), so the 8-GPUs-per-host packing argument behind D224 no longer
applies.

**The waiver is gone from the gate, not left as a no-op.** A waiver that
covers nothing still suppresses its class, so a regression would keep the gate
green — which is precisely how an 80 GB disk survived a week of builds while
being flagged 20 times each.

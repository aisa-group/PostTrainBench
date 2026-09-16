#!/usr/bin/env python3
"""Static configuration for the Kaggle/Harbor port of PostTrainBench.

Every value here is transcribed from a specific line of the upstream
pipeline. The provenance comment on each block is the contract: if upstream
changes, this file is wrong until it is updated to match.

Nothing here reimplements logic that already exists in the repo. Prompts come
from `src/eval/general/get_prompt.py`, judge prompts from
`src/judges/get_judge_prompt.py`, the timer from `src/utils/create_timer.sh`
and the scoring rules from `scripts/collect.py` — the generator invokes those
directly so they cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

# Repo root: .../PostTrainBench
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# The sweep matrix
# ---------------------------------------------------------------------------
# Verbatim from src/commit_utils/commit.sh:8-23. Order preserved.

MODELS: list[str] = [
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
]

BENCHMARKS: list[str] = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "healthbench",
]


def model_key(model_to_train: str) -> str:
    """The bare model name used as a dict key throughout the repo.

    scripts/baselines.json, scripts/utils.py:EXPECTED_MODELS and
    collect.py's `walk_latest_runs` (which splits the run-dir name on "_"
    into benchmark/org/model/run_id) all key on the segment after the
    organisation slash.
    """
    return model_to_train.split("/")[-1]


# Slug prefix, fixed by project decision (2026-08-26) as
# kaggle-posttrainbench-{benchmark}-{model}. Slugs are permanent — nothing can
# be deleted ("Delete is not supported by the server yet") and a re-push orphans
# mappings — so this is fixed, not a default to be tweaked.
TASK_SLUG_PREFIX = "kaggle-posttrainbench"

# Display name for the benchmark entity itself (create_benchmark.py --name).
BENCHMARK_DISPLAY_NAME = "Kaggle PostTrainBench"


def task_slug(benchmark_id: str, model_to_train: str) -> str:
    """Kaggle-safe task slug: lowercase, dots and underscores to dashes."""
    bare = model_key(model_to_train).lower().replace(".", "-").replace("_", "-")
    return f"{TASK_SLUG_PREFIX}-{benchmark_id}-{bare}"


# ---------------------------------------------------------------------------
# Evaluation retry schedule
# ---------------------------------------------------------------------------
# src/run_task.sh:452-512. Three phases; phase 1 runs with no token override.
# The retry counts are the literal arguments to run_evaluation_with_retry.

EVAL_PHASE_RETRIES: tuple[int, int, int] = (4, 3, 2)

# run_task.sh:455-480 (phase 2 `case`)
MAX_TOKENS_PHASE2: dict[str, str] = {
    "aime2025": "--max-tokens 12000",
    "arenahardwriting": "--max-new-tokens 12288",
    "bfcl": "--max-tokens 12000",
    "gpqamain": "--max-tokens 12000",
    "gsm8k": "--max-tokens 3000",
    "healthbench": "--max-new-tokens 12288",
    "humaneval": "--max-tokens 3000",
}

# run_task.sh:485-510 (phase 3 `case`)
MAX_TOKENS_PHASE3: dict[str, str] = {
    "aime2025": "--max-tokens 8000",
    "arenahardwriting": "--max-new-tokens 8192",
    "bfcl": "--max-tokens 8000",
    "gpqamain": "--max-tokens 8000",
    "gsm8k": "--max-tokens 2000",
    "healthbench": "--max-new-tokens 8192",
    "humaneval": "--max-tokens 2000",
}

# run_task.sh:441 — `timeout --signal=TERM --kill-after=60s 28800s`, per attempt.
EVAL_ATTEMPT_TIMEOUT_SEC = 28800


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------
# src/judges/judge_lib.sh:31 (ALL_JUDGES) — order is execution order.

ALL_JUDGES: list[str] = [
    "data_contamination_judge",
    "api_usage_judge",
    "ptb_lookup_judge",
    "general_judge",
]

# judge_lib.sh:35-36
JUDGE_DEFAULT_MODEL = "gpt-5.4"
JUDGE_DEFAULT_REASONING_EFFORT = "xhigh"

# get_judge_prompt.py:57 — benchmarks whose grading legitimately calls a
# hosted LLM API, and which therefore receive OPENAI_API_KEY.
API_JUDGE_EXCEPTION_BENCHMARKS = {"arenahardwriting", "healthbench"}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
# src/commit_utils/single_task.sub — verified identical across all nine
# submit files in src/commit_utils/.
#
#   request_cpus   = 16
#   request_memory = 131072            # MB
#   request_disk   = 400G
#   request_gpus   = 1
#   requirements   = TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"

REQUEST_CPUS = 16
REQUEST_MEMORY_MB = 131072
REQUEST_DISK_MB = 400 * 1024
REQUEST_GPUS = 1
GPU_TYPES = ["H100"]

# Default agent budget. AGENTS.md's single-task example and
# scripts/utils.py:BUDGET_SECONDS (10 * 3600) both use 10 hours.
DEFAULT_NUM_HOURS = 10

# run_task.sh:225 — the agent phase is wrapped in
# `timeout ... "$((NUM_HOURS * 60 + 5))m"`, i.e. the budget plus a five
# minute grace period.
AGENT_GRACE_MINUTES = 5


# ---------------------------------------------------------------------------
# Sandbox layout
# ---------------------------------------------------------------------------
# run_task.sh:246-247 — `--home "${JOB_DIR}:/home/ben" --pwd "/home/ben/task"`.
# The prompt depends on this: rule 6 points at the home folder for the HF
# cache, and get_prompt.py's decontamination section references
# `../test_data.json` and `../contamination_check.py` relative to the task dir.

SANDBOX_HOME = "/home/ben"
SANDBOX_WORKDIR = "/home/ben/task"

# src/commit_utils/set_env_vars.sh:24 — HF_HOME_NEW, the in-sandbox cache
# path. Also hardcoded in containers/other_home_data/.codex/config.toml.
SANDBOX_HF_HOME = "/home/ben/hf_cache"

# Not an upstream path. Upstream has no cache *mount point* at all: it builds a
# fuse-overlayfs and binds the merged tree straight onto SANDBOX_HF_HOME
# (run_task.sh:184, :242). Kaggle mounts the cache as a read-only dataset, and
# the mount must land outside /home/ben or the [[artifacts]] copy of the
# sandbox home tries to drag 160 GB through a 120-second tar. preflight.sh
# links it back under SANDBOX_HF_HOME, so the agent-visible path is unchanged.
HF_CACHE_MOUNT = "/mnt/hf-cache"

# run_task.sh:236 — passed to every phase as a constant, not a secret.
VLLM_API_KEY = "inspectai"

# Not an upstream path: harbor's own per-trial agent log directory
# (harbor/models/trial/paths.py:36, `agent_dir = logs_dir / "agent"`), where
# every adapter tees the raw agent trace. Named here because build_tasks.py
# declares it as an artifact and template/tests/test.sh reads it.
AGENT_LOGS_DIR = "/logs/agent"


# ---------------------------------------------------------------------------
# Container package pins
# ---------------------------------------------------------------------------
# containers/standard.def (agent) and containers/vllm_debug.def +
# containers/gpt_5_5.def (verifier). See template/*/Dockerfile.

BASE_IMAGE = "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04"
VLLM_VERSION = "0.11.0"
FLASH_ATTN_VERSION = "2.8.3"
INSPECT_EVALS_COMMIT = "06001a83e6d7c709c2ede0570dce7f1031a0bad8"

# standard.def:44 — the agent image ships codex only ("only for the judge,
# the other ones are installed in solve.sh"). Harbor installs the agent's
# own CLI, so nothing else is baked in.
AGENT_CODEX_VERSION = "0.137.0"

# judge_lib.sh:37 sends every judge to gpt_5_5.sif; gpt_5_5.def:42 pins
# codex there.
VERIFIER_CODEX_VERSION = "0.124.0"

# general_judge/judge.conf — this judge npm-installs its own codex release
# into the sandbox home and runs that instead of the image's.
GENERAL_JUDGE_CODEX_VERSION = "0.144.5"


# ---------------------------------------------------------------------------
# Scoring, transcribed from scripts/
# ---------------------------------------------------------------------------

def load_factors() -> dict[str, float]:
    """scripts/factors.json — per-benchmark weights for the overall metric.

    scripts/aggregate.py:compute_weighted_metric averages each benchmark
    across models, multiplies by its factor and sums. The factors sum to
    1.0, so the overall score is a weighted average over the seven
    benchmarks — which is exactly the Kaggle root aggregation.
    """
    with open(REPO_ROOT / "scripts" / "factors.json") as f:
        return json.load(f)


def load_baselines() -> dict:
    """scripts/baselines.json — zero-shot and few-shot baseline scores.

    collect.py substitutes the *zeroshot* value for a cell whose
    contamination or API-usage judge fired, so the verifier needs this
    table to compute a faithful reward without the aggregation step.
    """
    with open(REPO_ROOT / "scripts" / "baselines.json") as f:
        return json.load(f)


def zeroshot_baseline(benchmark_id: str, model_to_train: str) -> float:
    """The fallback score collect.py applies to a judge-flagged cell."""
    baselines = load_baselines()["zeroshot"]
    key = model_key(model_to_train)
    if key not in baselines:
        raise KeyError(f"baselines.json has no zeroshot entry for {key!r}")
    if benchmark_id not in baselines[key]:
        raise KeyError(
            f"baselines.json zeroshot[{key!r}] has no entry for {benchmark_id!r}"
        )
    return baselines[key][benchmark_id]

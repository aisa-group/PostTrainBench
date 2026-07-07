"""Modal adapter: run the full PostTrainBench pipeline on modal.com.

Each benchmark run (agent x model x task) is a chain of three Modal Functions,
mirroring the phases of src/run_task.sh:

    run_agent  (standard image,   H100)  -> spawns
    run_judge  (standard image,   CPU)   -> spawns
    run_eval   (vllm_debug image, H100)

All phases write into the `ptb-results` Volume using the exact EVAL_DIR layout
that scripts/collect.py expects, so results pulled with `modal volume get`
feed the existing analysis scripts unchanged.

Usage: see src/modal_adapter/README.md. Deploy with
    modal deploy src/modal_adapter/app.py
then submit runs with src/modal_adapter/submit.py.
"""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "posttrainbench"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTER = "/repo/src/modal_adapter"

RESULTS_MOUNT = "/results"
HF_RO_MOUNT = "/hf_cache_ro"
HF_RW_MOUNT = "/hf_cache"
JOB_DIR = "/home/ben"

# 24h Modal cap minus slack for setup + artifact copy.
MAX_NUM_HOURS = 22

app = modal.App(APP_NAME)

# Volume v2: the HF cache holds far more files than v1's comfortable limits.
hf_cache = modal.Volume.from_name("ptb-hf-cache", create_if_missing=True, version=2)
results = modal.Volume.from_name("ptb-results", create_if_missing=True, version=2)

# One secret holds everything. API keys mirror single_task.sub's environment
# line; CODEX_AUTH_JSON / CLAUDE_OAUTH_TOKEN are optional and replace the
# auth.json / oauth_token files that run_task.sh copies from agents/<agent>/.
keys_secret = modal.Secret.from_name("posttrainbench-keys")

CUDA_BASE = "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04"

NODESOURCE = "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs"
UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"

# Puts uv on the default PATH for any subprocess (the .def files do this via
# a PATH prepend in %environment, which Image.env() can't express).
UV_SYMLINK = (
    "ln -sf /root/.local/bin/uv /usr/local/bin/uv"
    " && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx"
)

# Package list from containers/standard.def
STANDARD_PACKAGES = [
    "accelerate",
    "boto3",
    "bitsandbytes",
    "datasets",
    "evaluate",
    "lm-eval",
    "openai",
    "pandas",
    "scikit-learn",
    "shortuuid",
    "tokenizers",
    "transformers",
    "trl",
    "peft",
    "tiktoken",
    "inspect-ai",
    "matplotlib",
    "certifi",
]

# Note: the .def files also prepend /root/.local/bin (uv) to PATH; here the
# phase scripts do that at runtime instead, because Image.env() shell-quotes
# values, so "$PATH" would not expand.
COMMON_ENV = {
    "PYTHONNOUSERSITE": "1",
    "NO_PROXY": "localhost,127.0.0.1",
    "no_proxy": "localhost,127.0.0.1",
}


def _with_repo(image: modal.Image) -> modal.Image:
    """Mount the repo files the pipeline needs at /repo (runtime mount)."""
    return (
        image.add_local_dir(
            REPO_ROOT / "src",
            "/repo/src",
            ignore=["**/__pycache__", "**/*.pyc"],
        )
        .add_local_dir(
            REPO_ROOT / "agents",
            "/repo/agents",
            # Never bake subscription credentials into the image; they are
            # provided via the posttrainbench-keys secret instead.
            ignore=["**/auth.json", "**/oauth_token", "**/__pycache__"],
        )
        .add_local_dir(
            REPO_ROOT / "containers",
            "/repo/containers",
            ignore=["**/*.sif", "**/__pycache__"],
        )
    )


# Port of containers/standard.def
standard_image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.10")
    .entrypoint([])
    .apt_install("git", "wget", "curl", "build-essential", "tree", "rsync")
    .run_commands(
        NODESOURCE,
        "npm install -g"
        " @anthropic-ai/claude-code@2.0.55"
        " @openai/codex@0.79.0"
        " @google/gemini-cli@0.18.4"
        " opencode-ai@1.1.59",
        UV_INSTALL,
    )
    # The .def files pass --torch-backend=auto, which needs a CUDA driver at
    # build time to pick an index; Modal builders are CPU-only, so resolve
    # from PyPI instead (torch wheels bundle their own CUDA runtime).
    .uv_pip_install("vllm==0.11.0")
    # wheel/setuptools/psutil are required for --no-build-isolation installs;
    # the .def files get them implicitly from the Ubuntu python environment.
    .uv_pip_install("ninja", "packaging", "wheel", "setuptools", "psutil")
    .uv_pip_install(*STANDARD_PACKAGES)
    .uv_pip_install("flash_attn", extra_options="--no-build-isolation")
    # standard.def installs inspect_evals from HEAD, but current HEAD requires
    # Python >=3.11; pin the same commit vllm_debug.def pins (3.10-compatible).
    .run_commands(
        "git clone https://github.com/UKGovernmentBEIS/inspect_evals.git /opt/inspect_evals"
        " && cd /opt/inspect_evals"
        " && git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8"
        " && python -m pip install --no-cache-dir ."
    )
    .run_commands(UV_SYMLINK)
    .env(COMMON_ENV)
)

# Port of containers/vllm_debug.def
vllm_debug_image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.10")
    .entrypoint([])
    .apt_install("git", "wget", "curl", "build-essential", "rsync")
    .run_commands(
        NODESOURCE,
        "npm install -g"
        " @anthropic-ai/claude-code@2.1.34"
        " @openai/codex@0.98.0"
        " @google/gemini-cli@0.18.4"
        " opencode-ai@1.1.59",
        UV_INSTALL,
    )
    # The .def files pass --torch-backend=auto, which needs a CUDA driver at
    # build time to pick an index; Modal builders are CPU-only, so resolve
    # from PyPI instead (torch wheels bundle their own CUDA runtime).
    .uv_pip_install("vllm==0.11.0")
    .uv_pip_install(requirements=[str(REPO_ROOT / "containers" / "requirements-direct.txt")])
    # wheel/setuptools/psutil are required for --no-build-isolation installs;
    # the .def files get them implicitly from the Ubuntu python environment.
    .uv_pip_install("wheel", "setuptools", "psutil")
    .uv_pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .run_commands(
        "git clone https://github.com/UKGovernmentBEIS/inspect_evals.git /opt/inspect_evals"
        " && cd /opt/inspect_evals"
        " && git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8"
        " && python -m pip install --no-cache-dir .",
        # The patched inspect_ai fork must be installed last so it overrides
        # the inspect-ai pulled in by earlier dependencies.
        "git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git /opt/inspect_ai_vllm_stdout"
        " && python -m pip install --no-cache-dir /opt/inspect_ai_vllm_stdout",
    )
    .run_commands(UV_SYMLINK)
    .env(COMMON_ENV)
)

standard_runtime = _with_repo(standard_image)
vllm_debug_runtime = _with_repo(vllm_debug_image)


# ---------------------------------------------------------------------------
# Run spec helpers
# ---------------------------------------------------------------------------

# Same character set as run_task.sh: tr '/:[]' '____'
_SAFE_TR = str.maketrans({c: "_" for c in "/:[]"})


def build_spec(
    eval_task: str,
    agent: str,
    agent_config: str,
    model_to_train: str,
    run_id: int,
    num_hours: int = 10,
    num_gpus: int = 1,
    experiment_name: str = "",
    prompt_variant: str = "prompt",
) -> dict:
    return {
        "eval_task": eval_task,
        "agent": agent,
        "agent_config": agent_config,
        "model_to_train": model_to_train,
        "run_id": int(run_id),
        "num_hours": int(num_hours),
        "num_gpus": int(num_gpus),
        "experiment_name": experiment_name,
        "prompt_variant": prompt_variant,
    }


def eval_dir_rel(spec: dict) -> str:
    """EVAL_DIR relative to the results root; must match run_task.sh:13-24."""
    agent_config_safe = spec["agent_config"].translate(_SAFE_TR)
    model_safe = spec["model_to_train"].translate(_SAFE_TR)
    gpu_suffix = f"_{spec['num_gpus']}gpu" if spec["num_gpus"] > 1 else ""
    method = (
        f"{spec['agent']}_{agent_config_safe}_{spec['num_hours']}h"
        f"{gpu_suffix}{spec['experiment_name']}"
    )
    return f"{method}/{spec['eval_task']}_{model_safe}_{spec['run_id']}"


def _phase_env(spec: dict, local_eval_dir: str) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "EVALUATION_TASK": spec["eval_task"],
            "AGENT": spec["agent"],
            "AGENT_CONFIG": spec["agent_config"],
            "MODEL_TO_TRAIN": spec["model_to_train"],
            "NUM_HOURS": str(spec["num_hours"]),
            "NUM_GPUS": str(spec["num_gpus"]),
            "RUN_ID": str(spec["run_id"]),
            "POST_TRAIN_BENCH_PROMPT": spec.get("prompt_variant", "prompt"),
            "LOCAL_EVAL_DIR": local_eval_dir,
            "REPO": "/repo",
            "JOB_DIR": JOB_DIR,
            "HF_RO": HF_RO_MOUNT,
        }
    )
    return env


def _publish(local_eval_dir: str, spec: dict) -> None:
    """Copy the staged EVAL_DIR contents onto the results volume and commit."""
    dest = Path(RESULTS_MOUNT) / eval_dir_rel(spec)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        local_eval_dir,
        dest,
        dirs_exist_ok=True,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    results.commit()


def _fetch(rel_path: str, local_path: str) -> bool:
    """Copy a file/dir from the results volume to local disk. True if found."""
    src = Path(RESULTS_MOUNT) / rel_path
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, local_path, dirs_exist_ok=True, symlinks=True)
    else:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
    return True


def _log_attempt(spec: dict, phase: str) -> None:
    """Record each container attempt: preemptions/retries restart phases from
    scratch, and this makes that visible in the run's results directory."""
    dest = Path(RESULTS_MOUNT) / eval_dir_rel(spec)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    task_id = os.environ.get("MODAL_TASK_ID", "unknown")
    with open(dest / "attempts.log", "a") as f:
        f.write(f"{stamp} phase={phase} modal_task={task_id}\n")
    results.commit()


def _spawn_next(function_name: str, spec: dict) -> None:
    modal.Function.from_name(APP_NAME, function_name).spawn(spec)


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------


@app.function(
    image=standard_runtime,
    gpu="H100",
    cpu=8,
    memory=65536,
    # Ceiling only; submit.py tightens this per run via .with_options().
    # The inner `timeout` in phase_agent.sh is what actually enforces num_hours.
    timeout=84600,
    retries=modal.Retries(initial_delay=0.0, max_retries=2),
    single_use_containers=True,
    volumes={
        HF_RO_MOUNT: hf_cache.with_mount_options(read_only=True),
        RESULTS_MOUNT: results,
    },
    secrets=[keys_secret],
)
def run_agent(spec: dict) -> str:
    """Phase 1: let the CLI agent post-train the model (run_task.sh:13-188)."""
    rel = eval_dir_rel(spec)
    _log_attempt(spec, "agent")

    local_eval_dir = "/tmp/eval_dir_out"
    Path(local_eval_dir).mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        ["bash", f"{ADAPTER}/phase_agent.sh"],
        env=_phase_env(spec, local_eval_dir),
    )
    # Mirror run_task.sh: agent failure is not fatal; judge + eval still run.
    # Raising here would also trigger Modal retries, re-billing the GPU hours.
    Path(local_eval_dir, "agent_phase_exit_code.txt").write_text(f"{proc.returncode}\n")

    _publish(local_eval_dir, spec)
    _spawn_next("run_judge", spec)
    return rel


@app.function(
    image=standard_runtime,
    cpu=8,
    memory=16384,
    timeout=7200,
    volumes={
        HF_RO_MOUNT: hf_cache.with_mount_options(read_only=True),
        RESULTS_MOUNT: results,
    },
    secrets=[keys_secret],
)
def run_judge(spec: dict) -> str:
    """Phase 2: contamination judge (run_task.sh:190-219).

    Runs in a fresh container so nothing the agent did to its own container
    can leak into the judge -- the same guarantee apptainer's per-exec
    writable tmpfs gives on the cluster. The judge is code inspection (codex
    CLI over the task directory), so it gets no GPU.
    """
    rel = eval_dir_rel(spec)
    _log_attempt(spec, "judge")

    local_eval_dir = "/tmp/eval_dir_out"
    Path(local_eval_dir).mkdir(parents=True, exist_ok=True)

    # Reconstruct the job dir the way the cluster's judge sees it.
    Path(JOB_DIR).mkdir(parents=True, exist_ok=True)
    if not _fetch(f"{rel}/task", f"{JOB_DIR}/task"):
        print(f"WARNING: no task directory found under {rel}; judge has nothing to inspect")
        Path(JOB_DIR, "task").mkdir(parents=True, exist_ok=True)
    _fetch(f"{rel}/solve_parsed.txt", f"{JOB_DIR}/solve_parsed.txt")

    proc = subprocess.run(
        ["bash", f"{ADAPTER}/phase_judge.sh"],
        env=_phase_env(spec, local_eval_dir),
    )
    Path(local_eval_dir, "judge_phase_exit_code.txt").write_text(f"{proc.returncode}\n")

    _publish(local_eval_dir, spec)
    _spawn_next("run_eval", spec)
    return rel


@app.function(
    image=vllm_debug_runtime,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=82800,
    volumes={
        HF_RO_MOUNT: hf_cache.with_mount_options(read_only=True),
        RESULTS_MOUNT: results,
    },
    secrets=[keys_secret],
)
def run_eval(spec: dict) -> str:
    """Phase 3: evaluate final_model (run_task.sh:243-363)."""
    rel = eval_dir_rel(spec)

    # Idempotency across Modal-level retries/preemptions: the ladder in
    # run_task.sh short-circuits on an existing metrics.json; do the same here.
    if (Path(RESULTS_MOUNT) / rel / "metrics.json").exists():
        print(f"metrics.json already present for {rel}; skipping evaluation")
        return rel

    _log_attempt(spec, "eval")

    local_eval_dir = "/tmp/eval_dir_out"
    Path(local_eval_dir).mkdir(parents=True, exist_ok=True)

    if not _fetch(f"{rel}/final_model", "/work/final_model"):
        # Mirror the cluster: the ladder still runs (and fails fast per
        # attempt), so the run leaves the same artifacts either way.
        print(f"WARNING: no final_model found under {rel}")

    proc = subprocess.run(
        ["bash", f"{ADAPTER}/phase_eval.sh"],
        env=_phase_env(spec, local_eval_dir),
    )
    Path(local_eval_dir, "eval_phase_exit_code.txt").write_text(f"{proc.returncode}\n")

    _publish(local_eval_dir, spec)
    return rel


# ---------------------------------------------------------------------------
# HF cache seeding
# ---------------------------------------------------------------------------


@app.function(
    image=standard_runtime,
    cpu=8,
    memory=65536,
    timeout=86400,
    volumes={HF_RW_MOUNT: hf_cache},
    secrets=[keys_secret],
)
def seed_hf_cache(models_filter: list = None, datasets_filter: list = None) -> str:
    """Populate the ptb-hf-cache volume by running the repo's own download
    script inside Modal (no laptop upload). Optional filters select a subset
    of resources.json for cheap smoke seeding."""
    src_dir = Path("/repo/containers/download_hf_cache")
    work = Path("/tmp/seed")
    work.mkdir(parents=True, exist_ok=True)
    # download_resources.py resolves resources.json next to itself, so run a
    # copy with a (possibly filtered) resources.json beside it.
    shutil.copy2(src_dir / "download_resources.py", work / "download_resources.py")
    resources = json.loads((src_dir / "resources.json").read_text())
    if models_filter is not None:
        resources["models"] = [m for m in resources["models"] if m in set(models_filter)]
    if datasets_filter is not None:
        wanted = set(datasets_filter)
        resources["datasets"] = [d for d in resources["datasets"] if d["dataset"] in wanted]
    (work / "resources.json").write_text(json.dumps(resources, indent=2))

    env = dict(os.environ)
    env["HF_HOME"] = HF_RW_MOUNT
    subprocess.run(
        ["python", str(work / "download_resources.py")], env=env, check=True
    )
    hf_cache.commit()
    du = subprocess.run(
        ["du", "-sh", HF_RW_MOUNT], capture_output=True, text=True
    ).stdout.strip()
    print(f"Seeding complete. Cache size: {du}")
    return du


@app.local_entrypoint()
def seed_smoke():
    """Minimal cache for a cheap end-to-end test: one model + one dataset."""
    print(seed_hf_cache.remote(["Qwen/Qwen3-1.7B-Base"], ["openai/gsm8k"]))


@app.local_entrypoint()
def seed_full():
    """Full cache (hours; run with `modal run --detach`)."""
    print(seed_hf_cache.remote(None, None))


# ---------------------------------------------------------------------------
# Smoke checks (no benchmark spend)
# ---------------------------------------------------------------------------


def _smoke_report(image_name: str) -> dict:
    report = {"image": image_name}

    for mod in ("torch", "vllm", "inspect_ai", "transformers", "datasets", "trl"):
        try:
            m = __import__(mod)
            # str(): e.g. torch.__version__ is a TorchVersion, which would
            # fail to unpickle on a laptop without torch installed.
            report[f"import:{mod}"] = str(getattr(m, "__version__", "ok"))
        except Exception as e:  # noqa: BLE001
            report[f"import:{mod}"] = f"FAIL: {e}"

    for cli in ("claude", "codex", "gemini", "opencode", "uv"):
        try:
            out = subprocess.run(
                [cli, "--version"], capture_output=True, text=True, timeout=120
            )
            report[f"cli:{cli}"] = (out.stdout or out.stderr).strip().splitlines()[0]
        except Exception as e:  # noqa: BLE001
            report[f"cli:{cli}"] = f"FAIL: {e}"

    needed = [
        "/repo/src/run_task.sh",
        "/repo/src/eval/general/get_prompt.py",
        "/repo/src/eval/tasks/gsm8k/evaluate.py",
        "/repo/src/disallowed_usage_judge/get_judge_prompt.py",
        "/repo/src/utils/create_timer.sh",
        "/repo/agents/claude/solve.sh",
        "/repo/containers/other_home_data/.codex",
        "/repo/containers/delete_hf_models.py",
        "/repo/containers/download_hf_cache/resources.json",
    ]
    report["repo_files"] = {p: os.path.exists(p) for p in needed}

    prompt = subprocess.run(
        [
            "python", "src/eval/general/get_prompt.py",
            "--model-to-train", "Qwen/Qwen3-1.7B-Base",
            "--benchmark-id", "gsm8k",
            "--num-hours", "1",
            "--num-gpus", "1",
            "--agent", "claude",
        ],
        capture_output=True, text=True, cwd="/repo",
    )
    report["get_prompt"] = "ok" if prompt.returncode == 0 and "GSM8K" in prompt.stdout else f"FAIL: {prompt.stderr[-500:]}"

    for key in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "OPENCODE_API_KEY", "DASHSCOPE_API_KEY", "ZAI_API_KEY",
    ):
        report[f"secret:{key}"] = "set" if os.environ.get(key) else "EMPTY"

    try:
        report["hf_cache_ro_entries"] = len(os.listdir(HF_RO_MOUNT))
    except Exception as e:  # noqa: BLE001
        report["hf_cache_ro_entries"] = f"FAIL: {e}"

    # Results volume write/read round-trip.
    try:
        probe = Path(RESULTS_MOUNT) / "_smoke" / f"probe_{int(time.time())}.txt"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        results.commit()
        report["results_volume"] = probe.read_text()
        probe.unlink()
        results.commit()
    except Exception as e:  # noqa: BLE001
        report["results_volume"] = f"FAIL: {e}"

    # Symlink-farm mechanics on a toy tree (the fuse-overlayfs replacement).
    try:
        lower = Path("/tmp/farm_lower/hub/models--org--m")
        lower.mkdir(parents=True)
        (lower / "weights.bin").write_text("lower-content")
        upper = Path("/tmp/farm_upper")
        subprocess.run(
            ["bash", "-c", f". {ADAPTER}/hf_cache_lib.sh && hf_symlink_farm /tmp/farm_lower {upper}"],
            check=True,
        )
        assert (upper / "hub/models--org--m/weights.bin").read_text() == "lower-content"
        (upper / "hub/models--org--m/new_file.txt").write_text("upper-write")
        report["symlink_farm"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["symlink_farm"] = f"FAIL: {e}"

    return report


@app.function(
    image=standard_runtime,
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={
        HF_RO_MOUNT: hf_cache.with_mount_options(read_only=True),
        RESULTS_MOUNT: results,
    },
    secrets=[keys_secret],
)
def smoke_standard() -> dict:
    return _smoke_report("standard")


@app.function(
    image=vllm_debug_runtime,
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={
        HF_RO_MOUNT: hf_cache.with_mount_options(read_only=True),
        RESULTS_MOUNT: results,
    },
    secrets=[keys_secret],
)
def smoke_vllm_debug() -> dict:
    return _smoke_report("vllm_debug")


@app.function(image=standard_runtime, gpu="H100", cpu=4, memory=16384, timeout=900)
def gpu_smoke() -> dict:
    report = {}
    for script in ("check_cuda.py", "check_cuda_writing.py"):
        proc = subprocess.run(
            ["python", f"/repo/src/utils/{script}"], capture_output=True, text=True
        )
        report[script] = "ok" if proc.returncode == 0 else f"FAIL: {proc.stdout[-300:]} {proc.stderr[-300:]}"
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    report["nvidia-smi"] = smi.stdout.strip() or smi.stderr.strip()
    return report


@app.local_entrypoint()
def smoke_check():
    """CPU-only checks of both images, volumes, secrets, and repo mounts."""
    for fn in (smoke_standard, smoke_vllm_debug):
        report = fn.remote()
        print(f"\n===== {report.pop('image')} image =====")
        for k, v in report.items():
            print(f"  {k}: {v}")


@app.local_entrypoint()
def gpu_smoke_check():
    """One short H100 container (~a minute) running the repo's CUDA checks."""
    for k, v in gpu_smoke.remote().items():
        print(f"  {k}: {v}")

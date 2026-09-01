"""Generate Harbor tasks from the PostTrainBench sources.

Everything benchmark-specific is read from the PostTrainBench tree so the
Harbor tasks stay in lockstep with the condor pipeline (src/run_task.sh):

  - src/eval/tasks/<id>/info.json      benchmark name, required_api_keys,
                                        allowed/disallowed data examples
  - src/eval/tasks/<id>/benchmark.txt   official benchmark name (prompt)
  - src/eval/tasks/<id>/test_data.json  test set for the decontamination tool
                                        (gitignored; see
                                        src/judges/test_data_download/)
  - src/eval/general/get_prompt.py      the agent prompt (rendered by calling
                                        it, so the Harbor instruction is
                                        byte-for-byte the condor prompt)
  - src/judges/judge_tools/             contamination checker given to the agent
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ADAPTER_NAME = "POSTTRAINBENCH"
TEMPLATE_DIR = Path(__file__).parent / "template"

# PostTrainBench repo root
POSTTRAINBENCH_ROOT = Path(__file__).parent.parent.parent
TASKS_ROOT = POSTTRAINBENCH_ROOT / "src" / "eval" / "tasks"
JUDGE_TOOLS_DIR = POSTTRAINBENCH_ROOT / "src" / "judges" / "judge_tools"
GET_PROMPT = POSTTRAINBENCH_ROOT / "src" / "eval" / "general" / "get_prompt.py"

# Benchmarks present in src/eval/tasks/ that are deliberately not offered as
# Harbor tasks.
#   aime2026: upstream's test-data downloader has no entry for it, so
#             run_task.sh cannot run it either (test_data.json is mandatory).
SKIP_BENCHMARKS = {"aime2026"}


@dataclass
class BenchmarkInfo:
    task_id: str            # e.g. "gsm8k"
    benchmark_name: str     # e.g. "GSM8K (Grade School Math 8K)" (info.json)
    required_api_keys: list[str]  # provider keys the benchmark's grading needs


@dataclass
class ModelInfo:
    model_id: str          # HuggingFace model ID, e.g., "Qwen/Qwen3-1.7B-Base"
    short_name: str        # Short name for task IDs, e.g., "qwen3-1.7b"


def _discover_benchmarks() -> dict[str, BenchmarkInfo]:
    """All benchmarks with an info.json under src/eval/tasks/, minus SKIP_BENCHMARKS."""
    found = {}
    for info_file in sorted(TASKS_ROOT.glob("*/info.json")):
        task_id = info_file.parent.name
        if task_id in SKIP_BENCHMARKS:
            continue
        info = json.loads(info_file.read_text(encoding="utf-8"))
        found[task_id] = BenchmarkInfo(
            task_id=task_id,
            benchmark_name=info["benchmark"],
            required_api_keys=list(info.get("required_api_keys", [])),
        )
    return found


BENCHMARKS: dict[str, BenchmarkInfo] = _discover_benchmarks()

MODELS = {
    "qwen3-1.7b": ModelInfo(model_id="Qwen/Qwen3-1.7B-Base", short_name="qwen3-1.7b"),
    "qwen3-4b": ModelInfo(model_id="Qwen/Qwen3-4B-Base", short_name="qwen3-4b"),
    "smollm3-3b": ModelInfo(model_id="HuggingFaceTB/SmolLM3-3B-Base", short_name="smollm3-3b"),
    "gemma3-4b": ModelInfo(model_id="google/gemma-3-4b-pt", short_name="gemma3-4b"),
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: int = 10,
        agent_name: str = "claude",
    ):
        """
        Args:
            output_dir: Directory where Harbor tasks will be generated.
            num_hours: Agent time budget in hours (default: 10).
            agent_name: PostTrainBench agent name passed to get_prompt.py
                (`--agent`). It only affects agent-specific prompt clauses
                (e.g. the non-interactive note added when the name contains
                "claude"); Harbor's claude-code agent corresponds to "claude".
        """
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.agent_name = agent_name
        self.posttrainbench_root = POSTTRAINBENCH_ROOT

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _task_src(benchmark_id: str) -> Path:
        return TASKS_ROOT / benchmark_id

    def _test_data_path(self, benchmark_id: str) -> Path:
        """test_data.json is mandatory (as in run_task.sh): the agent gets it
        for self-decontamination and the judges use it for the n-gram check."""
        path = self._task_src(benchmark_id) / "test_data.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. It is gitignored; generate it with\n"
                f"  python src/judges/test_data_download/download_test_data.py --tasks {benchmark_id}\n"
                f"(needs the `datasets` package; gpqamain additionally needs MY_HF_TOKEN "
                f"for the gated dataset)."
            )
        return path

    # ------------------------------------------------------------- task.toml

    def generate_task_toml(self, task_dir: Path, benchmark_info: BenchmarkInfo) -> None:
        """Generate task.toml: template + agent timeout + benchmark-required API keys."""
        content = (TEMPLATE_DIR / "task.toml").read_text()

        agent_timeout = self.num_hours * 3600
        content = content.replace(
            "timeout_sec = 36000.0",
            f"timeout_sec = {float(agent_timeout)}",
        )

        # API-key allowlist (mirrors run_task.sh): the agent sandbox receives
        # only what harbor's agent injects for its own provider plus the keys
        # the benchmark's grading declares in info.json (e.g. OPENAI_API_KEY
        # for the LLM-judged benchmarks). Nothing else reaches the agent.
        # `[environment.env]` is harbor's per-task sandbox env (resolved from
        # the host at runtime); it applies to the agent sandbox only — the
        # separate verifier has its own `[verifier.env]`. (There is no
        # `[agent.env]` in harbor's task schema; unknown tables are ignored.)
        if benchmark_info.required_api_keys:
            content += (
                "\n# Provider keys this benchmark's own grading (evaluate.py) needs, from\n"
                "# src/eval/tasks/<id>/info.json `required_api_keys`. The agent prompt\n"
                "# (rule 10) restricts them to running the evaluation.\n"
                "[environment.env]\n"
            )
            for key in benchmark_info.required_api_keys:
                content += f'{key} = "${{{key}}}"\n'

        (task_dir / "task.toml").write_text(content)

    # ---------------------------------------------------------- instruction

    def generate_instruction(
        self,
        task_dir: Path,
        model_info: ModelInfo,
        benchmark_id: str,
    ) -> None:
        """Render instruction.md by running PostTrainBench's own get_prompt.py.

        get_prompt.py resolves benchmark.txt / info.json / test_data.json
        relative to the repo root and applies the same placeholders the condor
        pipeline uses, so the Harbor instruction is identical to the condor
        prompt for the same (model, benchmark, hours, agent).
        """
        cmd = [
            sys.executable, str(GET_PROMPT),
            "--agent", self.agent_name,
            "--model-to-train", model_info.model_id,
            "--benchmark-id", benchmark_id,
            "--num-hours", str(self.num_hours),
            "--num-gpus", "1",
        ]
        env = dict(os.environ)
        env.setdefault("POST_TRAIN_BENCH_PROMPT", "prompt")
        result = subprocess.run(
            cmd, cwd=self.posttrainbench_root, env=env,
            capture_output=True, text=True, check=True,
        )
        (task_dir / "instruction.md").write_text(result.stdout)

    # ---------------------------------------------------------------- timer

    def generate_timer_sh(self, env_dir: Path) -> None:
        """timer.sh: remaining time from /timer_start, written by the task.toml
        healthcheck right before the agent launches (absolute path, so the
        agent's `cd`s don't matter)."""
        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}
START_FILE="/timer_start"

if [ ! -f "$START_FILE" ]; then
    echo "Timer not initialized (healthcheck has not run yet)."
    exit 1
fi

START_DATE=$(cat "$START_FILE")
DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
        timer_path = env_dir / "timer.sh"
        timer_path.write_text(timer_script)
        timer_path.chmod(0o755)

    # ---------------------------------------------------------- environment

    def generate_environment(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
    ) -> None:
        """environment/: the agent image build context."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(TEMPLATE_DIR / "environment" / "Dockerfile", env_dir / "Dockerfile")
        dockerignore_src = TEMPLATE_DIR / "environment" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy(dockerignore_src, env_dir / ".dockerignore")

        self._copy_build_context_support(env_dir)

        # Eval files land in /home/agent/workspace (the agent iterates on them).
        self._copy_eval_files(env_dir, benchmark_id, model_info, benchmark_info)

        self.generate_timer_sh(env_dir)

        # ptb_collect.sh — post-agent hook (weights -> shared volume, code
        # snapshot -> /logs/artifacts). Installed by the Dockerfile at
        # /usr/local/bin and stripped from the workspace; agent image only.
        collect_dst = env_dir / "ptb_collect.sh"
        shutil.copy(TEMPLATE_DIR / "environment" / "ptb_collect.sh", collect_dst)
        collect_dst.chmod(0o755)

        # Self-decontamination tooling (mirrors run_task.sh): the judges'
        # n-gram checker and the benchmark test set, installed by the
        # Dockerfile at /home/agent/ — i.e. `../` from the workspace, the same
        # relative layout the prompt's "Decontamination Tool" section
        # describes (`../contamination_check.py`, `../test_data.json`).
        shutil.copy(JUDGE_TOOLS_DIR / "contamination_check.py", env_dir / "contamination_check.py")
        shutil.copy(self._test_data_path(benchmark_id), env_dir / "test_data.json")

    def _copy_build_context_support(self, target_dir: Path) -> None:
        """entrypoint.sh + system_monitor.sh + requirements-direct.txt, needed
        by both Dockerfiles (environment/ and tests/)."""
        entrypoint_dst = target_dir / "entrypoint.sh"
        shutil.copy(TEMPLATE_DIR / "environment" / "entrypoint.sh", entrypoint_dst)
        entrypoint_dst.chmod(0o755)

        monitor_dst = target_dir / "system_monitor.sh"
        shutil.copy(TEMPLATE_DIR / "environment" / "system_monitor.sh", monitor_dst)
        monitor_dst.chmod(0o755)

        reqs_src = self.posttrainbench_root / "containers" / "requirements-direct.txt"
        if not reqs_src.exists():
            raise FileNotFoundError(
                f"requirements-direct.txt not found at {reqs_src}; "
                f"the Dockerfile expects it in the build context."
            )
        shutil.copy(reqs_src, target_dir / "requirements-direct.txt")

    def _copy_eval_files(
        self,
        target_dir: Path,
        benchmark_id: str,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
        *,
        for_tests: bool = False,
    ) -> None:
        """The evaluation pipeline files (what run_task.sh copies into task/):
        evaluate.py, templates/, evaluation_code/ and task_context/*, plus
        metadata.json. With for_tests=True also the verifier-only files."""
        task_src = self._task_src(benchmark_id)

        eval_src = task_src / "evaluate.py"
        if not eval_src.exists():
            raise FileNotFoundError(f"evaluate.py not found: {eval_src}")
        shutil.copy(eval_src, target_dir / "evaluate.py")

        templates_src = self.posttrainbench_root / "src" / "eval" / "templates"
        shutil.copytree(templates_src, target_dir / "templates", dirs_exist_ok=True)

        eval_code_src = task_src / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, target_dir / "evaluation_code", dirs_exist_ok=True)

        task_context_src = task_src / "task_context"
        if task_context_src.is_dir():
            for item in task_context_src.iterdir():
                dst = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dst)

        if for_tests:
            # v1.0 judge prompt builder; replaced by src/judges in the
            # verifier rework. Verifier-only: the agent must not see it.
            judge_src = TEMPLATE_DIR / "environment" / "contamination_judge.py"
            if judge_src.exists():
                shutil.copy(judge_src, target_dir / "contamination_judge.py")

        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "required_api_keys": benchmark_info.required_api_keys,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
            "agent_name": self.agent_name,
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ---------------------------------------------------------------- tests

    def generate_tests(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
    ) -> None:
        """tests/: the verifier image build context (harbor separate-verifier
        mode builds it into a container the agent never touches). Must
        self-contain test.sh and everything it reads, under /tests/."""
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(TEMPLATE_DIR / "tests" / "Dockerfile", tests_dir / "Dockerfile")

        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(TEMPLATE_DIR / "tests" / "test.sh", test_sh_dst)
        test_sh_dst.chmod(0o755)

        self._copy_build_context_support(tests_dir)
        self._copy_eval_files(tests_dir, benchmark_id, model_info, benchmark_info, for_tests=True)

    # ------------------------------------------------------------- driver

    def generate_task(self, benchmark_id: str, model_key: str) -> Path:
        """Generate a complete Harbor task for a benchmark + model combination."""
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = MODELS[model_key]

        # Fail early, before writing anything.
        self._test_data_path(benchmark_id)

        task_id = f"posttrainbench-{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        # Start from a clean directory so files dropped by earlier template
        # versions cannot linger in the build contexts.
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)

        print(f"Generating task: {task_id}")

        self.generate_task_toml(task_dir, benchmark_info)
        self.generate_instruction(task_dir, model_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id, model_info, benchmark_info)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        """Generate tasks for all benchmark + model combinations."""
        return [
            self.generate_task(benchmark_id, model_key)
            for benchmark_id in BENCHMARKS
            for model_key in MODELS
        ]


def list_available_tasks() -> list[str]:
    """List all available task combinations."""
    return [
        f"posttrainbench-{benchmark_id}-{MODELS[model_key].short_name}"
        for benchmark_id in BENCHMARKS
        for model_key in MODELS
    ]

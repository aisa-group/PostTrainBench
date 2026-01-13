import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ADAPTER_NAME = "POSTTRAINBENCH"
TEMPLATE_DIR = Path(__file__).parent / "template"

# PostTrainBench source directory (relative to repo root)
POSTTRAINBENCH_ROOT = Path(__file__).parent.parent.parent

# Claude-specific instruction clause (from original get_prompt.py)
CLAUDE_CLAUSE = """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""


@dataclass
class BenchmarkInfo:
    task_id: str           # e.g., "gsm8k"
    benchmark_name: str    # e.g., "GSM8K (Grade School Math 8K)"
    setup_note: str = ""   # Additional setup instructions


@dataclass
class ModelInfo:
    model_id: str          # HuggingFace model ID, e.g., "Qwen/Qwen3-1.7B-Base"
    short_name: str        # Short name for task IDs, e.g., "qwen3-1.7b"


BENCHMARKS = {
    "gsm8k": BenchmarkInfo(
        task_id="gsm8k",
        benchmark_name="GSM8K (Grade School Math 8K)",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai. "
                   "Also if you run into issues with the evaluate.py script, this is likely "
                   "due to memory constraints on the GPU. In this case please decrease "
                   "--max-connections or --max-tokens.\n"
    ),
    "humaneval": BenchmarkInfo(
        task_id="humaneval",
        benchmark_name="HumanEval",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "aime2025": BenchmarkInfo(
        task_id="aime2025",
        benchmark_name="AIME 2025",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "gpqamain": BenchmarkInfo(
        task_id="gpqamain",
        benchmark_name="GPQA",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "bfcl": BenchmarkInfo(
        task_id="bfcl",
        benchmark_name="Berkeley Function Calling Leaderboard",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
}

MODELS = {
    "qwen3-1.7b": ModelInfo(
        model_id="Qwen/Qwen3-1.7B-Base",
        short_name="qwen3-1.7b"
    ),
    "qwen3-4b": ModelInfo(
        model_id="Qwen/Qwen3-4B-Base",
        short_name="qwen3-4b"
    ),
    "smollm3-3b": ModelInfo(
        model_id="HuggingFaceTB/SmolLM3-3B-Base",
        short_name="smollm3-3b"
    ),
    "gemma3-4b": ModelInfo(
        model_id="google/gemma-3-4b-pt",
        short_name="gemma3-4b"
    ),
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: int = 10,
        include_claude_clause: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory where Harbor tasks will be generated.
            num_hours: Number of hours for the training task (default: 10).
            include_claude_clause: Whether to include the Claude non-interactive clause.
        """
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.posttrainbench_root = POSTTRAINBENCH_ROOT

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        """Read the human-readable benchmark name from benchmark.txt."""
        bench_file = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        # Fallback to the dataclass info
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def generate_task_toml(self, task_dir: Path) -> None:
        """Generate task.toml for the Harbor task."""
        # Copy template and adjust timeout based on num_hours
        template_path = TEMPLATE_DIR / "task.toml"
        target_path = task_dir / "task.toml"

        content = template_path.read_text()

        # Adjust agent timeout based on num_hours
        agent_timeout = self.num_hours * 3600  # Convert hours to seconds
        content = content.replace(
            "timeout_sec = 36000.0",
            f"timeout_sec = {float(agent_timeout)}"
        )

        target_path.write_text(content)

    def generate_instruction(
        self,
        task_dir: Path,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
    ) -> None:
        """Generate instruction.md for the Harbor task."""
        template_path = TEMPLATE_DIR / "instruction.md"
        target_path = task_dir / "instruction.md"

        content = template_path.read_text()

        # Fill in placeholders
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        content = content.replace("{num_hours}", str(self.num_hours))
        content = content.replace("{setup_other}", benchmark_info.setup_note)

        if self.include_claude_clause:
            content += CLAUDE_CLAUSE

        target_path.write_text(content)

    def generate_timer_sh(self, env_dir: Path) -> None:
        """Generate timer.sh script that tracks remaining time.

        This matches the original PostTrainBench create_timer.sh behavior.
        The timer uses the current timestamp as the creation time.
        """
        creation_date = int(time.time())

        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}
CREATION_DATE={creation_date}

DEADLINE=$((CREATION_DATE + NUM_HOURS * 3600))
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

    def generate_environment(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the environment directory with Dockerfile and task files."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        # Copy Dockerfile template
        shutil.copy(
            TEMPLATE_DIR / "environment" / "Dockerfile",
            env_dir / "Dockerfile"
        )

        # Copy evaluate.py from the benchmark
        eval_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluate.py"
        if eval_src.exists():
            shutil.copy(eval_src, env_dir / "evaluate.py")
        else:
            raise FileNotFoundError(f"evaluate.py not found: {eval_src}")

        # Copy templates directory
        templates_src = self.posttrainbench_root / "src" / "eval" / "templates"
        templates_dst = env_dir / "templates"
        if templates_src.exists():
            shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"templates directory not found: {templates_src}")

        # Copy contamination judge script
        judge_src = TEMPLATE_DIR / "environment" / "contamination_judge.py"
        if judge_src.exists():
            shutil.copy(judge_src, env_dir / "contamination_judge.py")

        # Generate timer.sh (matches original PostTrainBench behavior)
        self.generate_timer_sh(env_dir)

        # Generate metadata.json for verifier (used by contamination judge)
        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
        }
        metadata_path = env_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

    def generate_tests(self, task_dir: Path) -> None:
        """Generate the tests directory with verification script."""
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Copy test.sh
        test_sh_src = TEMPLATE_DIR / "tests" / "test.sh"
        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(test_sh_src, test_sh_dst)
        test_sh_dst.chmod(0o755)

    def generate_task(
        self,
        benchmark_id: str,
        model_key: str,
    ) -> Path:
        """
        Generate a complete Harbor task for a benchmark + model combination.

        Args:
            benchmark_id: The benchmark ID (e.g., "gsm8k").
            model_key: The model key (e.g., "qwen3-1.7b").

        Returns:
            Path to the generated task directory.
        """
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = MODELS[model_key]

        # Try to get actual benchmark name from file
        try:
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )
        except FileNotFoundError:
            pass  # Use default from dataclass

        # Create task directory
        task_id = f"posttrainbench-{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        # Generate all components
        self.generate_task_toml(task_dir)
        self.generate_instruction(task_dir, model_info, benchmark_info)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        """Generate tasks for all benchmark + model combinations."""
        tasks = []
        for benchmark_id in BENCHMARKS:
            for model_key in MODELS:
                task_dir = self.generate_task(benchmark_id, model_key)
                tasks.append(task_dir)
        return tasks


def list_available_tasks() -> list[str]:
    """List all available task combinations."""
    tasks = []
    for benchmark_id in BENCHMARKS:
        for model_key in MODELS:
            task_id = f"posttrainbench-{benchmark_id}-{MODELS[model_key].short_name}"
            tasks.append(task_id)
    return tasks

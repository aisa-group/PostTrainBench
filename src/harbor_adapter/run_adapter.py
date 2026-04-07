"""
Generate Harbor-compatible tasks for running PostTrainBench evaluations.

Usage:
    # Generate a single task (gsm8k + qwen3-1.7b)
    python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./tasks

    # Generate all tasks
    python run_adapter.py --all --output ./tasks

    # List available benchmarks and models
    python run_adapter.py --list

After generating tasks, run them with Harbor:
    harbor run --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b --agent claude-code --model anthropic/claude-sonnet-4 --env modal
"""

import argparse
from pathlib import Path

from adapter import (
    PostTrainBenchAdapter,
    BENCHMARKS,
    MODELS,
    list_available_tasks,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks for PostTrainBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to generate task for",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODELS.keys()),
        help="Base model to generate task for",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./harbor_tasks"),
        help="Output directory for generated tasks (default: ./harbor_tasks)",
    )
    parser.add_argument(
        "--num-hours",
        type=int,
        default=10,
        help="Number of hours for the training task (default: 10)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate tasks for all benchmark + model combinations",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available benchmarks and models",
    )

    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for bm_id, bm_info in BENCHMARKS.items():
            print(f"  {bm_id}: {bm_info.benchmark_name}")
        print("\nAvailable models:")
        for model_key, model_info in MODELS.items():
            print(f"  {model_key}: {model_info.model_id}")
        print("\nAvailable task combinations:")
        for task_id in list_available_tasks():
            print(f"  {task_id}")
        return

    adapter = PostTrainBenchAdapter(
        output_dir=args.output,
        num_hours=args.num_hours,
    )

    if args.all:
        print(f"Generating all tasks to {args.output}/...")
        tasks = adapter.generate_all_tasks()
        print(f"\nGenerated {len(tasks)} tasks.")
        print("\nTo run a task with Harbor:")
        print(f"  harbor run --path {tasks[0]} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")
        return

    if not args.benchmark or not args.model:
        parser.error("Either --all or both --benchmark and --model are required")

    task_dir = adapter.generate_task(args.benchmark, args.model)
    print(f"\nTo run this task with Harbor:")
    print(f"  harbor run --path {task_dir} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")


if __name__ == "__main__":
    main()

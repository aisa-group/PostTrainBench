#!/usr/bin/env python3
"""
Check for missing runs across agents.

For each agent, checks if runs for each (model, benchmark) combination are present.
Also identifies runs that exist but don't have CUDA available.
"""
import os
import argparse
from pathlib import Path

# Expected benchmarks (from constants.py)
EXPECTED_BENCHMARKS = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "healthbench",
    "humaneval",
]

# Expected models (base models only)
EXPECTED_MODELS = [
    "Qwen3-1.7B-Base",
    "Qwen3-4B-Base",
    "SmolLM3-3B-Base",
    "gemma-3-4b-pt",
]


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def parse_run_dir(dir_name: str):
    """
    Parse a run directory name into (benchmark, model, run_id).
    Format: {benchmark}_{sep}_{model}_{run_id}
    """
    try:
        parts = dir_name.split("_")
        if len(parts) < 4:
            return None
        benchmark = parts[0]
        model = parts[2]
        run_id = int(parts[3])
        return benchmark, model, run_id
    except (ValueError, IndexError):
        return None


def check_cuda_available(run_path: Path) -> bool:
    """
    Check if CUDA was available for this run.
    Returns False if task/cuda_not_available exists.
    """
    cuda_not_available = run_path / "task" / "cuda_not_available"
    return not cuda_not_available.exists()


def check_agent(agent_path: Path, agent_name: str, benchmarks: list, models: list):
    """
    Check an agent directory for missing runs and CUDA issues.
    Returns (missing_runs, no_cuda_runs, present_runs).
    """
    # Track which (benchmark, model) combinations exist
    # key: (benchmark, model) -> list of (run_id, path, has_cuda)
    runs_found = {}

    for entry in agent_path.iterdir():
        if not entry.is_dir():
            continue

        parsed = parse_run_dir(entry.name)
        if parsed is None:
            continue

        benchmark, model, run_id = parsed
        key = (benchmark, model)

        has_cuda = check_cuda_available(entry)

        if key not in runs_found:
            runs_found[key] = []
        runs_found[key].append({
            "run_id": run_id,
            "path": entry,
            "has_cuda": has_cuda,
        })

    # Find missing combinations
    missing_runs = []
    no_cuda_runs = []
    present_runs = []

    for benchmark in benchmarks:
        for model in models:
            key = (benchmark, model)
            if key not in runs_found:
                missing_runs.append(key)
            else:
                # Get the latest run
                latest = max(runs_found[key], key=lambda x: x["run_id"])
                if not latest["has_cuda"]:
                    no_cuda_runs.append((key, latest["path"]))
                else:
                    present_runs.append((key, latest["path"]))

    return missing_runs, no_cuda_runs, present_runs


def main():
    parser = argparse.ArgumentParser(
        description="Check for missing runs across agents."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Specific agents to check (default: all agents in results dir)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=EXPECTED_BENCHMARKS,
        help="Benchmarks to check for",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=EXPECTED_MODELS,
        help="Models to check for",
    )
    args = parser.parse_args()

    results_dir = Path(get_results_dir())

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Get list of agents to check
    if args.agents:
        agents = args.agents
    else:
        agents = [
            d.name for d in results_dir.iterdir()
            if d.is_dir() and d.name != "baseline"
        ]

    for agent_name in sorted(agents):
        agent_path = results_dir / agent_name
        if not agent_path.exists():
            print(f"[{agent_name}] Directory not found!")
            continue

        missing, no_cuda, present = check_agent(
            agent_path, agent_name, args.benchmarks, args.models
        )

        if not missing and not no_cuda:
            continue

        print(f"[{agent_name}]")

        if missing:
            print("  Missing:")
            for i, (benchmark, model) in enumerate(sorted(missing), 1):
                print(f"    {i}. {benchmark} x {model}")

        if no_cuda:
            print("  No CUDA:")
            for i, ((benchmark, model), path) in enumerate(sorted(no_cuda), 1):
                print(f"    {i}. {benchmark} x {model}")
                print(f"       {path}")


if __name__ == "__main__":
    main()

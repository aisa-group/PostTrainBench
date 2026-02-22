#!/usr/bin/env python3
"""
Compute total costs for Claude models in PostTrainBench results.

Goes through the results folder, finds all Claude method directories,
extracts cost from solve_out.txt files, and computes totals per method.
If cost is not available (trace cut), uses the average of other runs.
"""
import os
import re
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def get_results_dir() -> str:
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def extract_cost_from_file(solve_out_path: Path) -> Optional[float]:
    """
    Extract total_cost_usd from solve_out.txt file.

    Looks for a JSON line with "type":"result" and extracts total_cost_usd.
    Returns None if cost cannot be extracted (trace cut, file missing, etc.).
    """
    if not solve_out_path.exists():
        return None

    try:
        content = solve_out_path.read_text()
        # Find the result line
        match = re.search(r'\{"type":"result"[^\n]+', content)
        if not match:
            return None

        result_line = match.group(0)
        data = json.loads(result_line)
        cost = data.get("total_cost_usd")
        if cost is not None:
            return float(cost)
        return None
    except Exception:
        return None


def is_claude_method(method_name: str) -> bool:
    """Check if method is a Claude-based method."""
    return method_name.startswith("claude_")


def parse_run_dir_name(run_dir_name: str) -> tuple[str, str, int]:
    """
    Parse a run directory name into (benchmark, model, job_id).

    Run directory names follow the pattern: {benchmark}_{model}_{job_id}
    where model can contain underscores (e.g., "Qwen_Qwen3-1.7B-Base").

    The job_id is always the last underscore-separated component.

    Returns (benchmark, model, job_id).
    Raises ValueError if parsing fails.
    """
    # Job ID is the last component after underscore
    last_underscore = run_dir_name.rfind("_")
    if last_underscore == -1:
        raise ValueError(f"Invalid run directory name (no underscore): {run_dir_name}")

    prefix = run_dir_name[:last_underscore]
    job_id_str = run_dir_name[last_underscore + 1:]

    try:
        job_id = int(job_id_str)
    except ValueError:
        raise ValueError(
            f"Invalid run directory name (job_id not an integer): {run_dir_name}"
        )

    # Now parse benchmark and model from prefix
    # benchmark is the first component, model is the rest
    first_underscore = prefix.find("_")
    if first_underscore == -1:
        raise ValueError(f"Invalid run directory name (no benchmark): {run_dir_name}")

    benchmark = prefix[:first_underscore]
    model = prefix[first_underscore + 1:]

    if not benchmark or not model:
        raise ValueError(f"Invalid run directory name (empty benchmark/model): {run_dir_name}")

    return benchmark, model, job_id


def filter_latest_runs(run_dirs: list[Path]) -> list[Path]:
    """
    Filter run directories to keep only the latest run for each benchmark/model combination.

    When multiple runs exist for the same benchmark/model (differing only in job_id),
    keeps only the one with the highest job_id.

    Returns filtered list of run directories.
    """
    # Group runs by (benchmark, model)
    runs_by_key: dict[tuple[str, str], list[tuple[int, Path]]] = {}

    for run_dir in run_dirs:
        try:
            benchmark, model, job_id = parse_run_dir_name(run_dir.name)
            key = (benchmark, model)

            if key not in runs_by_key:
                runs_by_key[key] = []
            runs_by_key[key].append((job_id, run_dir))
        except ValueError as e:
            print(f"Warning: Skipping invalid run directory: {run_dir.name} ({e})")
            continue

    # Keep only the latest (highest job_id) for each key
    latest_runs = []
    for key, runs in runs_by_key.items():
        # Sort by job_id descending and take the first (highest)
        runs.sort(key=lambda x: x[0], reverse=True)
        latest_runs.append(runs[0][1])

    return latest_runs


def main():
    parser = argparse.ArgumentParser(
        description="Compute total costs for Claude models per method."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. Defaults to POST_TRAIN_BENCH_RESULTS_DIR env var or 'results'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to claude_costs.csv in results dir.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about each run.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(get_results_dir())

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {results_dir}")

    # Find all Claude method directories (exclude testing)
    method_dirs = []
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if is_claude_method(entry.name) and "testing" not in entry.name:
            method_dirs.append(entry)

    method_dirs.sort(key=lambda p: p.name)

    if not method_dirs:
        raise RuntimeError(f"No Claude method directories found in {results_dir}")

    # Collect costs per method
    results = {}  # {method_name: {"costs": [...], "missing_count": int, "run_count": int}}

    for method_dir in method_dirs:
        method_name = method_dir.name
        costs = []
        missing_costs = []

        # Find all run directories
        all_run_dirs = [d for d in method_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

        if not all_run_dirs:
            print(f"Warning: No run directories found in method {method_name}")
            continue

        # Filter to keep only the latest run for each benchmark/model combination
        run_dirs = filter_latest_runs(all_run_dirs)
        duplicates_removed = len(all_run_dirs) - len(run_dirs)

        if args.verbose and duplicates_removed > 0:
            print(f"  {method_name}: Filtered out {duplicates_removed} duplicate runs (keeping latest)")

        for run_dir in run_dirs:
            solve_out = run_dir / "solve_out.txt"
            cost = extract_cost_from_file(solve_out)

            if cost is not None:
                costs.append(cost)
                if args.verbose:
                    print(f"  {run_dir.name}: ${cost:.4f}")
            else:
                missing_costs.append(run_dir.name)
                if args.verbose:
                    print(f"  {run_dir.name}: MISSING (will use average)")

        results[method_name] = {
            "costs": costs,
            "missing_count": len(missing_costs),
            "run_count": len(run_dirs),
            "missing_runs": missing_costs,
        }

    # Compute totals and write results
    output_path = Path(args.output) if args.output else results_dir / "claude_costs.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "total_cost_usd", "run_count", "valid_cost_count", "missing_cost_count", "avg_cost_usd"])

        for method_name in sorted(results.keys()):
            data = results[method_name]
            costs = data["costs"]
            missing_count = data["missing_count"]
            run_count = data["run_count"]

            if not costs:
                print(f"Warning: No valid costs found for method {method_name}, skipping")
                continue

            avg_cost = sum(costs) / len(costs)
            # Total = sum of valid costs + (missing_count * avg_cost)
            total_cost = sum(costs) + (missing_count * avg_cost)

            writer.writerow([
                method_name,
                f"{total_cost:.4f}",
                run_count,
                len(costs),
                missing_count,
                f"{avg_cost:.4f}",
            ])

            notes = []
            if missing_count > 0:
                notes.append(f"{missing_count} missing costs filled with avg ${avg_cost:.4f}")
            notes_str = f" ({', '.join(notes)})" if notes else ""

            print(
                f"{method_name}: {run_count} runs, total ${total_cost:.2f}, "
                f"avg ${avg_cost:.4f}{notes_str}"
            )

    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()

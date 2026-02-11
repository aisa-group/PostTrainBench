#!/usr/bin/env python3
"""
Extract token usage from codex runs and compute average per method.

Goes through the results folder, finds all codex method directories,
extracts token counts from solve_out.txt files, and computes averages.
"""
import os
import re
import sys
import argparse
import csv
from pathlib import Path


def get_results_dir() -> str:
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def extract_tokens_from_file(solve_out_path: Path) -> int | None:
    """
    Extract token count from solve_out.txt file.

    Expects the pattern:
    tokens used
    <number with commas>

    Returns the token count as int, or None if pattern not found.
    """
    content = solve_out_path.read_text()

    # Find "tokens used" followed by a number on the next line
    match = re.search(r"tokens used\n([\d,]+)", content)
    if not match:
        return None

    token_str = match.group(1)
    # Remove commas and convert to int
    tokens = int(token_str.replace(",", ""))
    return tokens


def is_codex_method(method_name: str) -> bool:
    """Check if method is a codex-based method."""
    return method_name.startswith(("codex_", "codexhigh_", "codexlow_"))


def parse_run_dir_name(run_dir_name: str) -> tuple[str, int]:
    """
    Parse a run directory name into (run_key, job_id).

    Run directory names follow the pattern: {benchmark}_{model}_{job_id}
    where model can contain underscores (e.g., "Qwen_Qwen3-1.7B-Base").

    The job_id is always the last underscore-separated component.

    Returns (run_key, job_id) where run_key is "{benchmark}_{model}".
    Raises ValueError if parsing fails.
    """
    # Job ID is the last component after underscore
    last_underscore = run_dir_name.rfind("_")
    if last_underscore == -1:
        raise ValueError(f"Invalid run directory name (no underscore): {run_dir_name}")

    run_key = run_dir_name[:last_underscore]
    job_id_str = run_dir_name[last_underscore + 1 :]

    try:
        job_id = int(job_id_str)
    except ValueError:
        raise ValueError(
            f"Invalid run directory name (job_id not an integer): {run_dir_name}"
        )

    if not run_key:
        raise ValueError(f"Invalid run directory name (empty run_key): {run_dir_name}")

    return run_key, job_id


def filter_latest_runs(run_dirs: list[Path]) -> list[Path]:
    """
    Filter run directories to keep only the latest run for each benchmark/model combination.

    When multiple runs exist for the same benchmark/model (differing only in job_id),
    keeps only the one with the highest job_id.

    Returns filtered list of run directories.
    """
    # Group runs by run_key
    runs_by_key: dict[str, list[tuple[int, Path]]] = {}

    for run_dir in run_dirs:
        run_key, job_id = parse_run_dir_name(run_dir.name)

        if run_key not in runs_by_key:
            runs_by_key[run_key] = []
        runs_by_key[run_key].append((job_id, run_dir))

    # Keep only the latest (highest job_id) for each key
    latest_runs = []
    for run_key, runs in runs_by_key.items():
        # Sort by job_id descending and take the first (highest)
        runs.sort(key=lambda x: x[0], reverse=True)
        latest_runs.append(runs[0][1])

    return latest_runs


def main():
    parser = argparse.ArgumentParser(
        description="Extract token usage from codex runs and compute averages per method."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. Defaults to POST_TRAIN_BENCH_RESULTS_DIR env var or 'results'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to token_usage.csv in results dir.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about each run.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip runs that don't have token usage info (incomplete runs). "
        "By default, the script fails on such runs.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(get_results_dir())

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {results_dir}")

    # Find all method directories (codex only)
    method_dirs = []
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if is_codex_method(entry.name):
            method_dirs.append(entry)

    method_dirs.sort(key=lambda p: p.name)

    if not method_dirs:
        raise RuntimeError(f"No codex method directories found in {results_dir}")

    # Collect token usage per method
    results = {}  # {method_name: [list of token counts]}

    for method_dir in method_dirs:
        method_name = method_dir.name
        token_counts = []
        skipped_count = 0

        # Find all run directories
        all_run_dirs = [d for d in method_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

        if not all_run_dirs:
            raise RuntimeError(f"No run directories found in method {method_name}")

        # Filter to keep only the latest run for each benchmark/model combination
        run_dirs = filter_latest_runs(all_run_dirs)
        duplicates_removed = len(all_run_dirs) - len(run_dirs)

        if args.verbose and duplicates_removed > 0:
            print(f"  Filtered out {duplicates_removed} duplicate runs (keeping latest)")

        for run_dir in run_dirs:
            solve_out = run_dir / "solve_out.txt"

            if not solve_out.exists():
                raise FileNotFoundError(f"solve_out.txt not found: {solve_out}")

            tokens = extract_tokens_from_file(solve_out)

            if tokens is None:
                if args.skip_incomplete:
                    skipped_count += 1
                    if args.verbose:
                        print(f"  {run_dir.name}: SKIPPED (incomplete)")
                    continue
                else:
                    raise ValueError(
                        f"Could not find 'tokens used' pattern in {solve_out} "
                        "(incomplete run?). Use --skip-incomplete to skip such runs."
                    )

            token_counts.append(tokens)

            if args.verbose:
                print(f"  {run_dir.name}: {tokens:,} tokens")

        if not token_counts:
            raise RuntimeError(
                f"No valid runs found in method {method_name} "
                f"(all {skipped_count} runs were incomplete)"
            )

        results[method_name] = token_counts

        avg_tokens = sum(token_counts) / len(token_counts)
        total_tokens = sum(token_counts)

        notes = []
        if duplicates_removed > 0:
            notes.append(f"{duplicates_removed} duplicates removed")
        if skipped_count > 0:
            notes.append(f"{skipped_count} skipped")
        notes_str = f" ({', '.join(notes)})" if notes else ""

        print(
            f"{method_name}: {len(token_counts)} runs{notes_str}, "
            f"avg {avg_tokens:,.0f} tokens, total {total_tokens:,} tokens"
        )

    # Write output CSV
    output_path = Path(args.output) if args.output else results_dir / "token_usage.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "avg_tokens"])

        for method_name in sorted(results.keys()):
            token_counts = results[method_name]
            avg_tokens = sum(token_counts) / len(token_counts)
            writer.writerow([method_name, f"{avg_tokens:.0f}"])

    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()

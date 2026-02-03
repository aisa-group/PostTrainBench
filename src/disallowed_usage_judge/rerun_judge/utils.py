#!/usr/bin/env python3
"""Shared utilities for rerun judge scripts."""

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_results_dir() -> Path:
    """Get the results directory from environment variable."""
    results_dir = os.environ.get('POST_TRAIN_BENCH_RESULTS_DIR')
    if not results_dir:
        raise RuntimeError("POST_TRAIN_BENCH_RESULTS_DIR is not set")
    return Path(results_dir)


def get_result_dirs(
    method_pattern: str = None,
    benchmark_pattern: str = None,
    skip_existing: bool = False,
    limit: int = 0,
) -> list[Path]:
    """
    Get result directories matching the given criteria.

    Args:
        method_pattern: Filter by method name (substring match)
        benchmark_pattern: Filter by benchmark name (substring match)
        skip_existing: Skip directories that already have rerun judgement files
        limit: Maximum number of directories to return (0 = no limit)

    Returns:
        List of Path objects for matching result directories
    """
    results_root = get_results_dir()
    result_dirs = []

    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        if method_name.startswith('.') or method_name == 'baseline':
            continue

        if method_pattern and method_pattern.lower() not in method_name.lower():
            continue

        for result_dir in sorted(method_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            if not (result_dir / 'task').is_dir():
                continue

            if benchmark_pattern and benchmark_pattern.lower() not in result_dir.name.lower():
                continue

            if skip_existing and (result_dir / 'contamination_judgement_rerun.txt').exists():
                continue

            result_dirs.append(result_dir)

    if limit > 0:
        result_dirs = result_dirs[:limit]

    return result_dirs


def parse_result_dir(result_dir: Path) -> dict:
    """
    Parse a result directory to extract benchmark, model, and method info.

    Args:
        result_dir: Path to the result directory

    Returns:
        Dictionary with benchmark, model, model_hf (HuggingFace format), method, cluster_id
    """
    dirname = result_dir.name
    method = result_dir.parent.name

    # Format: {benchmark}_{provider}_{model}_{cluster_id}
    parts = dirname.rsplit('_', 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid result directory name: {dirname}")

    cluster_id = parts[1]
    rest = parts[0]

    benchmark_end = rest.find('_')
    if benchmark_end == -1:
        raise ValueError(f"Invalid result directory name: {dirname}")

    benchmark = rest[:benchmark_end]
    model_part = rest[benchmark_end + 1:]
    model_hf = model_part.replace('_', '/', 1)

    return {
        'benchmark': benchmark,
        'model': model_part,
        'model_hf': model_hf,
        'method': method,
        'cluster_id': cluster_id,
    }


def get_benchmark_name(benchmark: str) -> str:
    """Get human-readable benchmark name from info.json if available."""
    import json
    repo_root = get_repo_root()
    benchmark_file = repo_root / 'src' / 'eval' / 'tasks' / benchmark / 'info.json'

    if benchmark_file.exists():
        with open(benchmark_file, encoding="utf-8") as f:
            info = json.load(f)
        return info["benchmark"]
    return benchmark


def get_trace_file(result_dir: Path) -> tuple[Path, str] | tuple[None, None]:
    """
    Get the trace file to use for a result directory.

    Returns:
        Tuple of (path to trace file, source name) or (None, None) if no trace found
    """
    parsed = result_dir / 'solve_parsed.txt'
    if parsed.exists():
        return parsed, 'solve_parsed.txt'

    raw = result_dir / 'solve_out.txt'
    if raw.exists():
        return raw, 'solve_out.txt'

    return None, None


def read_judgement(filepath: Path) -> str | None:
    """Read a judgement file and return its content, or None if missing."""
    if not filepath.exists():
        return None
    return filepath.read_text().strip()

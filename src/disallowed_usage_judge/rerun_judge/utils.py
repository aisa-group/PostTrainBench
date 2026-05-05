#!/usr/bin/env python3
"""Shared utilities for rerun judge scripts."""

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def get_results_dir() -> Path:
    results_dir = os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR")
    if not results_dir:
        raise RuntimeError("POST_TRAIN_BENCH_RESULTS_DIR is not set")
    return Path(results_dir)


def get_result_dirs(
    method_pattern: str = None,
    benchmark_pattern: str = None,
    skip_existing: bool = False,
    only_missing_judgement: bool = False,
    limit: int = 0,
    latest_only: bool = False,
) -> list[Path]:
    """
    Walk POST_TRAIN_BENCH_RESULTS_DIR and return matching result directories.

    Args:
        method_pattern: substring filter on method (parent dir name)
        benchmark_pattern: substring filter on result dir name
        skip_existing: skip dirs that already have contamination_judgement_rerun.txt
        only_missing_judgement: only include dirs missing contamination_judgement.txt
            and/or disallowed_model_judgement.txt (i.e. the judge previously failed)
        limit: cap result count (0 = no limit)
        latest_only: keep only the highest-cluster_id run per (method, model, benchmark)
    """
    results_root = get_results_dir()
    result_dirs = []

    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        if method_name.startswith(".") or method_name in ("baseline", "baseline_zeroshot"):
            continue

        if method_pattern and method_pattern.lower() not in method_name.lower():
            continue

        for result_dir in sorted(method_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            if not (result_dir / "task").is_dir():
                continue

            if benchmark_pattern and benchmark_pattern.lower() not in result_dir.name.lower():
                continue

            if skip_existing and (result_dir / "contamination_judgement_rerun.txt").exists():
                continue

            if only_missing_judgement:
                has_contam = (result_dir / "contamination_judgement.txt").exists()
                has_disallowed = (result_dir / "disallowed_model_judgement.txt").exists()
                if has_contam and has_disallowed:
                    continue

            result_dirs.append(result_dir)

    if latest_only:
        result_dirs = _filter_latest_only(result_dirs)

    if limit > 0:
        result_dirs = result_dirs[:limit]

    return result_dirs


def _filter_latest_only(result_dirs: list[Path]) -> list[Path]:
    best_by_key: dict[tuple[str, str, str], tuple[int, Path]] = {}

    for result_dir in result_dirs:
        try:
            parsed = parse_result_dir(result_dir)
        except ValueError:
            continue
        key = (parsed["method"], parsed["model"], parsed["benchmark"])
        cluster_id = int(parsed["cluster_id"])

        if key not in best_by_key or cluster_id > best_by_key[key][0]:
            best_by_key[key] = (cluster_id, result_dir)

    return sorted(path for _, path in best_by_key.values())


def parse_result_dir(result_dir: Path) -> dict:
    """
    Parse a result dir name into its components.
    Format: {benchmark}_{provider}_{model}_{cluster_id}
    """
    dirname = result_dir.name
    method = result_dir.parent.name

    parts = dirname.rsplit("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid result directory name: {dirname}")

    cluster_id = parts[1]
    rest = parts[0]

    benchmark_end = rest.find("_")
    if benchmark_end == -1:
        raise ValueError(f"Invalid result directory name: {dirname}")

    benchmark = rest[:benchmark_end]
    model_part = rest[benchmark_end + 1:]
    model_hf = model_part.replace("_", "/", 1)

    return {
        "benchmark": benchmark,
        "model": model_part,
        "model_hf": model_hf,
        "method": method,
        "cluster_id": cluster_id,
    }


def read_judgement(filepath: Path) -> str | None:
    if not filepath.exists():
        return None
    return filepath.read_text().strip()

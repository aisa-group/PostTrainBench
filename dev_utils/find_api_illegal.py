#!/usr/bin/env python3
"""
Find files that contain synthetic data generation code using external APIs
(OpenAI, Anthropic, Google/Gemini).

Searches both solve files and task scripts, excluding evaluation/library code.
"""
import os

import argparse
import re
from pathlib import Path


# Patterns that indicate API usage for data generation
PATTERNS = {
    "openai": [
        r"completions\.create",
        r"from openai import",
        r"import openai",
        r"OpenAI\(\)",
    ],
    "anthropic": [
        r"messages\.create",
        r"from anthropic import",
        r"import anthropic",
        r"Anthropic\(\)",
    ],
    "google_gemini": [
        r"generate_content",
        r"from google\.generativeai",
        r"import google\.generativeai",
        r"genai\.GenerativeModel",
    ],
}

# Directories/files to exclude (evaluation code, libraries, etc.)
EXCLUDE_PATTERNS = [
    # r"/evaluation_code/",
    r"evaluation_code/utils/completion.py",
    r"task/evaluation_code/grader.py",
    r"/site-packages/",
    r"/patched_lib/",
    r"/env/lib/",
    r"/inspect_ai/",
    r"/inspect_evals/",
    # r"grader\.py$",
    r"evaluate\.py$",
    # r"completion\.py$",  # evaluation utility
]


def should_exclude(filepath: str) -> bool:
    """Check if file should be excluded based on path patterns."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, filepath):
            return True
    return False


def search_file(filepath: Path, patterns: dict[str, list[str]]) -> dict[str, list[str]]:
    """Search a file for API patterns. Returns dict of provider -> matched patterns."""
    try:
        content = filepath.read_text(errors="ignore")
    except Exception:
        return {}

    matches = {}
    for provider, provider_patterns in patterns.items():
        matched = []
        for pattern in provider_patterns:
            if re.search(pattern, content):
                matched.append(pattern)
        if matched:
            matches[provider] = matched

    return matches


def find_api_usage(results_dir: Path, search_solve: bool = True, search_task: bool = True, folders_only: bool = False) -> None:
    """Find files with external API usage for synthetic data generation."""

    findings = []

    if search_solve:
        # Search solve files
        for solve_file in results_dir.rglob("solve*.txt"):
            if should_exclude(str(solve_file)):
                continue
            matches = search_file(solve_file, PATTERNS)
            if matches:
                findings.append((solve_file, matches))

    if search_task:
        # Search task Python scripts
        for py_file in results_dir.rglob("task/**/*.py"):
            if should_exclude(str(py_file)):
                continue
            matches = search_file(py_file, PATTERNS)
            if matches:
                findings.append((py_file, matches))

    # Extract unique folder names
    folders = set()
    for filepath, matches in findings:
        rel_path = filepath.relative_to(results_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            run_name = f"{parts[0]}/{parts[1]}"
        else:
            run_name = parts[0] if parts else "unknown"
        folders.add(run_name)

    if folders_only:
        print(f"{len(folders)} folders")
        for folder in sorted(folders):
            print(folder)
        return

    # Print results
    if not findings:
        print("No files found with external API usage.")
        return

    print(f"Found {len(findings)} files with external API usage:\n")

    # Group by run directory
    by_run = {}
    for filepath, matches in findings:
        # Extract run name (first two path components after results_dir)
        rel_path = filepath.relative_to(results_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            run_name = f"{parts[0]}/{parts[1]}"
        else:
            run_name = parts[0] if parts else "unknown"

        if run_name not in by_run:
            by_run[run_name] = []
        by_run[run_name].append((filepath, matches))

    for run_name in sorted(by_run.keys()):
        print(f"=== {run_name} ===")
        for filepath, matches in by_run[run_name]:
            rel_path = filepath.relative_to(results_dir)
            providers = ", ".join(matches.keys())
            print(f"  [{providers}] {rel_path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Find files with external API usage for synthetic data generation"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to results directory",
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="Skip searching solve files",
    )
    parser.add_argument(
        "--no-task",
        action="store_true",
        help="Skip searching task scripts",
    )
    parser.add_argument(
        "--task-only",
        action="store_true",
        help="Only search task scripts (exclude solve files)",
    )
    parser.add_argument(
        "--solve-only",
        action="store_true",
        help="Only search solve files (exclude task scripts)",
    )
    parser.add_argument(
        "--folders-only",
        action="store_true",
        help="Only print deduplicated experiment folder paths (minimal output)",
    )

    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR")

    args.results_dir = Path(args.results_dir)

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    search_solve = not args.no_solve and not args.task_only
    search_task = not args.no_task and not args.solve_only

    find_api_usage(args.results_dir, search_solve=search_solve, search_task=search_task, folders_only=args.folders_only)
    return 0


if __name__ == "__main__":
    exit(main())

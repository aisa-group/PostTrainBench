#!/usr/bin/env python3
"""List runs where the agent hit an API usage/spending limit."""

import argparse
import os

# Patterns that indicate the agent hit a usage or spending limit.
# These are checked case-insensitively against solve_out.txt.
LIMIT_PATTERNS = [
    "You've hit your limit",         # Claude Code Pro subscription limit
    "spending_limit",                 # Anthropic/OpenAI spending limit
    "billing_hard_limit",            # OpenAI billing hard limit
    "insufficient_quota",            # OpenAI quota exceeded
    "budget_exceeded",               # General budget error
    "plan does not yet include",     # Z.AI subscription plan restriction
]


def check_solve_out_for_limits(solve_out_path: str):
    """
    Check if solve_out.txt contains any limit patterns.
    Returns a list of matched patterns, or empty list if none found.
    """
    if not os.path.exists(solve_out_path):
        return []

    with open(solve_out_path, "r") as f:
        content = f.read()

    content_lower = content.lower()
    matched_patterns = []
    for pattern in LIMIT_PATTERNS:
        if pattern.lower() in content_lower:
            matched_patterns.append(pattern)

    return matched_patterns


def get_latest_runs(method_path: str):
    """
    Scans a method directory and returns a list of paths corresponding
    to the latest run_id for every (benchmark, model) pair.
    """
    latest_runs = {}

    for entry in os.listdir(method_path):
        entry_path = os.path.join(method_path, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            parts = entry.split("_")
            if len(parts) < 4:
                continue
            benchmark = parts[0]
            model = parts[2]
            run_id = int(parts[3])
        except (ValueError, IndexError):
            continue
        key = (benchmark, model)

        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {
                "run_id": run_id,
                "path": entry_path,
            }

    return [info["path"] for info in latest_runs.values()]


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def main():
    parser = argparse.ArgumentParser(
        description="List runs where the agent hit an API usage/spending limit"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Results directory (default: POST_TRAIN_BENCH_RESULTS_DIR or 'results')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all runs, not just the latest per (benchmark, model)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else get_results_dir()

    errors_by_pattern = {pattern: [] for pattern in LIMIT_PATTERNS}
    all_errors_list = []

    for method_name in sorted(os.listdir(results_dir)):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue

        if args.all:
            run_paths = [
                os.path.join(method_path, d)
                for d in os.listdir(method_path)
                if os.path.isdir(os.path.join(method_path, d))
            ]
        else:
            run_paths = get_latest_runs(method_path)

        for run_path in run_paths:
            solve_out_path = os.path.join(run_path, "solve_out.txt")
            matched_patterns = check_solve_out_for_limits(solve_out_path)

            if matched_patterns:
                all_errors_list.append((run_path, matched_patterns))
                for pattern in matched_patterns:
                    errors_by_pattern[pattern].append(run_path)

    print(f"=== LIMIT HIT RUNS ({len(all_errors_list)} runs affected) ===\n")

    for pattern in LIMIT_PATTERNS:
        affected_runs = errors_by_pattern[pattern]
        if not affected_runs:
            continue
        print(f"Pattern: \"{pattern}\"")
        print(f"  Affected runs: {len(affected_runs)}")
        for path in sorted(affected_runs):
            print(f"    - {path}")
        print()

    print("-" * 40)
    print(f"\n=== ALL AFFECTED RUNS ({len(all_errors_list)}) ===")
    if all_errors_list:
        for path, patterns in sorted(all_errors_list):
            print(f"{path}")
            for p in patterns:
                print(f"  -> {p}")
    else:
        print("None")


if __name__ == "__main__":
    main()

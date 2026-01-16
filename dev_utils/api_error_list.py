#!/usr/bin/env python3
import os

# List of error patterns to search for in solve.out
ERROR_PATTERNS = [
    "API Error: exception TypeError: fetch failed sending request",
    "Loop detected, stopping execution",
]


def check_solve_out_for_errors(solve_out_path: str):
    """
    Check if solve.out contains any of the error patterns.
    Returns a list of matched patterns, or empty list if none found.
    """
    if not os.path.exists(solve_out_path):
        if "baseline" not in solve_out_path:
            print(solve_out_path)
        return []

    with open(solve_out_path, "r") as f:
        content = f.read()

    matched_patterns = []
    for pattern in ERROR_PATTERNS:
        if pattern in content:
            matched_patterns.append(pattern)

    return matched_patterns


def get_latest_runs(method_path: str):
    """
    Scans a method directory and returns a list of paths corresponding
    to the latest run_id for every (benchmark, model) pair.
    """
    # key: (benchmark, model) -> value: {"run_id": int, "path": str}
    latest_runs = {}

    for entry in os.listdir(method_path):
        entry_path = os.path.join(method_path, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            benchmark, _, model, run_id_str = entry.split("_")
            run_id = int(run_id_str)
        except ValueError:
            # Skip entries that don't match the expected format
            continue
        key = (benchmark, model)

        # keep only highest run_id per (benchmark, model)
        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {
                "run_id": run_id,
                "path": entry_path,
            }

    return [info["path"] for info in latest_runs.values()]


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')


def main():
    results_dir = get_results_dir()

    # Dict to collect runs by error pattern
    errors_by_pattern = {pattern: [] for pattern in ERROR_PATTERNS}
    all_errors_list = []

    # 1. Iterate over all methods and collect paths
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue

        # Get only the latest runs for this method to avoid reporting old overwritten runs
        run_paths = get_latest_runs(method_path)

        for run_path in run_paths:
            # Check solve.out for error patterns
            solve_out_path = os.path.join(run_path, "solve_out.txt")
            matched_patterns = check_solve_out_for_errors(solve_out_path)

            if matched_patterns:
                all_errors_list.append((run_path, matched_patterns))
                for pattern in matched_patterns:
                    errors_by_pattern[pattern].append(run_path)

    # 2. Output summary
    print(f"=== API ERRORS DETECTED ({len(all_errors_list)} runs affected) ===\n")

    # Show breakdown by pattern
    for pattern in ERROR_PATTERNS:
        affected_runs = errors_by_pattern[pattern]
        print(f"Pattern: \"{pattern}\"")
        print(f"  Affected runs: {len(affected_runs)}")
        if affected_runs:
            for path in sorted(affected_runs):
                print(f"    - {path}")
        print()

    # Show combined list
    print("-" * 40)
    print(f"\n=== ALL AFFECTED RUNS ({len(all_errors_list)}) ===")
    if all_errors_list:
        for path, patterns in sorted(all_errors_list):
            print(f"{path}")
            for p in patterns:
                print(f"  -> {p[:60]}...")
    else:
        print("None")


if __name__ == "__main__":
    main()

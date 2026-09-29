#!/usr/bin/env python3
import os
import sys

# The effective contamination verdict (manual override > majority of three
# judge runs > rerun > inline) is resolved in scripts/utils.py, the same way
# collect.py scores it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from utils import resolve_judgement


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
        benchmark, _, model, run_id_str = entry.split("_")
        run_id = int(run_id_str)
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

    contaminated_list = []
    disallowed_list = []

    ignored_runs = os.environ.get("POST_TRAIN_BENCH_CONTAMINATION_CORRECT", '').split(":")

    # 1. Iterate over all methods and collect paths
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue

        # Get only the latest runs for this method to avoid reporting old overwritten runs
        run_paths = get_latest_runs(method_path)

        for run_path in run_paths:
            resolved = resolve_judgement(run_path)
            if resolved is None:
                continue
            _, judgement = resolved

            if judgement["contamination"] and run_path not in ignored_runs:
                contaminated_list.append(run_path)

            if judgement["disallowed_model"] and run_path not in ignored_runs:
                disallowed_list.append(run_path)

    # 2. Output the lists
    print(f"=== CONTAMINATION DETECTED ({len(contaminated_list)}) ===")
    if contaminated_list:
        for path in sorted(contaminated_list):
            print(path)
    else:
        print("None")

    print("\n" + "-"*40 + "\n")

    print(f"=== DISALLOWED MODELS DETECTED ({len(disallowed_list)}) ===")
    if disallowed_list:
        for path in sorted(disallowed_list):
            print(path)
    else:
        print("None")


if __name__ == "__main__":
    main()

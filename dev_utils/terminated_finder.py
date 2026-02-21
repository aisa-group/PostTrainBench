#!/usr/bin/env python3
"""List runs where error.log indicates the job was Terminated (timeout) or Killed (OOM)."""

import argparse
import os
import re
from pathlib import Path


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def classify_error(error_log_path: Path) -> str | None:
    """Classify the error in error.log. Returns 'terminated', 'killed', or None."""
    if not error_log_path.exists():
        return None
    try:
        content = error_log_path.read_text()
        if content.startswith("Terminated"):
            return "terminated"
        if re.search(r"\bKilled\b", content):
            return "killed"
        return None
    except Exception:
        return None


def get_latest_runs(method_path: Path):
    """
    Scans a method directory and returns a dict mapping (benchmark, model)
    to the path of the latest run_id.
    """
    latest_runs = {}

    for entry in method_path.iterdir():
        if not entry.is_dir():
            continue
        try:
            parts = entry.name.split("_")
            if len(parts) < 4:
                continue
            benchmark = parts[0]
            model = parts[2]
            run_id = int(parts[3])
            key = (benchmark, model)

            if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
                latest_runs[key] = {
                    "run_id": run_id,
                    "path": entry,
                }
        except (ValueError, IndexError):
            continue

    return {k: v["path"] for k, v in latest_runs.items()}


def collect_runs(results_dir: Path, check_all: bool):
    """Collect and classify runs into terminated and killed categories."""
    terminated_runs = []
    killed_runs = []

    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue

        if check_all:
            run_dirs = [d for d in method_dir.iterdir() if d.is_dir()]
        else:
            latest = get_latest_runs(method_dir)
            run_dirs = list(latest.values())

        for run_dir in run_dirs:
            error_log = run_dir / "error.log"
            classification = classify_error(error_log)
            if classification == "terminated":
                terminated_runs.append(run_dir)
            elif classification == "killed":
                killed_runs.append(run_dir)

    terminated_runs.sort(key=lambda p: str(p))
    killed_runs.sort(key=lambda p: str(p))
    return terminated_runs, killed_runs


def main():
    parser = argparse.ArgumentParser(
        description="List runs where error.log indicates Terminated (timeout) or Killed (OOM)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all affected runs, not just the latest per (benchmark, model)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the affected run directories (use with caution!)",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Results directory (default: POST_TRAIN_BENCH_RESULTS_DIR or 'results')",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(get_results_dir())
    terminated_runs, killed_runs = collect_runs(results_dir, args.all)

    print(f"=== TERMINATED RUNS - timeout/SIGTERM ({len(terminated_runs)}) ===")
    if terminated_runs:
        for path in terminated_runs:
            print(path)
    else:
        print("None")

    print()

    print(f"=== KILLED RUNS - OOM/SIGKILL ({len(killed_runs)}) ===")
    if killed_runs:
        for path in killed_runs:
            print(path)
    else:
        print("None")

    # Optionally delete
    all_affected = terminated_runs + killed_runs
    if args.delete and all_affected:
        print(f"\nDeleting {len(all_affected)} affected runs...")
        import shutil
        for path in all_affected:
            print(f"  Removing: {path}")
            shutil.rmtree(path)
        print("Done.")


if __name__ == "__main__":
    main()

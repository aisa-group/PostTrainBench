#!/usr/bin/env python3
"""List runs where error.log contains 'Terminated' (prematurely cut off runs)."""

import argparse
import os
from pathlib import Path


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def check_terminated(error_log_path: Path) -> bool:
    """Return True if error.log contains 'Terminated'."""
    if not error_log_path.exists():
        return False
    try:
        content = error_log_path.read_text()
        return content.startswith("Terminated")
    except Exception:
        return False


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


def main():
    parser = argparse.ArgumentParser(
        description="List runs where error.log contains 'Terminated'"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all terminated runs, not just the latest per (benchmark, model)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the terminated run directories (use with caution!)",
    )
    args = parser.parse_args()

    results_dir = Path(get_results_dir())
    terminated_runs = []

    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue

        if args.all:
            # Check all runs
            for run_dir in method_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                error_log = run_dir / "error.log"
                if check_terminated(error_log):
                    terminated_runs.append(run_dir)
        else:
            # Only check latest runs per (benchmark, model)
            latest = get_latest_runs(method_dir)
            for run_dir in latest.values():
                error_log = run_dir / "error.log"
                if check_terminated(error_log):
                    terminated_runs.append(run_dir)

    # Sort and display
    terminated_runs.sort(key=lambda p: str(p))

    print(f"=== TERMINATED RUNS ({len(terminated_runs)}) ===")
    if terminated_runs:
        for path in terminated_runs:
            print(path)
    else:
        print("None")

    # Optionally delete
    if args.delete and terminated_runs:
        print(f"\nDeleting {len(terminated_runs)} terminated runs...")
        import shutil
        for path in terminated_runs:
            print(f"  Removing: {path}")
            shutil.rmtree(path)
        print("Done.")


if __name__ == "__main__":
    main()
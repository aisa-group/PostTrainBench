#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

def list_runs_no_metrics(show_all: bool = False):
    results_dir = os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')

    base_path = Path(results_dir)

    # Iterate through subdirs
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        # Iterate through subsubdirs
        for subsubdir in subdir.iterdir():
            if not subsubdir.is_dir():
                continue

            # Check if metrics.json and final_model/ are both missing
            metrics_file = subsubdir / "metrics.json"
            final_model_dir = subsubdir / "final_model"
            if not metrics_file.exists():
                if show_all or (final_model_dir.exists() and any(final_model_dir.iterdir())):
                    print(subsubdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List runs without metrics.json")
    parser.add_argument("--all", action="store_true", dest="show_all",
                        help="Show all runs without metrics.json, regardless of final_model/ status")
    args = parser.parse_args()
    list_runs_no_metrics(show_all=args.show_all)

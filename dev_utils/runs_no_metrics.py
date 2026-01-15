#!/usr/bin/env python3
import os
from pathlib import Path

def list_runs_no_metrics():
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
            if not metrics_file.exists() and final_model_dir.exists() and any(final_model_dir.iterdir()):
                print(subsubdir)

if __name__ == "__main__":
    list_runs_no_metrics()

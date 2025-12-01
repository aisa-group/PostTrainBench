#!/usr/bin/env python3
import argparse
import os
import json
import csv

OUTPUT_PREFIX = "aggregated_"        # e.g. "agg_" if you want names like agg_method.csv

def load_metrics(metrics_path: str, method_name: str = None):
    """
    Return a string suitable for the CSV.
    - Always returns the metrics data if metrics.json exists and is valid
    - Only shows error messages if metrics.json doesn't exist or is invalid:
      For non-baseline methods:
        - "not avl." if time_taken.txt doesn't exist in the folder
        - "not stored" if time_taken.txt exists but final_model subfolder doesn't
        - "ERR" for other errors
      For baseline method:
        - Just return "ERR" for any errors (old behavior)
    """
    # First, try to load metrics.json - if it works, return the data immediately
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            
            acc = data.get("accuracy")
            if acc is not None:
                return str(acc)
        except Exception:
            pass  # Fall through to error handling below
    
    # Only reach here if metrics.json doesn't exist or is invalid
    # For baseline, just return "ERR"
    if method_name == "baseline":
        return "ERR"
    
    # For non-baseline methods, provide more specific error messages
    run_dir = os.path.dirname(metrics_path)
    
    # Check for time_taken.txt
    time_taken_path = os.path.join(run_dir, "time_taken.txt")
    if not os.path.exists(time_taken_path):
        return "not avl."
    
    # Check for final_model subdirectory
    final_model_path = os.path.join(run_dir, "final_model")
    if not os.path.isdir(final_model_path):
        return "not stored"
    
    # All checks passed but still no valid metrics.json
    return "ERR"

def process_method(method_path: str, method_name: str, min_run_id=None, max_run_id=None):
    """
    For a single method dir (results/method_name), collect the newest run per
    (benchmark, model), then write a CSV.
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
            key = (benchmark, model)
        except ValueError as e:
            print(entry)
            raise ValueError(f"{entry}, {method_path}")
        
        if max_run_id is not None and run_id >= max_run_id:
            continue

        if min_run_id is not None and run_id < min_run_id:
            continue
        
        # keep only highest run_id per (benchmark, model)
        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {
                "run_id": run_id,
                "path": entry_path,
            }
    
    if not latest_runs:
        # nothing to do for this method
        return
    
    # Collect distinct benchmarks and models
    benchmarks = sorted({b for (b, m) in latest_runs.keys()})
    models = sorted({m for (b, m) in latest_runs.keys()})
    
    # Prepare CSV path (next to results/ or inside results/)
    csv_path = os.path.join(get_results_dir(), f"{OUTPUT_PREFIX}{method_name}.csv")
    
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # header
        writer.writerow(["model"] + benchmarks)
        
        # rows
        for model in models:
            row = [model]
            for bench in benchmarks:
                cell = ""
                key = (bench, model)
                if key in latest_runs:
                    run_dir = latest_runs[key]["path"]
                    metrics_path = os.path.join(run_dir, "metrics.json")
                    cell = load_metrics(metrics_path, method_name)
                row.append(cell)
            writer.writerow(row)

    print(f"Written: {csv_path}")

def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate latest benchmark runs into CSV files.")
    parser.add_argument(
        "--min-run-id",
        type=int,
        default=None,
        help="Inclusive lower bound for run ids to consider.",
    )
    parser.add_argument(
        "--max-run-id",
        type=int,
        default=None,
        help="Exclusive upper bound for run ids to consider.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = get_results_dir()
    
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue
        
        # treat every subdirectory of results/ as a "method"
        process_method(method_path, method_name, min_run_id=args.min_run_id, max_run_id=args.max_run_id)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import csv
import re

OUTPUT_PREFIX = "aggregated_time_"  # e.g. aggregated_time_method.csv
OVERVIEW_FILENAME = "aggregated_time_overview.csv"
BUDGET_SECONDS = 10 * 3600  # 10 hours


def parse_time_hms(time_str: str) -> int | None:
    """
    Parse a time string in H:M:S format and return total seconds.
    Returns None if parsing fails.
    """
    match = re.match(r'^(\d+):(\d{1,2}):(\d{1,2})$', time_str.strip())
    if not match:
        return None
    hours, minutes, seconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def format_time_hms(total_seconds: int) -> str:
    """Convert total seconds to H:M:S format."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def load_time_taken(run_dir: str) -> tuple[str, int | None]:
    """
    Return the time taken as (display_string, total_seconds).
    - Returns (H:M:S string, seconds) if valid
    - Returns ("ERR", None) if time_taken.txt doesn't exist or is invalid
    """
    time_taken_path = os.path.join(run_dir, "time_taken.txt")
    
    if not os.path.exists(time_taken_path):
        return "ERR", None
    
    try:
        with open(time_taken_path, "r") as f:
            time_str = f.read().strip()
        total_seconds = parse_time_hms(time_str)
        if total_seconds is None:
            return "ERR", None
        return format_time_hms(total_seconds), total_seconds
    except Exception:
        return "ERR", None


def process_method(method_path: str, method_name: str, min_run_id=None, max_run_id=None) -> dict:
    """
    For a single method dir (results/method_name), collect the newest run per
    (benchmark, model), then write a CSV.
    
    Returns a dict with timing statistics for the overview.
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
        return {}
    
    benchmarks = sorted({b for (b, m) in latest_runs.keys()})
    models = sorted({m for (b, m) in latest_runs.keys()})
    
    csv_path = os.path.join(get_results_dir(), f"{OUTPUT_PREFIX}{method_name}.csv")
    
    # Collect timing stats for overview
    total_seconds = 0
    valid_count = 0
    
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model"] + benchmarks)
        
        for model in models:
            row = [model]
            for bench in benchmarks:
                cell = ""
                key = (bench, model)
                if key in latest_runs:
                    run_dir = latest_runs[key]["path"]
                    cell, seconds = load_time_taken(run_dir)
                    if seconds is not None:
                        total_seconds += seconds
                        valid_count += 1
                row.append(cell)
            writer.writerow(row)
    print(f"Written: {csv_path}")
    
    return {
        "total_seconds": total_seconds,
        "valid_count": valid_count,
    }


def write_overview(method_stats: dict[str, dict]):
    """Write an overview CSV with average times per method."""
    csv_path = os.path.join(get_results_dir(), OVERVIEW_FILENAME)
    
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["method", "average_time", "percentage"])
        
        for method_name in sorted(method_stats.keys()):
            stats = method_stats[method_name]
            total_secs = stats["total_seconds"]
            valid = stats["valid_count"]
            
            if valid > 0:
                avg_secs = total_secs // valid
                avg_str = format_time_hms(avg_secs)
                pct = (avg_secs / BUDGET_SECONDS) * 100
                pct_str = f"{pct:.1f}%"
            else:
                avg_str = "N/A"
                pct_str = "N/A"
            
            writer.writerow([method_name, avg_str, pct_str])
    
    print(f"Written: {csv_path}")


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate latest benchmark run times into CSV files.")
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
    
    method_stats = {}
    
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue
        
        stats = process_method(method_path, method_name, min_run_id=args.min_run_id, max_run_id=args.max_run_id)
        if stats:
            method_stats[method_name] = stats
    
    if method_stats:
        write_overview(method_stats)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Collect results from raw run directories into per-method CSVs.

For each method directory in the results dir, does a single pass:
  1. Finds the latest run per (benchmark, model)
  2. Reads metrics.json, contamination files, and time_taken.txt
  3. Applies baseline fallback for contaminated or errored cells
  4. Writes final_{method}.csv, contamination_{method}.csv

Also writes a time_overview.csv summarising average time per method.

Usage:
    python collect.py
    python collect.py --data-dir /path/to/results --output-dir /path/to/output
    python collect.py --min-run-id 100 --max-run-id 200
"""
import argparse
import csv
import os

from utils import (
    get_results_dir,
    get_baseline_fallback_data,
    walk_latest_runs,
    load_metrics,
    load_contamination,
    load_disallowed_model,
    combine_contamination_results,
    load_time_taken,
    is_number,
    format_time_hms,
    BUDGET_SECONDS,
)

# Directories to skip (baselines are hardcoded in baselines.json)
SKIP_METHODS = {"baseline", "baseline_zeroshot"}


def collect_method(
    method_path: str,
    method_name: str,
    baseline_data: dict[str, dict[str, str]],
    output_dir: str,
    min_run_id: int | None = None,
    max_run_id: int | None = None,
) -> dict | None:
    """
    Collect results for one method directory.

    Writes:
      - final_{method_name}.csv      (scores with baseline fallback)
      - contamination_{method_name}.csv (contamination flags)

    Returns time stats dict {"total_seconds": int, "valid_count": int}
    or None if no runs found.
    """
    latest_runs = walk_latest_runs(method_path, min_run_id, max_run_id)
    if not latest_runs:
        return None

    benchmarks = sorted({b for b, m in latest_runs})
    models = sorted({m for b, m in latest_runs})

    # Collect metrics, contamination, and time in one pass
    metrics_grid = {}  # {model: {bench: str}}
    contamination_grid = {}  # {model: {bench: str}}
    time_total_seconds = 0
    time_valid_count = 0

    for model in models:
        metrics_grid[model] = {}
        contamination_grid[model] = {}

        for bench in benchmarks:
            key = (bench, model)
            if key not in latest_runs:
                metrics_grid[model][bench] = ""
                contamination_grid[model][bench] = ""
                continue

            run_dir = latest_runs[key]["path"]

            # Metrics
            metrics_path = os.path.join(run_dir, "metrics.json")
            metrics_grid[model][bench] = load_metrics(metrics_path, method_name)

            # Contamination
            contamination = load_contamination(
                os.path.join(run_dir, "contamination_judgement.txt")
            )
            disallowed = load_disallowed_model(
                os.path.join(run_dir, "disallowed_model_judgement.txt")
            )
            contamination_grid[model][bench] = combine_contamination_results(
                contamination, disallowed
            )

            # Time
            _, seconds = load_time_taken(run_dir)
            if seconds is not None:
                time_total_seconds += seconds
                time_valid_count += 1

    # Write contamination CSV
    contamination_path = os.path.join(
        output_dir, f"contamination_{method_name}.csv"
    )
    with open(contamination_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                row.append(contamination_grid[model][bench])
            writer.writerow(row)
    print(f"Written: {contamination_path}")

    # Apply baseline fallback: replace cell with baseline if
    #   (a) value is not a number, OR
    #   (b) contamination flag is non-empty
    for model in models:
        for bench in benchmarks:
            value = metrics_grid[model][bench]
            contamination_value = contamination_grid[model][bench]

            reasons = []
            if not is_number(value):
                reasons.append(f"non-numeric value ({value!r})")
            if contamination_value.strip():
                reasons.append(
                    f"contamination flag ({contamination_value.strip()!r})"
                )

            if not reasons:
                continue

            if model not in baseline_data or bench not in baseline_data[model]:
                raise KeyError(
                    f"baselines.json missing entry for model={model!r} "
                    f"benchmark={bench!r}; needed as fallback in method "
                    f"{method_name!r} (triggered by {', '.join(reasons)})"
                )
            metrics_grid[model][bench] = baseline_data[model][bench]

    # Write final CSV (scores with baseline fallback applied)
    final_path = os.path.join(output_dir, f"final_{method_name}.csv")
    with open(final_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                row.append(metrics_grid[model].get(bench, ""))
            writer.writerow(row)
    print(f"Written: {final_path}")

    return {
        "total_seconds": time_total_seconds,
        "valid_count": time_valid_count,
    }


def write_time_overview(method_stats: dict[str, dict], output_dir: str):
    """Write time_overview.csv with average time per method."""
    csv_path = os.path.join(output_dir, "time_overview.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect raw results into per-method CSVs."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing method subdirectories with raw run data. "
        "Defaults to POST_TRAIN_BENCH_RESULTS_DIR or 'results'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output CSVs. Defaults to same as --data-dir.",
    )
    parser.add_argument(
        "--min-run-id",
        type=int,
        default=None,
        help="Inclusive lower bound for run IDs to consider.",
    )
    parser.add_argument(
        "--max-run-id",
        type=int,
        default=None,
        help="Exclusive upper bound for run IDs to consider.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir or get_results_dir()
    output_dir = args.output_dir or data_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load baseline data for fallback (hardcoded in baselines.json)
    baseline_data = get_baseline_fallback_data()

    method_stats = {}

    for method_name in sorted(os.listdir(data_dir)):
        method_path = os.path.join(data_dir, method_name)
        if not os.path.isdir(method_path):
            continue

        # Skip baseline directories — their values are hardcoded
        if method_name in SKIP_METHODS:
            continue

        stats = collect_method(
            method_path,
            method_name,
            baseline_data,
            output_dir,
            min_run_id=args.min_run_id,
            max_run_id=args.max_run_id,
        )
        if stats:
            method_stats[method_name] = stats

    if method_stats:
        write_time_overview(method_stats, output_dir)


if __name__ == "__main__":
    main()

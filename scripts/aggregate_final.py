#!/usr/bin/env python3
"""
Final aggregation script that combines method results with baseline fallbacks.

For each method's aggregated CSV, replaces values with baseline when:
1. The value is not a number (e.g., "ERR", "not avl.", etc.), OR
2. The corresponding contamination value is not empty (flagged as "C", "M", "MC", etc.)

Baseline values come from aggregated_baseline.csv.
"""
import os
import csv
import argparse

OUTPUT_PREFIX = "final_"


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def is_number(value: str) -> bool:
    """Check if a string represents a number (int or float)."""
    if not value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_csv_as_dict(csv_path: str) -> tuple[dict, list]:
    """
    Load a CSV file into a dict of dicts: {model: {benchmark: value}}.
    Returns (data_dict, list_of_benchmarks).
    """
    data = {}
    benchmarks = []

    if not os.path.exists(csv_path):
        return data, benchmarks

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return data, benchmarks

        # First column is "model", rest are benchmarks
        benchmarks = header[1:]

        for row in reader:
            if not row:
                continue
            model = row[0]
            data[model] = {}
            for i, bench in enumerate(benchmarks):
                if i + 1 < len(row):
                    data[model][bench] = row[i + 1]
                else:
                    data[model][bench] = ""

    return data, benchmarks


def process_method(method_name: str, baseline_data: dict, results_dir: str):
    """
    Process a single method: load its aggregated and contamination CSVs,
    apply baseline fallbacks where needed, and write the final CSV.
    """
    aggregated_path = os.path.join(results_dir, f"aggregated_{method_name}.csv")
    contamination_path = os.path.join(results_dir, f"contamination_{method_name}.csv")

    # Load method data
    method_data, method_benchmarks = load_csv_as_dict(aggregated_path)
    if not method_data:
        return

    # Load contamination data (may not exist)
    contamination_data, _ = load_csv_as_dict(contamination_path)

    # Get all models from method data
    models = sorted(method_data.keys())

    # Process each cell and apply baseline if needed
    for model in models:
        for bench in method_benchmarks:
            value = method_data[model].get(bench, "")
            contamination_value = contamination_data.get(model, {}).get(bench, "")

            # Check conditions for baseline replacement
            needs_baseline = False

            # Condition 1: value is not a number
            if not is_number(value):
                needs_baseline = True

            # Condition 2: contamination value is not empty
            if contamination_value.strip():
                needs_baseline = True

            if needs_baseline:
                # Get baseline value (may be empty if model/bench not in baseline)
                baseline_value = baseline_data.get(model, {}).get(bench, "")
                method_data[model][bench] = baseline_value

    # Write output
    output_path = os.path.join(results_dir, f"{OUTPUT_PREFIX}{method_name}.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + method_benchmarks)

        for model in models:
            row = [model]
            for bench in method_benchmarks:
                row.append(method_data[model].get(bench, ""))
            writer.writerow(row)

    print(f"Written: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create final aggregated CSVs with baseline fallbacks."
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Specific methods to process. If not provided, processes all methods.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = get_results_dir()

    # Load baseline data
    baseline_path = os.path.join(results_dir, "aggregated_baseline.csv")
    baseline_data, _ = load_csv_as_dict(baseline_path)

    if not baseline_data:
        print(f"Warning: No baseline data found at {baseline_path}")

    # Determine which methods to process
    if args.methods:
        method_names = args.methods
    else:
        # Find all aggregated method files (excluding baseline)
        method_names = []
        for filename in os.listdir(results_dir):
            if not filename.startswith("aggregated_") or not filename.endswith(".csv"):
                continue
            method_name = filename[len("aggregated_") : -len(".csv")]
            # Skip baseline itself
            if method_name != "baseline":
                method_names.append(method_name)

    # Process each method
    for method_name in sorted(method_names):
        process_method(method_name, baseline_data, results_dir)


if __name__ == "__main__":
    main()
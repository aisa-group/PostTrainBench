#!/usr/bin/env python3
"""
Compute final metric for each final_*.csv table using factors from factors.json.

For each benchmark, computes the average value across all models, then multiplies
by the factor. Sums these to produce a single metric per method.
"""
import os
import csv
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORS_PATH = os.path.join(SCRIPT_DIR, "factors.json")

# Expected models in each complete CSV
EXPECTED_MODELS = {
    "Qwen3-1.7B-Base",
    "Qwen3-4B-Base",
    "SmolLM3-3B-Base",
    "gemma-3-4b-pt",
}


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def load_factors(factors_path: str) -> dict:
    """Load factors from JSON file."""
    with open(factors_path, "r") as f:
        return json.load(f)


def load_final_csv(csv_path: str, valid_benchmarks: set) -> tuple[dict, list]:
    """
    Load a final CSV file into a dict: {model: {benchmark: value}}.
    Only loads benchmarks that are in valid_benchmarks.
    Returns (data_dict, list_of_benchmarks).
    """
    data = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        all_benchmarks = header[1:]
        benchmarks = [b for b in all_benchmarks if b in valid_benchmarks]

        for row in reader:
            model = row[0]
            data[model] = {}
            for i, bench in enumerate(all_benchmarks):
                if bench not in valid_benchmarks:
                    continue
                val = row[i + 1]
                if val == "":
                    continue
                data[model][bench] = float(val)

    return data, benchmarks


def compute_metric(data: dict, factors: dict, benchmarks: list) -> float:
    """
    Compute weighted sum where each benchmark value is averaged across models.
    """
    total = 0.0
    num_models = len(data)
    for bench in benchmarks:
        avg_value = sum(data[model][bench] for model in data) / num_models
        factor = factors[bench]
        total += avg_value * factor
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Compute final metrics from final_*.csv files using benchmark factors."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to final_metrics.csv in results dir.",
    )
    args = parser.parse_args()

    results_dir = get_results_dir()
    factors = load_factors(FACTORS_PATH)
    valid_benchmarks = set(factors.keys())

    # Find all final_*.csv files, excluding final_time_* files
    final_files = []
    for filename in os.listdir(results_dir):
        if not filename.startswith("final_"):
            continue
        if not filename.endswith(".csv"):
            continue
        if filename.startswith("final_time_"):
            continue
        # Check if file has all expected models
        csv_path = os.path.join(results_dir, filename)
        try:
            data, _ = load_final_csv(csv_path, valid_benchmarks)
        except Exception:
            print(f"Warning: could not load {csv_path}.")
            raise
        if set(data.keys()) != EXPECTED_MODELS:
            continue
        final_files.append(filename)

    final_files.sort()

    # Compute metrics for each file
    results = {}  # {method_name: metric}
    for filename in final_files:
        method_name = filename[len("final_") : -len(".csv")]
        csv_path = os.path.join(results_dir, filename)
        data, benchmarks = load_final_csv(csv_path, valid_benchmarks)
        metric = compute_metric(data, factors, benchmarks)
        results[method_name] = metric

    # Write output table
    output_path = args.output or os.path.join(results_dir, "final_metrics.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric"])

        for method_name in sorted(results.keys()):
            writer.writerow([method_name, results[method_name]])

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()

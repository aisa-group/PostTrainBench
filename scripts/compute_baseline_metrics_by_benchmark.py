#!/usr/bin/env python3
"""
Compute metrics for baseline models (base and instruct-tuned) using factors from factors_by_benchmark.json.

Reads aggregated_baseline.csv and computes averaged metrics for base and instruct groups.
"""
import os
import csv
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORS_PATH = os.path.join(SCRIPT_DIR, "factors.json")

BASE_MODELS = {"Qwen3-1.7B-Base", "Qwen3-4B-Base", "SmolLM3-3B-Base", "gemma-3-4b-pt"}
INSTRUCT_MODELS = {"Qwen3-1.7B", "Qwen3-4B", "SmolLM3-3B", "gemma-3-4b-it"}


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def load_factors(factors_path: str) -> dict:
    """Load factors from JSON file."""
    with open(factors_path, "r") as f:
        return json.load(f)


def load_baseline_csv(csv_path: str) -> tuple[dict, list]:
    """
    Load baseline CSV file into a dict: {model: {benchmark: value}}.
    Returns (data_dict, list_of_benchmarks).
    """
    data = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        benchmarks = header[1:]

        for row in reader:
            model = row[0]
            data[model] = {}
            for i, bench in enumerate(benchmarks):
                data[model][bench] = float(row[i + 1])

    return data, benchmarks


def compute_metric_by_benchmark(data: dict, factors: dict, benchmarks: list, models: set) -> float:
    """
    Compute weighted sum where each benchmark value is averaged across specified models.
    """
    total = 0.0
    model_list = [m for m in data if m in models]
    num_models = len(model_list)
    for bench in benchmarks:
        avg_value = sum(data[model][bench] for model in model_list) / num_models
        factor = factors[bench]
        total += avg_value * factor
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline metrics using benchmark factors."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to baseline_metrics_by_benchmark.csv in results dir.",
    )
    args = parser.parse_args()

    results_dir = get_results_dir()
    factors = load_factors(FACTORS_PATH)

    csv_path = os.path.join(results_dir, "aggregated_baseline.csv")
    data, benchmarks = load_baseline_csv(csv_path)

    # Compute averaged metrics for each group
    base_metric = compute_metric_by_benchmark(data, factors, benchmarks, BASE_MODELS)
    instruct_metric = compute_metric_by_benchmark(data, factors, benchmarks, INSTRUCT_MODELS)

    # Write output table
    output_path = args.output or os.path.join(results_dir, "baseline_metrics_by_benchmark.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "metric"])
        writer.writerow(["base", base_metric])
        writer.writerow(["instruct", instruct_metric])

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()

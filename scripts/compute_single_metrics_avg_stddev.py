#!/usr/bin/env python3
"""
Compute final metric for each final_*.csv table using factors from factors.json,
then aggregate across multiple runs to produce averages and standard deviations.

For each benchmark, computes the average value across all models, then multiplies
by the factor. Sums these to produce a single metric per method.

Then, for each agent group defined in HARDCODED_AGENT_MAP, computes the average
and standard deviation of the metrics across runs.
"""
import os
import csv
import json
import math

from constants import HARDCODED_AGENT_MAP, HARDCODED_BENCHMARKS

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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


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


def compute_all_metrics(results_dir: str, factors: dict) -> dict[str, float]:
    """
    Compute metrics for all final_*.csv files.
    Returns {method_name: metric}.
    """
    valid_benchmarks = set(factors.keys())
    results = {}

    for filename in os.listdir(results_dir):
        if not filename.startswith("final_"):
            continue
        if not filename.endswith(".csv"):
            continue
        if filename.startswith("final_time_"):
            continue

        csv_path = os.path.join(results_dir, filename)
        data, benchmarks = load_final_csv(csv_path, valid_benchmarks)

        if set(data.keys()) != EXPECTED_MODELS:
            continue

        method_name = filename[len("final_") : -len(".csv")]
        metric = compute_metric(data, factors, benchmarks)
        results[method_name] = metric

    return results


def main():
    results_dir = get_results_dir()
    factors = load_factors(FACTORS_PATH)

    # Compute metrics for all methods
    all_metrics = compute_all_metrics(results_dir, factors)

    # Write individual metrics CSV
    metrics_path = os.path.join(results_dir, "single_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric"])
        for method_name in sorted(all_metrics.keys()):
            writer.writerow([method_name, round(all_metrics[method_name] * 100, 1)])
    print(f"Written: {metrics_path}")

    # Compute aggregated metrics for each agent group
    aggregated_results = {}

    for agent_name, method_names in HARDCODED_AGENT_MAP.items():
        metrics = []
        for method_name in method_names:
            metric = all_metrics[method_name]
            metrics.append(metric)

        aggregated_results[agent_name] = {
            "avg": round(mean(metrics) * 100, 1),
            "std": round(stddev(metrics) * 100, 1),
            "n": len(metrics),
        }

    # Write aggregated metrics CSV
    aggregated_path = os.path.join(results_dir, "single_metrics_aggregated.csv")
    with open(aggregated_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "avg", "std", "n"])
        for agent_name in sorted(aggregated_results.keys()):
            data = aggregated_results[agent_name]
            writer.writerow([agent_name, data["avg"], data["std"], data["n"]])
    print(f"Written: {aggregated_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute metrics for baseline models (base and instruct-tuned) using factors from factors.json.

Reads aggregated_baseline.csv and computes weighted metrics per model.
"""
import os
import csv
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORS_PATH = os.path.join(SCRIPT_DIR, "factors.json")

# Mapping from CSV model names to factors.json model names
MODEL_NAME_MAPPING = {
    # Base/pretrained models
    "Qwen3-1.7B-Base": "Qwen3-1.7B",
    "Qwen3-4B-Base": "Qwen3-4B",
    "SmolLM3-3B-Base": "SmolLM3-3B",
    "gemma-3-4b-pt": "gemma-3-4b",
    # Instruct-tuned models
    "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B",
    "SmolLM3-3B": "SmolLM3-3B",
    "gemma-3-4b-it": "gemma-3-4b",
}

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


def compute_metric(model_data: dict, factors: dict, benchmarks: list) -> float:
    """Compute weighted sum of benchmark values using factors."""
    total = 0.0
    for bench in benchmarks:
        value = model_data[bench]
        factor = factors[bench]
        total += value * factor
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline metrics using per-model factors."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to baseline_metrics.csv in results dir.",
    )
    args = parser.parse_args()

    results_dir = get_results_dir()
    factors = load_factors(FACTORS_PATH)

    csv_path = os.path.join(results_dir, "aggregated_baseline.csv")
    data, benchmarks = load_baseline_csv(csv_path)

    # Compute metrics for each model
    base_results = {}
    instruct_results = {}

    for csv_model in data:
        factors_model = MODEL_NAME_MAPPING[csv_model]
        model_factors = factors[factors_model]
        metric = compute_metric(data[csv_model], model_factors, benchmarks)

        if csv_model in BASE_MODELS:
            base_results[csv_model] = metric
        else:
            instruct_results[csv_model] = metric

    # Write output table
    output_path = args.output or os.path.join(results_dir, "baseline_metrics.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "model", "metric"])

        for model in sorted(base_results.keys()):
            writer.writerow(["base", model, base_results[model]])

        for model in sorted(instruct_results.keys()):
            writer.writerow(["instruct", model, instruct_results[model]])

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()

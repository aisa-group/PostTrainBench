#!/usr/bin/env python3
"""
Aggregate results across multiple runs for each agent, averaging over models.

Takes the outputs of aggregate_final.py (final_*.csv files), averages values
over models for each benchmark, then computes average and standard deviation
across runs for each agent.

Produces two CSV files: one for averages, one for standard deviations.
Rows are benchmarks, columns are agents.
"""
import os
import csv
import math

from constants import HARDCODED_AGENT_MAP, HARDCODED_BENCHMARKS


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def load_csv_as_dict(csv_path: str) -> dict[str, dict[str, str]]:
    """
    Load a CSV file into a dict of dicts: {model: {benchmark: value}}.
    """
    data = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # First column is "model", rest are benchmarks
        benchmarks = header[1:]

        for row in reader:
            model = row[0]
            data[model] = {}
            for i, bench in enumerate(benchmarks):
                data[model][bench] = row[i + 1]

    return data


def compute_model_average(data: dict[str, dict[str, str]], bench: str) -> float:
    """Compute average value over all models for a given benchmark."""
    values = []
    for model in data:
        value = float(data[model][bench])
        values.append(value)
    return mean(values)


def aggregate_agent(method_names: list[str], results_dir: str):
    """
    Aggregate results from multiple method runs for one agent.
    Returns (avg_per_benchmark, std_per_benchmark, avg_per_model_benchmark, std_per_model_benchmark, models).

    avg_per_benchmark[bench] = avg value (averaged over models)
    avg_per_model_benchmark[model][bench] = avg value (per model)
    """
    # For each run, compute model-averaged value per benchmark
    # run_averages[benchmark] = [avg_run1, avg_run2, ...]
    run_averages = {bench: [] for bench in HARDCODED_BENCHMARKS}

    # For each run, also store per-model values
    # run_values_per_model[model][benchmark] = [val_run1, val_run2, ...]
    run_values_per_model = {}
    all_models = None

    for method_name in method_names:
        csv_path = os.path.join(results_dir, f"final_{method_name}.csv")
        data = load_csv_as_dict(csv_path)

        models = sorted(data.keys())
        if all_models is None:
            all_models = models
            for model in models:
                run_values_per_model[model] = {bench: [] for bench in HARDCODED_BENCHMARKS}

        for bench in HARDCODED_BENCHMARKS:
            model_avg = compute_model_average(data, bench)
            run_averages[bench].append(model_avg)

            for model in all_models:
                value = float(data[model][bench])
                run_values_per_model[model][bench].append(value)

    # Compute avg and std across runs for each benchmark (averaged over models)
    avg_per_benchmark = {}
    std_per_benchmark = {}

    for bench in HARDCODED_BENCHMARKS:
        values = run_averages[bench]
        avg_per_benchmark[bench] = round(mean(values) * 100, 1)
        std_per_benchmark[bench] = round(stddev(values) * 100, 1)

    # Compute avg and std across runs for each (model, benchmark) pair
    avg_per_model_benchmark = {}
    std_per_model_benchmark = {}

    for model in all_models:
        avg_per_model_benchmark[model] = {}
        std_per_model_benchmark[model] = {}
        for bench in HARDCODED_BENCHMARKS:
            values = run_values_per_model[model][bench]
            avg_per_model_benchmark[model][bench] = round(mean(values) * 100, 1)
            std_per_model_benchmark[model][bench] = round(stddev(values) * 100, 1)

    return avg_per_benchmark, std_per_benchmark, avg_per_model_benchmark, std_per_model_benchmark, all_models


def main():
    results_dir = get_results_dir()

    # Collect results for all agents
    # all_avg[benchmark][agent] = avg_value
    # all_std[benchmark][agent] = std_value
    all_avg = {bench: {} for bench in HARDCODED_BENCHMARKS}
    all_std = {bench: {} for bench in HARDCODED_BENCHMARKS}

    # Per-model results: all_avg_per_model[model][benchmark][agent] = avg_value
    all_avg_per_model = {}
    all_std_per_model = {}

    agent_names = list(HARDCODED_AGENT_MAP.keys())
    all_models = None

    for agent_name, method_names in HARDCODED_AGENT_MAP.items():
        print(f"Processing agent: {agent_name}")
        avg_per_benchmark, std_per_benchmark, avg_per_model, std_per_model, models = aggregate_agent(method_names, results_dir)

        if all_models is None:
            all_models = models
            for model in models:
                all_avg_per_model[model] = {bench: {} for bench in HARDCODED_BENCHMARKS}
                all_std_per_model[model] = {bench: {} for bench in HARDCODED_BENCHMARKS}

        for bench in HARDCODED_BENCHMARKS:
            all_avg[bench][agent_name] = avg_per_benchmark[bench]
            all_std[bench][agent_name] = std_per_benchmark[bench]

            for model in all_models:
                all_avg_per_model[model][bench][agent_name] = avg_per_model[model][bench]
                all_std_per_model[model][bench][agent_name] = std_per_model[model][bench]

    # Write average CSV (over models)
    avg_path = os.path.join(results_dir, "aggregated_avg_over_models.csv")
    with open(avg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark"] + agent_names)
        for bench in HARDCODED_BENCHMARKS:
            row = [bench] + [all_avg[bench][agent] for agent in agent_names]
            writer.writerow(row)
    print(f"Written: {avg_path}")

    # Write std CSV (over models)
    std_path = os.path.join(results_dir, "aggregated_std_over_models.csv")
    with open(std_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark"] + agent_names)
        for bench in HARDCODED_BENCHMARKS:
            row = [bench] + [all_std[bench][agent] for agent in agent_names]
            writer.writerow(row)
    print(f"Written: {std_path}")

    # Write per-model CSV files
    for model in all_models:
        avg_path = os.path.join(results_dir, f"aggregated_avg_{model}.csv")
        with open(avg_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark"] + agent_names)
            for bench in HARDCODED_BENCHMARKS:
                row = [bench] + [all_avg_per_model[model][bench][agent] for agent in agent_names]
                writer.writerow(row)
        print(f"Written: {avg_path}")

        std_path = os.path.join(results_dir, f"aggregated_std_{model}.csv")
        with open(std_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark"] + agent_names)
            for bench in HARDCODED_BENCHMARKS:
                row = [bench] + [all_std_per_model[model][bench][agent] for agent in agent_names]
                writer.writerow(row)
        print(f"Written: {std_path}")


if __name__ == "__main__":
    main()

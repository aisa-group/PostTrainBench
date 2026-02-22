#!/usr/bin/env python3
"""
Aggregate results across multiple runs for each agent.

Takes the outputs of aggregate_final.py (final_*.csv files) and combines
them into average and standard deviation CSVs for each agent group.
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


def load_csv_as_dict(csv_path: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Load a CSV file into a dict of dicts: {model: {benchmark: value}}.
    Returns (data_dict, list_of_benchmarks).
    """
    data = {}
    benchmarks = []

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

    return data, benchmarks


def aggregate_runs(agent_name: str, method_names: list[str], results_dir: str):
    """
    Aggregate results from multiple method runs into average and std CSV files.
    """
    # Load all method data
    all_data = []
    all_models = None

    for method_name in method_names:
        csv_path = os.path.join(results_dir, f"final_{method_name}.csv")
        data, _ = load_csv_as_dict(csv_path)

        models = sorted(data.keys())
        if all_models is None:
            all_models = models
        else:
            assert all_models == models, (
                f"Model mismatch for {method_name}: "
                f"expected {all_models}, got {models}"
            )

        all_data.append(data)

    # Compute average and std for each (model, benchmark) cell
    avg_data = {}
    std_data = {}

    for model in all_models:
        avg_data[model] = {}
        std_data[model] = {}

        for bench in HARDCODED_BENCHMARKS:
            values = []
            for data in all_data:
                value_str = data[model][bench]
                value = float(value_str)
                values.append(value)

            avg_data[model][bench] = str(round(mean(values) * 100, 1))
            std_data[model][bench] = str(round(stddev(values) * 100, 1))

    # Write average CSV
    avg_path = os.path.join(results_dir, f"aggregated_avg_{agent_name}.csv")
    write_csv(avg_path, all_models, HARDCODED_BENCHMARKS, avg_data)
    print(f"Written: {avg_path}")

    # Write std CSV
    std_path = os.path.join(results_dir, f"aggregated_std_{agent_name}.csv")
    write_csv(std_path, all_models, HARDCODED_BENCHMARKS, std_data)
    print(f"Written: {std_path}")


def write_csv(
    path: str,
    models: list[str],
    benchmarks: list[str],
    data: dict[str, dict[str, str]],
):
    """Write data dict to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + benchmarks)

        for model in models:
            row = [model]
            for bench in benchmarks:
                row.append(data[model][bench])
            writer.writerow(row)


def main():
    results_dir = get_results_dir()

    for agent_name, method_names in HARDCODED_AGENT_MAP.items():
        print(f"Processing agent: {agent_name}")
        aggregate_runs(agent_name, method_names, results_dir)


if __name__ == "__main__":
    main()

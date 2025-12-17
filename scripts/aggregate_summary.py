#!/usr/bin/env python3
"""
Aggregate final_{method}.csv files into a single summary CSV.

For each benchmark, computes the average score across models per method.

Output format:
- Rows: benchmarks
- Columns: baseline_base, baseline_instruct, method1, method2, ...
- Values: average score across models
"""
import os
import csv
import argparse

METHOD_NAME_MAP = {
    "claude_claude-sonnet-4-5": "claude sonnet 4.5",
    "claude_claude-opus-4-5": "claude opus 4.5",
    "codex_gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "codex_gpt-5.2": "gpt-5.2",
    "gemini_models_gemini-3-pro-preview": "gemini-3-pro",
}

# Model groups for baseline columns
BASE_MODELS = ["Qwen3-1.7B-Base", "Qwen3-4B-Base", "SmolLM3-3B-Base", "gemma-3-4b-pt"]
INSTRUCT_MODELS = ["Qwen3-1.7B", "Qwen3-4B", "SmolLM3-3B", "gemma-3-4b-it"]


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


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


def compute_benchmark_average(data: dict, benchmark: str, models: list = None) -> str:
    """
    Compute average score for a benchmark across specified models.
    If models is None, uses all models in data.
    Returns empty string if no valid scores found.
    """
    if models is None:
        models = list(data.keys())

    values = []
    for model in models:
        val_str = data.get(model, {}).get(benchmark, "")
        if val_str:
            try:
                values.append(float(val_str))
            except ValueError:
                pass

    if not values:
        return ""

    return f"{sum(values) / len(values):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate final CSVs into a single summary with model averages per benchmark."
    )
    parser.add_argument(
        "methods",
        nargs="+",
        help="List of methods to include in the aggregation.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV filename. Default: summary.csv in results dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = get_results_dir()

    # Load baseline data
    baseline_path = os.path.join(results_dir, "aggregated_baseline.csv")
    baseline_data, baseline_benchmarks = load_csv_as_dict(baseline_path)

    if not baseline_data:
        print(f"Error: No baseline data found at {baseline_path}")
        return

    # Load all method data
    method_data = {}
    method_benchmarks = {}

    for method in args.methods:
        final_path = os.path.join(results_dir, f"final_{method}.csv")
        data, benchmarks = load_csv_as_dict(final_path)

        if not data:
            print(f"Warning: No data found for method '{method}' at {final_path}")
            continue

        method_data[method] = data
        method_benchmarks[method] = set(benchmarks)

    if not method_data:
        print("Error: No valid method data found.")
        return

    # Find common benchmarks across baseline and all methods
    common_benchmarks = set(baseline_benchmarks)
    for method, benchmarks in method_benchmarks.items():
        common_benchmarks &= benchmarks

    common_benchmarks = sorted(common_benchmarks)

    if not common_benchmarks:
        print("Error: No common benchmarks found across all files.")
        return

    print(f"Common benchmarks ({len(common_benchmarks)}): {', '.join(common_benchmarks)}")

    # Prepare output
    output_path = args.output or os.path.join(results_dir, "summary.csv")
    methods_ordered = [m for m in args.methods if m in method_data]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header: benchmark, baseline_base, baseline_instruct, method1, method2, ...
        # Apply METHOD_NAME_MAP to simplify method names in the header
        display_methods = [METHOD_NAME_MAP.get(m, m) for m in methods_ordered]
        writer.writerow(["benchmark", "baseline_base", "baseline_instruct"] + display_methods)

        # Benchmark rows
        for bench in common_benchmarks:
            row = [bench]

            # Baseline base models average
            row.append(compute_benchmark_average(baseline_data, bench, BASE_MODELS))

            # Baseline instruct models average
            row.append(compute_benchmark_average(baseline_data, bench, INSTRUCT_MODELS))

            # Method averages (over all models in each method's file)
            for method in methods_ordered:
                row.append(compute_benchmark_average(method_data[method], bench))

            writer.writerow(row)

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
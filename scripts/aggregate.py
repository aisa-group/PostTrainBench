#!/usr/bin/env python3
"""
Aggregate results across multiple runs per agent.

Reads final_{method}.csv files produced by collect.py and computes:
  --per-cell     : aggregated_avg_{agent}.csv, aggregated_std_{agent}.csv
  --leaderboard  : single_metrics.csv, single_metrics_aggregated.csv
  --time         : time_aggregated.csv
  --all          : everything (default)

Usage:
    python aggregate.py
    python aggregate.py --data-dir /path/to/results --output-dir /path/to/output
    python aggregate.py --per-cell --leaderboard
"""
import argparse
import csv
import os
import re

from utils import (
    get_results_dir,
    load_csv_as_dict,
    write_csv,
    load_factors,
    mean,
    stddev,
    is_number,
    format_time_hms,
    HARDCODED_AGENT_MAP,
    HARDCODED_BENCHMARKS,
    EXPECTED_MODELS,
)


# ---------------------------------------------------------------------------
# Per-cell avg/std across runs
# ---------------------------------------------------------------------------

def aggregate_per_cell(
    agent_name: str,
    method_names: list[str],
    data_dir: str,
    output_dir: str,
):
    """
    For each (model, benchmark) cell, compute mean and sample stddev
    across the runs. Write aggregated_avg_{agent}.csv and aggregated_std_{agent}.csv.
    """
    all_data = []
    all_models = None

    for method_name in method_names:
        csv_path = os.path.join(data_dir, f"final_{method_name}.csv")
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

    avg_data = {}
    std_data = {}

    for model in all_models:
        avg_data[model] = {}
        std_data[model] = {}

        for bench in HARDCODED_BENCHMARKS:
            values = []
            for data in all_data:
                values.append(float(data[model][bench]))

            avg_data[model][bench] = str(mean(values))
            std_data[model][bench] = str(stddev(values))

    avg_path = os.path.join(output_dir, f"aggregated_avg_{agent_name}.csv")
    write_csv(avg_path, all_models, HARDCODED_BENCHMARKS, avg_data)
    print(f"Written: {avg_path}")

    std_path = os.path.join(output_dir, f"aggregated_std_{agent_name}.csv")
    write_csv(std_path, all_models, HARDCODED_BENCHMARKS, std_data)
    print(f"Written: {std_path}")

    return avg_data, std_data


# ---------------------------------------------------------------------------
# Weighted single metric
# ---------------------------------------------------------------------------

def compute_weighted_metric(
    data: dict[str, dict[str, str]],
    factors: dict[str, float],
) -> float:
    """
    Compute weighted sum: for each benchmark, average across models,
    multiply by factor, sum.
    """
    valid_benchmarks = set(factors.keys())
    total = 0.0
    num_models = len(data)
    for bench in sorted(valid_benchmarks):
        values = [float(data[model][bench]) for model in data]
        avg_value = sum(values) / num_models
        total += avg_value * factors[bench]
    return total


def aggregate_leaderboard(data_dir: str, output_dir: str):
    """
    Compute weighted metric for every final_*.csv that has all expected models.
    Then group by HARDCODED_AGENT_MAP for avg/std.

    Also writes final_avg_{agent}.csv and final_std_{agent}.csv (identical to
    aggregated_ versions) so their metrics appear in single_metrics.csv.
    """
    factors = load_factors()
    valid_benchmarks = set(factors.keys())

    # Phase 1: compute per-cell avg/std and write final_avg/std files
    # so they get picked up in the metric scan below
    for agent_name, method_names in HARDCODED_AGENT_MAP.items():
        avg_data, std_data = _load_avg_std_for_agent(
            agent_name, method_names, data_dir
        )
        if avg_data is not None:
            # Write final_avg_{agent}.csv (identical to aggregated_avg_)
            avg_path = os.path.join(output_dir, f"final_avg_{agent_name}.csv")
            write_csv(
                avg_path,
                sorted(avg_data.keys()),
                HARDCODED_BENCHMARKS,
                avg_data,
            )
            std_path = os.path.join(output_dir, f"final_std_{agent_name}.csv")
            write_csv(
                std_path,
                sorted(std_data.keys()),
                HARDCODED_BENCHMARKS,
                std_data,
            )

    # Phase 2: compute metrics for ALL final_*.csv files in the output dir
    all_metrics = {}

    for filename in os.listdir(output_dir):
        if not filename.startswith("final_"):
            continue
        if not filename.endswith(".csv"):
            continue
        if filename.startswith("final_time_"):
            continue

        csv_path = os.path.join(output_dir, filename)
        try:
            data, _ = load_csv_as_dict(csv_path)
        except Exception:
            print(f"Warning: could not load {csv_path}.")
            raise

        if set(data.keys()) != EXPECTED_MODELS:
            continue

        method_name = filename[len("final_"):-len(".csv")]
        all_metrics[method_name] = compute_weighted_metric(data, factors)

    # Write individual metrics
    metrics_path = os.path.join(output_dir, "single_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric"])
        for method_name in sorted(all_metrics.keys()):
            writer.writerow([method_name, all_metrics[method_name]])
    print(f"Written: {metrics_path}")

    # Compute aggregated metrics per agent group
    aggregated_path = os.path.join(output_dir, "single_metrics_aggregated.csv")
    with open(aggregated_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "avg", "std", "n"])
        for agent_name in sorted(HARDCODED_AGENT_MAP.keys()):
            method_names = HARDCODED_AGENT_MAP[agent_name]
            # Skip agents with missing runs
            if not all(m in all_metrics for m in method_names):
                print(f"Skipping agent {agent_name} in leaderboard: missing metrics")
                continue
            metrics = [all_metrics[m] for m in method_names]
            writer.writerow([
                agent_name,
                mean(metrics),
                stddev(metrics),
                len(metrics),
            ])
    print(f"Written: {aggregated_path}")


def _load_avg_std_for_agent(
    agent_name: str,
    method_names: list[str],
    data_dir: str,
) -> tuple[dict | None, dict | None]:
    """Load final_*.csv for each run and compute per-cell avg/std."""
    all_data = []
    all_models = None

    for method_name in method_names:
        csv_path = os.path.join(data_dir, f"final_{method_name}.csv")
        if not os.path.exists(csv_path):
            return None, None
        data, _ = load_csv_as_dict(csv_path)
        models = sorted(data.keys())
        if all_models is None:
            all_models = models
        all_data.append(data)

    avg_data = {}
    std_data = {}
    for model in all_models:
        avg_data[model] = {}
        std_data[model] = {}
        for bench in HARDCODED_BENCHMARKS:
            values = [float(d[model][bench]) for d in all_data]
            avg_data[model][bench] = str(mean(values))
            std_data[model][bench] = str(stddev(values))

    return avg_data, std_data


# ---------------------------------------------------------------------------
# Time aggregation
# ---------------------------------------------------------------------------

def parse_time_to_hours(time_str: str) -> float:
    """Parse time string like '8:17:28' to hours as float."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours + minutes / 60 + seconds / 3600


def aggregate_time(data_dir: str, output_dir: str):
    """
    Read time_overview.csv, group by HARDCODED_AGENT_MAP, compute avg/std.
    Write time_aggregated.csv.
    """
    time_csv_path = os.path.join(data_dir, "time_overview.csv")

    time_data = {}
    with open(time_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            avg_time = row["average_time"]
            if avg_time and avg_time != "N/A":
                time_data[method] = parse_time_to_hours(avg_time)

    output_path = os.path.join(output_dir, "time_aggregated.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "avg_time", "std_time", "n"])
        for agent_name, method_names in HARDCODED_AGENT_MAP.items():
            if not all(m in time_data for m in method_names):
                print(f"Skipping agent {agent_name} in time: missing data")
                continue
            hours_list = [time_data[m] for m in method_names]
            writer.writerow([
                agent_name,
                format_time_hms(int(mean(hours_list) * 3600)),
                format_time_hms(int(stddev(hours_list) * 3600)),
                len(hours_list),
            ])
    print(f"Written: {output_path}")


def _all_finals_exist(method_names: list[str], data_dir: str) -> bool:
    """Check if all final_*.csv files exist for the given methods."""
    return all(
        os.path.exists(os.path.join(data_dir, f"final_{m}.csv"))
        for m in method_names
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate results across multiple runs per agent."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing final_*.csv files (from collect.py). "
        "Defaults to POST_TRAIN_BENCH_RESULTS_DIR or 'results'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output CSVs. Defaults to same as --data-dir.",
    )
    parser.add_argument("--per-cell", action="store_true",
                        help="Write per-cell avg/std CSVs per agent.")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Write single_metrics.csv and single_metrics_aggregated.csv.")
    parser.add_argument("--time", action="store_true",
                        help="Write time_aggregated.csv.")
    parser.add_argument("--all", action="store_true",
                        help="Write everything (default if no flags given).")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir or get_results_dir()
    output_dir = args.output_dir or data_dir

    os.makedirs(output_dir, exist_ok=True)

    do_all = args.all or not (args.per_cell or args.leaderboard or args.time)

    if do_all or args.per_cell:
        for agent_name, method_names in HARDCODED_AGENT_MAP.items():
            # Skip agents whose run data isn't available
            if not _all_finals_exist(method_names, data_dir):
                print(f"Skipping agent {agent_name}: missing final CSVs")
                continue
            print(f"Processing agent: {agent_name}")
            aggregate_per_cell(agent_name, method_names, data_dir, output_dir)

    if do_all or args.leaderboard:
        aggregate_leaderboard(data_dir, output_dir)

    if do_all or args.time:
        aggregate_time(data_dir, output_dir)


if __name__ == "__main__":
    main()

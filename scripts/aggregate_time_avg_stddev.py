#!/usr/bin/env python3
"""
Compute averages and standard deviations for time taken across multiple runs.

Reads from aggregated_time_overview.csv and computes statistics for each agent
group defined in HARDCODED_AGENT_MAP.
"""
import os
import csv
import math

from constants import HARDCODED_AGENT_MAP


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def parse_time_to_hours(time_str: str) -> float:
    """Parse time string like '8:17:28' or '0:55:17' to hours as float."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours + minutes / 60 + seconds / 3600


def load_time_csv(csv_path: str) -> dict[str, float]:
    """
    Load time CSV into dict: {method: hours}.
    """
    data = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            data[method] = parse_time_to_hours(row["average_time"])

    return data


def format_hours_to_time(hours: float) -> str:
    """Convert hours float back to H:MM:SS format."""
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def main():
    results_dir = get_results_dir()
    time_csv_path = os.path.join(results_dir, "aggregated_time_overview.csv")

    # Load time data
    time_data = load_time_csv(time_csv_path)

    # Compute aggregated statistics for each agent group
    aggregated_results = {}

    for agent_name, method_names in HARDCODED_AGENT_MAP.items():
        hours_list = []

        for method_name in method_names:
            hours_list.append(time_data[method_name])

        aggregated_results[agent_name] = {
            "avg_hours": mean(hours_list),
            "std_hours": stddev(hours_list),
            "n": len(hours_list),
        }

    # Write aggregated time CSV
    output_path = os.path.join(results_dir, "time_aggregated.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "avg_time", "std_time", "n"])
        for agent_name in HARDCODED_AGENT_MAP.keys():
            data = aggregated_results[agent_name]
            writer.writerow([
                agent_name,
                format_hours_to_time(data["avg_hours"]),
                format_hours_to_time(data["std_hours"]),
                data["n"],
            ])
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()

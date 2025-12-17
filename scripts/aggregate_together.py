#!/usr/bin/env python3
"""
Aggregate final_{method}.csv files into a single concatenated CSV.

Output format:
- Method name row
- Header row (model, benchmark1, benchmark2, ...)
- Data rows for that method
- Blank line
- Next method...
"""
import os
import csv
import argparse

METHOD_NAME_MAP = {
    "claude_claude-sonnet-4-5_final_v3": "claude sonnet 4.5",
    "claude_claude-opus-4-5_final_v3": "claude opus 4.5",
    "codex_gpt-5.1-codex-max_final_v3": "gpt-5.1-codex-max",
    "codex_gpt-5.2_final_v3": "gpt-5.2",
    "gemini_models_gemini-3-pro-preview_final_v3": "gemini-3-pro",
}

def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def load_csv_rows(csv_path: str) -> tuple[list, list]:
    """
    Load a CSV file and return (header, rows).
    """
    header = []
    rows = []

    if not os.path.exists(csv_path):
        return header, rows

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return [], []

        for row in reader:
            if row:
                rows.append(row)

    return header, rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate final CSVs into a single concatenated file."
    )
    parser.add_argument(
        "methods",
        nargs="+",
        help="List of methods to include in the aggregation.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV filename. Default: summary_concat.csv in results dir.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include baseline data as the first section.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = get_results_dir()

    output_path = args.output or os.path.join(results_dir, "summary_concat.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Optionally include baseline first
        if args.include_baseline:
            baseline_path = os.path.join(results_dir, "aggregated_baseline.csv")
            header, rows = load_csv_rows(baseline_path)

            if header and rows:
                writer.writerow(["baseline"])
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
                writer.writerow([])  # blank line
            else:
                print(f"Warning: No baseline data found at {baseline_path}")

        # Process each method
        for method in args.methods:
            final_path = os.path.join(results_dir, f"final_{method}.csv")
            header, rows = load_csv_rows(final_path)

            if not header or not rows:
                print(f"Warning: No data found for method '{method}' at {final_path}")
                continue

            display_name = METHOD_NAME_MAP.get(method, method)

            # Method name row
            writer.writerow([display_name])
            # Header row
            writer.writerow(header)
            # Data rows
            for row in rows:
                writer.writerow(row)
            # Blank line
            writer.writerow([])

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
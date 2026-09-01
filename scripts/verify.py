#!/usr/bin/env python3
"""
Verify that refactored aggregation scripts produce identical outputs
to the original pipeline.

Usage:
    python verify.py --ground-truth /fast/hbhatnagar/ptb_results/ \
                     --new-output /fast/hbhatnagar/ptb_results_new/

Compares all key output CSVs cell-by-cell:
  - final_{method}.csv          (per-method score grids)
  - contamination_{method}.csv  (per-method contamination flags)
  - single_metrics.csv          (weighted score per run)
  - single_metrics_aggregated.csv (avg/std per agent group)
  - aggregated_avg_{agent}.csv  (per-cell avg for multi-run agents)
  - aggregated_std_{agent}.csv  (per-cell std for multi-run agents)
  - time_aggregated.csv         (avg/std time per agent)
"""
import argparse
import csv
import os
import sys


FLOAT_TOLERANCE = 1e-10


def is_number(s: str) -> bool:
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_csv(path: str) -> list[list[str]]:
    with open(path, "r", newline="") as f:
        return list(csv.reader(f))


def compare_csvs(gt_path: str, new_path: str) -> list[str]:
    """
    Compare two CSVs cell-by-cell.
    Returns list of mismatch descriptions (empty = pass).
    """
    errors = []

    gt_rows = load_csv(gt_path)
    new_rows = load_csv(new_path)

    if len(gt_rows) != len(new_rows):
        errors.append(f"Row count differs: {len(gt_rows)} vs {len(new_rows)}")
        # Still compare what we can
        max_rows = min(len(gt_rows), len(new_rows))
    else:
        max_rows = len(gt_rows)

    for row_idx in range(max_rows):
        gt_row = gt_rows[row_idx]
        new_row = new_rows[row_idx]

        if len(gt_row) != len(new_row):
            errors.append(
                f"  Row {row_idx}: column count differs: "
                f"{len(gt_row)} vs {len(new_row)}"
            )
            max_cols = min(len(gt_row), len(new_row))
        else:
            max_cols = len(gt_row)

        for col_idx in range(max_cols):
            gt_val = gt_row[col_idx]
            new_val = new_row[col_idx]

            if gt_val == new_val:
                continue

            # Try numeric comparison with tolerance
            if is_number(gt_val) and is_number(new_val):
                if abs(float(gt_val) - float(new_val)) < FLOAT_TOLERANCE:
                    continue

            # Header row for context
            header_label = ""
            if row_idx > 0 and gt_rows[0]:
                col_name = gt_rows[0][col_idx] if col_idx < len(gt_rows[0]) else "?"
                row_name = gt_row[0] if gt_row else "?"
                header_label = f" ({row_name}, {col_name})"

            errors.append(
                f"  Row {row_idx}, Col {col_idx}{header_label}: "
                f"'{gt_val}' vs '{new_val}'"
            )

    return errors


def find_matching_files(gt_dir: str, new_dir: str) -> dict[str, tuple[str, str]]:
    """
    Find CSVs that exist in both directories, filtered to the ones we care about.
    Returns {filename: (gt_path, new_path)}.
    """
    matches = {}

    gt_files = set(f for f in os.listdir(gt_dir) if f.endswith(".csv"))
    new_files = set(f for f in os.listdir(new_dir) if f.endswith(".csv"))

    # Files we care about
    for f in sorted(gt_files & new_files):
        if should_verify(f):
            matches[f] = (os.path.join(gt_dir, f), os.path.join(new_dir, f))

    return matches


def should_verify(filename: str) -> bool:
    """Decide if a CSV file should be verified."""
    # Skip deprecated / intermediate / artifact files
    if filename in (
        "aggregated_avg_over_models.csv",
        "aggregated_std_over_models.csv",
    ):
        return False

    # Skip intermediate time CSVs (only time_aggregated.csv is a final output)
    if filename.startswith("aggregated_time_"):
        return False

    # Per-method final scores
    if filename.startswith("final_") and filename.endswith(".csv"):
        # Skip deprecated/artifact files
        if filename.startswith("final_avg_"):
            return False
        if filename.startswith("final_std_"):
            return False
        if filename.startswith("final_time_"):
            return False
        # Skip baselines (hardcoded in baselines.json, not regenerated)
        if filename in ("final_baseline.csv", "final_baseline_zeroshot.csv"):
            return False
        return True

    # Contamination flags
    if filename.startswith("contamination_") and filename.endswith(".csv"):
        # Skip baselines
        if filename in (
            "contamination_baseline.csv",
            "contamination_baseline_zeroshot.csv",
        ):
            return False
        return True

    # Single metric outputs
    if filename in ("single_metrics.csv", "single_metrics_aggregated.csv"):
        return True

    # Per-agent avg/std (multi-run agents)
    if filename.startswith("aggregated_avg_") or filename.startswith("aggregated_std_"):
        return True

    # Time aggregation
    if filename == "time_aggregated.csv":
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify refactored aggregation outputs match ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Directory with ground truth CSV outputs (from original scripts).",
    )
    parser.add_argument(
        "--new-output",
        required=True,
        help="Directory with new CSV outputs (from refactored scripts).",
    )
    args = parser.parse_args()

    gt_dir = args.ground_truth
    new_dir = args.new_output

    if not os.path.isdir(gt_dir):
        print(f"Error: ground truth dir not found: {gt_dir}")
        sys.exit(1)
    if not os.path.isdir(new_dir):
        print(f"Error: new output dir not found: {new_dir}")
        sys.exit(1)

    matches = find_matching_files(gt_dir, new_dir)

    if not matches:
        print("No matching CSV files found to compare.")
        sys.exit(1)

    # Check for files in ground truth that are missing from new output
    gt_verifiable = set(
        f for f in os.listdir(gt_dir) if f.endswith(".csv") and should_verify(f)
    )
    new_verifiable = set(
        f for f in os.listdir(new_dir) if f.endswith(".csv") and should_verify(f)
    )

    missing_from_new = gt_verifiable - new_verifiable
    extra_in_new = new_verifiable - gt_verifiable

    total_files = len(matches)
    passed = 0
    failed = 0
    failure_details = []

    print(f"Comparing {total_files} CSV files...\n")

    for filename, (gt_path, new_path) in sorted(matches.items()):
        errors = compare_csvs(gt_path, new_path)
        if errors:
            failed += 1
            failure_details.append((filename, errors))
            print(f"  FAIL  {filename}")
        else:
            passed += 1
            print(f"  PASS  {filename}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {total_files} total")

    if missing_from_new:
        print(f"\nMISSING from new output ({len(missing_from_new)}):")
        for f in sorted(missing_from_new):
            print(f"  - {f}")

    if extra_in_new:
        print(f"\nEXTRA in new output ({len(extra_in_new)}):")
        for f in sorted(extra_in_new):
            print(f"  + {f}")

    if failure_details:
        print(f"\nFailure details:")
        for filename, errors in failure_details:
            print(f"\n  {filename}:")
            for err in errors[:10]:  # Cap at 10 errors per file
                print(f"    {err}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")

    if failed or missing_from_new:
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

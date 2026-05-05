#!/usr/bin/env python3
"""
Aggregate rerun judge results and compare with original judgements.

Usage:
    python aggregate_rerun_results.py                     # Show summary
    python aggregate_rerun_results.py --csv output.csv    # Write to CSV
    python aggregate_rerun_results.py --diff-only         # Only show changed judgements
    python aggregate_rerun_results.py --filled-only       # Only show dirs where the rerun
                                                          #   filled in a previously missing
                                                          #   judgement
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from utils import get_result_dirs, parse_result_dir, read_judgement


def main():
    parser = argparse.ArgumentParser(description="Aggregate rerun judge results")
    parser.add_argument("--csv", type=str, help="Output CSV file")
    parser.add_argument("--diff-only", action="store_true",
                        help="Only show results where judgement changed")
    parser.add_argument("--filled-only", action="store_true",
                        help="Only show dirs where a missing original was filled by the rerun")
    parser.add_argument("--method", type=str, help="Filter by method pattern")
    parser.add_argument("--dirs", type=str, nargs="+",
                        help="Only process these specific result directories")
    args = parser.parse_args()

    if args.dirs:
        result_dirs = [Path(d) for d in args.dirs]
    else:
        result_dirs = get_result_dirs(method_pattern=args.method)

    results = []
    stats = defaultdict(int)

    for result_dir in result_dirs:
        try:
            parsed = parse_result_dir(result_dir)
        except ValueError:
            continue

        contam_orig = read_judgement(result_dir / "contamination_judgement.txt")
        contam_rerun = read_judgement(result_dir / "contamination_judgement_rerun.txt")
        model_orig = read_judgement(result_dir / "disallowed_model_judgement.txt")
        model_rerun = read_judgement(result_dir / "disallowed_model_judgement_rerun.txt")

        contam_changed = (
            contam_rerun is not None and contam_orig is not None and contam_orig != contam_rerun
        )
        model_changed = (
            model_rerun is not None and model_orig is not None and model_orig != model_rerun
        )
        contam_filled = contam_orig is None and contam_rerun is not None
        model_filled = model_orig is None and model_rerun is not None

        stats["total"] += 1
        if contam_rerun is not None:
            stats["has_rerun"] += 1
        if contam_changed:
            stats["contamination_changed"] += 1
        if model_changed:
            stats["model_changed"] += 1
        if contam_filled:
            stats["contamination_filled"] += 1
        if model_filled:
            stats["model_filled"] += 1

        result = {
            "method": parsed["method"],
            "benchmark": parsed["benchmark"],
            "model": parsed["model_hf"],
            "cluster_id": parsed["cluster_id"],
            "contamination_orig": contam_orig,
            "contamination_rerun": contam_rerun,
            "contamination_changed": contam_changed,
            "contamination_filled": contam_filled,
            "model_orig": model_orig,
            "model_rerun": model_rerun,
            "model_changed": model_changed,
            "model_filled": model_filled,
            "result_dir": str(result_dir),
        }

        if args.diff_only and not (contam_changed or model_changed):
            continue
        if args.filled_only and not (contam_filled or model_filled):
            continue

        results.append(result)

    if args.csv:
        if results:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"Wrote {len(results)} results to {args.csv}")
    else:
        print("=" * 80)
        print("Rerun Judge Results Summary")
        print("=" * 80)
        print()

        for result in results:
            print(f"Method: {result['method']}")
            print(f"  Folder: {result['result_dir']}")
            print(f"  Benchmark: {result['benchmark']}  Model: {result['model']}")

            tags = []
            if result["contamination_changed"]:
                tags.append("CHANGED")
            if result["contamination_filled"]:
                tags.append("FILLED")
            tag_str = f" [{','.join(tags)}]" if tags else ""
            print(
                f"  Contamination: {result['contamination_orig']} -> "
                f"{result['contamination_rerun']}{tag_str}"
            )

            tags = []
            if result["model_changed"]:
                tags.append("CHANGED")
            if result["model_filled"]:
                tags.append("FILLED")
            tag_str = f" [{','.join(tags)}]" if tags else ""
            print(
                f"  Model usage:   {result['model_orig']} -> "
                f"{result['model_rerun']}{tag_str}"
            )
            print()

    print("=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"Total result directories: {stats['total']}")
    print(f"With rerun judgements: {stats['has_rerun']}")
    print(f"Contamination changed (orig vs rerun): {stats['contamination_changed']}")
    print(f"Model usage changed (orig vs rerun): {stats['model_changed']}")
    print(f"Contamination filled (no orig, now rerun): {stats['contamination_filled']}")
    print(f"Model usage filled (no orig, now rerun): {stats['model_filled']}")


if __name__ == "__main__":
    main()

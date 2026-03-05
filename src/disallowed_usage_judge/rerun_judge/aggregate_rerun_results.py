#!/usr/bin/env python3
"""
Aggregate rerun judge results and compare with original judgements.

Usage:
    python aggregate_rerun_results.py                     # Show summary
    python aggregate_rerun_results.py --csv output.csv    # Write to CSV
    python aggregate_rerun_results.py --diff-only         # Only show changed judgements
"""

import csv
import argparse
from collections import defaultdict
from utils import get_result_dirs, parse_result_dir, get_trace_file, read_judgement


def main():
    parser = argparse.ArgumentParser(description="Aggregate rerun judge results")
    parser.add_argument("--csv", type=str, help="Output CSV file")
    parser.add_argument("--diff-only", action="store_true",
                        help="Only show results where judgement changed")
    parser.add_argument("--method", type=str, help="Filter by method pattern")
    args = parser.parse_args()

    result_dirs = get_result_dirs(method_pattern=args.method)

    results = []
    stats = defaultdict(int)

    for result_dir in result_dirs:
        try:
            parsed = parse_result_dir(result_dir)
        except ValueError:
            continue

        contamination_orig = read_judgement(result_dir / 'contamination_judgement.txt')
        contamination_rerun = read_judgement(result_dir / 'contamination_judgement_rerun.txt')
        model_orig = read_judgement(result_dir / 'disallowed_model_judgement.txt')
        model_rerun = read_judgement(result_dir / 'disallowed_model_judgement_rerun.txt')

        _, trace_source = get_trace_file(result_dir)
        trace_source = trace_source.replace('solve_', '').replace('.txt', '') if trace_source else 'none'

        contamination_changed = contamination_orig != contamination_rerun and contamination_rerun is not None
        model_changed = model_orig != model_rerun and model_rerun is not None

        stats['total'] += 1
        if contamination_rerun is not None:
            stats['has_rerun'] += 1
            if contamination_changed:
                stats['contamination_changed'] += 1
            if model_changed:
                stats['model_changed'] += 1

        result = {
            'method': parsed['method'],
            'benchmark': parsed['benchmark'],
            'model': parsed['model_hf'],
            'cluster_id': parsed['cluster_id'],
            'trace_source': trace_source,
            'contamination_orig': contamination_orig,
            'contamination_rerun': contamination_rerun,
            'contamination_changed': contamination_changed,
            'model_orig': model_orig,
            'model_rerun': model_rerun,
            'model_changed': model_changed,
            'result_dir': str(result_dir),
        }

        if args.diff_only and not (contamination_changed or model_changed):
            continue

        results.append(result)

    if args.csv:
        if results:
            with open(args.csv, 'w', newline='') as f:
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
            print(f"  Benchmark: {result['benchmark']}")
            print(f"  Model: {result['model']}")
            print(f"  Trace: {result['trace_source']}")

            if result['contamination_rerun']:
                marker = " [CHANGED]" if result['contamination_changed'] else ""
                print(f"  Contamination: {result['contamination_orig']} -> {result['contamination_rerun']}{marker}")

            if result['model_rerun']:
                marker = " [CHANGED]" if result['model_changed'] else ""
                print(f"  Model Usage: {result['model_orig']} -> {result['model_rerun']}{marker}")

            print()

    print("=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"Total result directories: {stats['total']}")
    print(f"With rerun judgements: {stats['has_rerun']}")
    print(f"Contamination judgement changed: {stats['contamination_changed']}")
    print(f"Model usage judgement changed: {stats['model_changed']}")


if __name__ == "__main__":
    main()

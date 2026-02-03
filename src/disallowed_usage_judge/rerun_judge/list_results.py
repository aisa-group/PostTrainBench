#!/usr/bin/env python3
"""
List and filter result directories for judge rerun.

Usage:
    python list_results.py                          # List all result directories
    python list_results.py --method "claude"        # Filter by method pattern
    python list_results.py --benchmark "aime"       # Filter by benchmark pattern
    python list_results.py --missing-rerun          # Show directories without rerun judgement
    python list_results.py --with-trace             # Show which trace file each directory has
    python list_results.py --paths-only             # Print just paths (for piping to other tools)
"""

import argparse
from utils import get_result_dirs, get_trace_file


def main():
    parser = argparse.ArgumentParser(description="List and filter result directories")
    parser.add_argument("--method", type=str, help="Filter by method pattern")
    parser.add_argument("--benchmark", type=str, help="Filter by benchmark pattern")
    parser.add_argument("--missing-rerun", action="store_true",
                        help="Only show directories without rerun judgement")
    parser.add_argument("--with-trace", action="store_true",
                        help="Show which trace file each directory has")
    parser.add_argument("--count-only", action="store_true",
                        help="Only show counts, not individual directories")
    parser.add_argument("--paths-only", action="store_true",
                        help="Print just paths (for piping)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of results")
    args = parser.parse_args()

    result_dirs = get_result_dirs(
        method_pattern=args.method,
        benchmark_pattern=args.benchmark,
        skip_existing=args.missing_rerun,
        limit=args.limit,
    )

    if args.count_only:
        print(len(result_dirs))
        return

    if args.paths_only:
        for d in result_dirs:
            print(d)
        return

    stats = {'total': 0, 'has_parsed': 0, 'has_out_only': 0, 'no_trace': 0, 'has_rerun': 0}

    for result_dir in result_dirs:
        stats['total'] += 1
        trace_path, trace_name = get_trace_file(result_dir)
        has_rerun = (result_dir / 'contamination_judgement_rerun.txt').exists()

        if trace_name == 'solve_parsed.txt':
            stats['has_parsed'] += 1
        elif trace_name == 'solve_out.txt':
            stats['has_out_only'] += 1
        else:
            stats['no_trace'] += 1

        if has_rerun:
            stats['has_rerun'] += 1

        if args.with_trace:
            trace_info = trace_name or "NO TRACE"
            rerun_status = "[RERUN]" if has_rerun else ""
            print(f"{result_dir} [{trace_info}] {rerun_status}")
        else:
            print(result_dir)

    print()
    print("=" * 50)
    print(f"Total: {stats['total']}")
    print(f"  With solve_parsed.txt: {stats['has_parsed']}")
    print(f"  With solve_out.txt only: {stats['has_out_only']}")
    print(f"  No trace file: {stats['no_trace']}")
    print(f"  Already has rerun judgement: {stats['has_rerun']}")


if __name__ == "__main__":
    main()

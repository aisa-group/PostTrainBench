#!/usr/bin/env python3
"""
List and filter result directories for judge rerun.

Examples:
    python list_results.py                              # List all result directories
    python list_results.py --method "claude"            # Filter by method substring
    python list_results.py --benchmark "aime"           # Filter by benchmark substring
    python list_results.py --skip-existing              # Skip dirs that already have rerun output
    python list_results.py --only-missing-judgement     # Only dirs where the original judge failed
    python list_results.py --paths-only                 # Print just paths (for piping)
    python list_results.py --latest-only                # Latest cluster_id per method/model/benchmark
"""
import argparse
from utils import get_result_dirs


def main():
    parser = argparse.ArgumentParser(description="List and filter result directories")
    parser.add_argument("--method", type=str, help="Filter by method pattern")
    parser.add_argument("--benchmark", type=str, help="Filter by benchmark pattern")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip directories that already have contamination_judgement_rerun.txt")
    parser.add_argument("--only-missing-judgement", action="store_true",
                        help="Only include directories where the original judge step "
                             "didn't write contamination_judgement.txt or "
                             "disallowed_model_judgement.txt")
    parser.add_argument("--paths-only", action="store_true",
                        help="Print just paths (for piping)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of results")
    parser.add_argument("--latest-only", action="store_true",
                        help="Only return latest cluster_id per method/model/benchmark")
    args = parser.parse_args()

    result_dirs = get_result_dirs(
        method_pattern=args.method,
        benchmark_pattern=args.benchmark,
        skip_existing=args.skip_existing,
        only_missing_judgement=args.only_missing_judgement,
        limit=args.limit,
        latest_only=args.latest_only,
    )

    if args.paths_only:
        for d in result_dirs:
            print(d)
        return

    has_rerun_count = 0
    missing_orig_count = 0
    for result_dir in result_dirs:
        has_rerun = (result_dir / "contamination_judgement_rerun.txt").exists()
        has_orig_contam = (result_dir / "contamination_judgement.txt").exists()
        has_orig_disallowed = (result_dir / "disallowed_model_judgement.txt").exists()

        flags = []
        if has_rerun:
            flags.append("RERUN")
            has_rerun_count += 1
        if not has_orig_contam or not has_orig_disallowed:
            flags.append("ORIG-MISSING")
            missing_orig_count += 1
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"{result_dir}{flag_str}")

    print()
    print("=" * 50)
    print(f"Total: {len(result_dirs)}")
    print(f"  Already has _rerun output: {has_rerun_count}")
    print(f"  Missing original judgement files: {missing_orig_count}")


if __name__ == "__main__":
    main()

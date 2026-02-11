#!/usr/bin/env python3
"""
Migrate old judgement files to new JSON format.

This script converts:
  - contamination_judgement.txt + disallowed_model_judgement.txt -> judge_result.json
  - contamination_judgement_rerun.txt + disallowed_model_judgement_rerun.txt -> judge_result_rerun.json

Usage:
    python migrate_judgement_files.py                          # Dry run (shows what would be done)
    python migrate_judgement_files.py --execute                # Actually perform the migration
    python migrate_judgement_files.py --execute --delete-old   # Migrate and delete old files
    python migrate_judgement_files.py --method "claude"        # Filter by method pattern
"""

import argparse
import json
import os
from pathlib import Path


def get_results_dir() -> Path:
    """Get the results directory from environment variable."""
    results_dir = os.environ.get('POST_TRAIN_BENCH_RESULTS_DIR')
    if not results_dir:
        raise RuntimeError("POST_TRAIN_BENCH_RESULTS_DIR is not set")
    return Path(results_dir)


def parse_contamination_file(content: str) -> bool | None:
    """Parse contamination_judgement.txt content to boolean."""
    content = content.strip()
    if content == "contamination detected":
        return True
    elif content == "no contamination detected":
        return False
    return None


def parse_disallowed_model_file(content: str) -> bool | None:
    """Parse disallowed_model_judgement.txt content to boolean."""
    content = content.strip()
    if content == "disallowed use detected":
        return True
    elif content == "only allowed use detected":
        return False
    return None


def migrate_result_dir(result_dir: Path, suffix: str, execute: bool, delete_old: bool) -> dict:
    """
    Migrate a single result directory's judgement files.

    Args:
        result_dir: Path to the result directory
        suffix: "" for normal, "_rerun" for rerun files
        execute: If True, actually create the new file
        delete_old: If True, delete old files after migration

    Returns:
        Dictionary with migration status
    """
    contamination_file = result_dir / f"contamination_judgement{suffix}.txt"
    disallowed_file = result_dir / f"disallowed_model_judgement{suffix}.txt"
    output_file = result_dir / f"judge_result{suffix}.json"

    status = {
        'result_dir': str(result_dir),
        'suffix': suffix,
        'has_contamination': contamination_file.exists(),
        'has_disallowed': disallowed_file.exists(),
        'has_output': output_file.exists(),
        'action': 'none',
        'contamination_value': None,
        'disallowed_value': None,
        'error': None,
    }

    # Skip if output already exists
    if output_file.exists():
        status['action'] = 'skip_exists'
        return status

    # Skip if neither input file exists
    if not contamination_file.exists() and not disallowed_file.exists():
        status['action'] = 'skip_no_input'
        return status

    # Parse contamination file
    contamination_detected = None
    if contamination_file.exists():
        content = contamination_file.read_text()
        contamination_detected = parse_contamination_file(content)
        if contamination_detected is None:
            status['error'] = f"Invalid contamination file content: {content.strip()!r}"
            status['action'] = 'error'
            return status
        status['contamination_value'] = contamination_detected

    # Parse disallowed model file
    disallowed_detected = None
    if disallowed_file.exists():
        content = disallowed_file.read_text()
        disallowed_detected = parse_disallowed_model_file(content)
        if disallowed_detected is None:
            status['error'] = f"Invalid disallowed model file content: {content.strip()!r}"
            status['action'] = 'error'
            return status
        status['disallowed_value'] = disallowed_detected

    # Create the JSON output
    judge_result = {}
    if contamination_detected is not None:
        judge_result['contamination_detected'] = contamination_detected
    if disallowed_detected is not None:
        judge_result['disallowed_model_detected'] = disallowed_detected

    if execute:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(judge_result, f, indent=2)
        status['action'] = 'migrated'

        if delete_old:
            if contamination_file.exists():
                contamination_file.unlink()
            if disallowed_file.exists():
                disallowed_file.unlink()
            status['action'] = 'migrated_deleted'
    else:
        status['action'] = 'would_migrate'

    return status


def get_all_result_dirs(method_pattern: str = None) -> list[Path]:
    """Get all result directories, optionally filtered by method."""
    results_root = get_results_dir()
    result_dirs = []

    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        if method_name.startswith('.') or method_name == 'baseline':
            continue

        if method_pattern and method_pattern.lower() not in method_name.lower():
            continue

        for result_dir in sorted(method_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            result_dirs.append(result_dir)

    return result_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Migrate old judgement files to new JSON format"
    )
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the migration (default is dry run)")
    parser.add_argument("--delete-old", action="store_true",
                        help="Delete old txt files after successful migration")
    parser.add_argument("--method", type=str,
                        help="Filter by method pattern")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all results, not just actions")
    args = parser.parse_args()

    result_dirs = get_all_result_dirs(method_pattern=args.method)

    stats = {
        'total_dirs': 0,
        'migrated': 0,
        'migrated_deleted': 0,
        'would_migrate': 0,
        'skip_exists': 0,
        'skip_no_input': 0,
        'error': 0,
    }

    print("=" * 70)
    if args.execute:
        print("MIGRATION MODE: Executing changes")
        if args.delete_old:
            print("WARNING: Old files will be deleted after migration")
    else:
        print("DRY RUN MODE: No changes will be made")
    print("=" * 70)
    print()

    for result_dir in result_dirs:
        stats['total_dirs'] += 1

        # Process normal judgement files
        status = migrate_result_dir(result_dir, "", args.execute, args.delete_old)
        stats[status['action']] = stats.get(status['action'], 0) + 1

        if status['action'] not in ('skip_exists', 'skip_no_input') or args.verbose:
            print(f"{result_dir}")
            print(f"  Normal: {status['action']}")
            if status['contamination_value'] is not None:
                print(f"    contamination_detected: {status['contamination_value']}")
            if status['disallowed_value'] is not None:
                print(f"    disallowed_model_detected: {status['disallowed_value']}")
            if status['error']:
                print(f"    ERROR: {status['error']}")

        # Process rerun judgement files
        status_rerun = migrate_result_dir(result_dir, "_rerun", args.execute, args.delete_old)
        stats[status_rerun['action']] = stats.get(status_rerun['action'], 0) + 1

        if status_rerun['action'] not in ('skip_exists', 'skip_no_input') or args.verbose:
            if status['action'] in ('skip_exists', 'skip_no_input'):
                print(f"{result_dir}")
            print(f"  Rerun: {status_rerun['action']}")
            if status_rerun['contamination_value'] is not None:
                print(f"    contamination_detected: {status_rerun['contamination_value']}")
            if status_rerun['disallowed_value'] is not None:
                print(f"    disallowed_model_detected: {status_rerun['disallowed_value']}")
            if status_rerun['error']:
                print(f"    ERROR: {status_rerun['error']}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total directories scanned: {stats['total_dirs']}")
    print(f"Would migrate (dry run): {stats.get('would_migrate', 0)}")
    print(f"Migrated: {stats.get('migrated', 0)}")
    print(f"Migrated and deleted old: {stats.get('migrated_deleted', 0)}")
    print(f"Skipped (output exists): {stats.get('skip_exists', 0)}")
    print(f"Skipped (no input files): {stats.get('skip_no_input', 0)}")
    print(f"Errors: {stats.get('error', 0)}")

    if not args.execute and stats.get('would_migrate', 0) > 0:
        print()
        print("Run with --execute to perform the migration")


if __name__ == "__main__":
    main()

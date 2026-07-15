#!/usr/bin/env python3
"""
Find result directories where the GPT-5.4 API-usage judge flagged disallowed
third-party LLM API usage.

Only the latest run (highest cluster_id) per (method, benchmark, base-model)
combination is considered, mirroring the selection used by the api-only rerun
submitter (commit_gpt_api_only.sh / list_results.py --latest-only).

The API verdict is read from judgement_api_rerun.json (written by
run_judges.sh --judges api_usage_judge); judgement_api.json is used as a
fallback for any directory that only carries the non-rerun file.

By default only the absolute paths of flagged directories are printed to
stdout (one per line) so the output can be piped. A run summary is written to
stderr.

POST_TRAIN_BENCH_RESULTS_DIR is taken from the environment if set, otherwise
read directly from the repo .env file (no need to source set_env_vars.sh).

Usage:
    python find_disallowed_api_usage.py                  # paths of flagged dirs
    python find_disallowed_api_usage.py --method claude  # restrict to a method
    python find_disallowed_api_usage.py --benchmark gsm8k
    python find_disallowed_api_usage.py --justification  # detailed report (stderr)
"""

import argparse
import json
import os
import sys
from pathlib import Path

from utils import get_repo_root, get_result_dirs

API_VERDICT_FILES = ("judgement_api_rerun.json", "judgement_api.json")


def load_results_dir_from_env() -> None:
    """Populate POST_TRAIN_BENCH_RESULTS_DIR from the repo .env if it is not
    already set in the environment.

    Mirrors the bash commit_* scripts, which read the variable straight out of
    .env rather than sourcing set_env_vars.sh (whose module-loading block fails
    on nodes without tclsh).
    """
    if os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR"):
        return

    env_file = get_repo_root() / ".env"
    if not env_file.exists():
        raise RuntimeError(f".env file not found at {env_file}")

    for line in env_file.read_text().splitlines():
        if line.startswith("POST_TRAIN_BENCH_RESULTS_DIR="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if not value:
                raise RuntimeError(
                    f"POST_TRAIN_BENCH_RESULTS_DIR is empty in {env_file}"
                )
            os.environ["POST_TRAIN_BENCH_RESULTS_DIR"] = value
            return

    raise RuntimeError(f"POST_TRAIN_BENCH_RESULTS_DIR not set in {env_file}")


def read_api_verdict(result_dir: Path) -> tuple[Path, dict] | tuple[None, None]:
    """Return (path, judgement_dict) for the API judge file, preferring the
    _rerun variant. Returns (None, None) if no API judge file exists.

    Malformed JSON or a missing/non-bool `disallowed_api_usage` field is a
    genuine error and is allowed to crash.
    """
    for name in API_VERDICT_FILES:
        path = result_dir / name
        if path.exists():
            data = json.loads(path.read_text())
            if not isinstance(data.get("disallowed_api_usage"), bool):
                raise ValueError(
                    f"{path}: 'disallowed_api_usage' missing or not a bool"
                )
            return path, data
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print absolute paths of latest result dirs flagged for "
        "disallowed third-party API usage."
    )
    parser.add_argument("--method", type=str, help="Filter by method pattern")
    parser.add_argument("--benchmark", type=str, help="Filter by benchmark pattern")
    parser.add_argument(
        "--justification",
        action="store_true",
        help="Also print each flagged dir's justification to stderr",
    )
    args = parser.parse_args()

    load_results_dir_from_env()

    result_dirs = get_result_dirs(
        method_pattern=args.method,
        benchmark_pattern=args.benchmark,
        latest_only=True,
    )

    flagged: list[tuple[Path, str]] = []
    scanned = 0
    missing_verdict = 0

    for result_dir in result_dirs:
        path, judgement = read_api_verdict(result_dir)
        if judgement is None:
            missing_verdict += 1
            continue
        scanned += 1
        if judgement["disallowed_api_usage"]:
            justification = judgement.get("justification_disallowed_api_usage", "")
            flagged.append((result_dir.resolve(), justification))

    # stdout: pure absolute paths, one per line (pipeable).
    for result_dir, _ in flagged:
        print(result_dir)

    # stderr: summary (and optionally justifications).
    print("=" * 60, file=sys.stderr)
    print(f"Latest result dirs considered:  {len(result_dirs)}", file=sys.stderr)
    print(f"  With an API judge verdict:    {scanned}", file=sys.stderr)
    print(f"  Missing an API judge verdict: {missing_verdict}", file=sys.stderr)
    print(f"  Flagged (disallowed API):     {len(flagged)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if args.justification and flagged:
        print("", file=sys.stderr)
        for result_dir, justification in flagged:
            print(f"### {result_dir}", file=sys.stderr)
            print(justification, file=sys.stderr)
            print("", file=sys.stderr)


if __name__ == "__main__":
    main()

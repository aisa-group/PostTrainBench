#!/usr/bin/env python3
"""
List every result directory where the PTB-lookup judge fired.

Walks ALL run directories (not just the latest per benchmark/model) under
POST_TRAIN_BENCH_RESULTS_DIR plus each root in
POST_TRAIN_BENCH_EXTRA_RESULTS_DIRS, both read from the project's .env file.
Roots pointing at the same directory are deduplicated, and a method directory
appearing under more than one root is only scanned once (first root wins,
mirroring collect.py).

The verdict is read via utils.load_ptb_lookup_judgement, which prefers
judgement_ptb_lookup_rerun.json (rerun pipeline) over judgement_ptb_lookup.json
(initial run_task.sh run). Run directories without either file are ignored —
they predate the judge — and only show up as a count in the summary.

stdout carries the absolute paths of flagged run dirs, one per line, so the
output can be piped. The summary (and, with --justification, the judge's
reasoning per flagged dir) goes to stderr.

Usage:
    python scripts/find_disallowed_ptb_lookup.py
    python scripts/find_disallowed_ptb_lookup.py --justification
"""

import argparse
import json
import os
import sys

from utils import (
    PTB_LOOKUP_FIELD,
    get_extra_results_dirs,
    get_results_dir,
    load_ptb_lookup_judgement,
    ptb_lookup_judgement_path,
)


def get_all_roots() -> list[str]:
    """Primary results root plus extras from .env, deduplicated by realpath."""
    roots = []
    seen = set()
    for root in [get_results_dir()] + get_extra_results_dirs():
        real = os.path.realpath(root)
        if real in seen:
            continue
        seen.add(real)
        roots.append(root)
    return roots


def iter_run_dirs(roots: list[str]):
    """Yield every run directory under every method directory of each root.

    A missing root is a hard error. Method names starting with '_' are
    derived-artifact dirs (e.g. _aggregated/), never methods. A method name
    seen under an earlier root shadows later copies (warned, like collect.py).
    """
    seen_method_root: dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"results root does not exist: {root}")

        for method_name in sorted(os.listdir(root)):
            method_path = os.path.join(root, method_name)
            if not os.path.isdir(method_path) or method_name.startswith("_"):
                continue

            if method_name in seen_method_root:
                print(
                    f"WARNING: method {method_name!r} found in {root} but "
                    f"already scanned from {seen_method_root[method_name]}; "
                    f"skipping this copy",
                    file=sys.stderr,
                )
                continue
            seen_method_root[method_name] = root

            for run_name in sorted(os.listdir(method_path)):
                run_path = os.path.join(method_path, run_name)
                if os.path.isdir(run_path):
                    yield run_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print absolute paths of all result dirs flagged by the "
        "PTB-lookup judge."
    )
    parser.add_argument(
        "--justification",
        action="store_true",
        help="Also print each flagged dir's justification to stderr",
    )
    args = parser.parse_args()

    roots = get_all_roots()

    flagged: list[tuple[str, str, str]] = []  # (run_dir, verdict_file, why)
    with_verdict = 0
    without_verdict = 0

    for run_dir in iter_run_dirs(roots):
        verdict = load_ptb_lookup_judgement(run_dir)
        if verdict is None:
            without_verdict += 1
            continue
        with_verdict += 1
        if verdict:
            verdict_path = ptb_lookup_judgement_path(run_dir)
            with open(verdict_path, "r") as f:
                justification = json.load(f).get(
                    f"justification_{PTB_LOOKUP_FIELD}", ""
                )
            flagged.append(
                (os.path.realpath(run_dir), verdict_path, justification)
            )

    # stdout: pure absolute paths, one per line (pipeable).
    for run_dir, _, _ in flagged:
        print(run_dir)

    # stderr: summary (and optionally justifications).
    print("=" * 60, file=sys.stderr)
    print(f"Results roots scanned:            {len(roots)}", file=sys.stderr)
    print(f"Run dirs with a lookup verdict:   {with_verdict}", file=sys.stderr)
    print(f"Run dirs without one (ignored):   {without_verdict}", file=sys.stderr)
    print(f"Flagged (disallowed PTB lookup):  {len(flagged)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if args.justification and flagged:
        print("", file=sys.stderr)
        for run_dir, verdict_path, justification in flagged:
            print(f"### {run_dir}", file=sys.stderr)
            print(f"    verdict file: {os.path.basename(verdict_path)}", file=sys.stderr)
            print(justification, file=sys.stderr)
            print("", file=sys.stderr)


if __name__ == "__main__":
    main()

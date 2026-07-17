#!/usr/bin/env python3
"""
List the latest result directory per benchmark/model flagged by a given judge.

The judge is selected with --judge <judge_name> (a directory under
src/judges/, e.g. ptb_lookup_judge or general_judge). Its verdict file name
comes from JUDGE_OUTPUT_ID in the judge's judge.conf, so any judge — present
or future — works without changes here. Every top-level boolean field in the
verdict JSON is treated as a flag (the single-flag judges have one, the
contamination judge has two); a run is flagged when any of them is true.

Walks only the latest run directory per (benchmark, model) within each
method directory — the same rule collect.py scores by, via
utils.walk_latest_runs — under POST_TRAIN_BENCH_RESULTS_DIR plus each root in
POST_TRAIN_BENCH_EXTRA_RESULTS_DIRS, both read from the project's .env file.
Roots pointing at the same directory are deduplicated, and a method directory
appearing under more than one root is only scanned once (first root wins,
mirroring collect.py).

The verdict path is resolved via utils.optional_judgement_path, which prefers
judgement_<id>_rerun.json (rerun pipeline) over judgement_<id>.json (initial
run_task.sh run). Run directories without either file are ignored — they
predate the judge — and only show up as a count in the summary.

stdout carries the absolute paths of flagged run dirs, one per line, so the
output can be piped. The summary (and, with --justification, the judge's
reasoning per flagged dir) goes to stderr. For multi-flag judges (the
contamination judge: contamination vs. disallowed_model) the summary breaks
the flagged count down per field, and each flagged dir is listed on stderr
prefixed with the field(s) that fired.

Usage:
    python scripts/find_flagged_runs.py --judge ptb_lookup_judge
    python scripts/find_flagged_runs.py --judge api_usage_judge
    python scripts/find_flagged_runs.py --judge data_contamination_judge
    python scripts/find_flagged_runs.py --judge general_judge --justification
"""

import argparse
import json
import os
import sys
from collections import Counter

from utils import (
    PROJECT_ROOT,
    get_extra_results_dirs,
    get_results_dir,
    optional_judgement_path,
    walk_latest_runs,
)

JUDGES_DIR = os.path.join(PROJECT_ROOT, "src", "judges")


def available_judges() -> list[str]:
    """Names of all judge directories under src/judges/ (those with a judge.conf)."""
    judges = sorted(
        name
        for name in os.listdir(JUDGES_DIR)
        if os.path.isfile(os.path.join(JUDGES_DIR, name, "judge.conf"))
    )
    if not judges:
        raise FileNotFoundError(f"no judge.conf found under {JUDGES_DIR}")
    return judges


def get_judge_output_id(judge_name: str) -> str:
    """Read JUDGE_OUTPUT_ID from the judge's judge.conf (KEY="value" lines)."""
    conf_path = os.path.join(JUDGES_DIR, judge_name, "judge.conf")
    with open(conf_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key.strip() == "JUDGE_OUTPUT_ID":
                return value.strip().strip('"')
    raise ValueError(f"{conf_path}: JUDGE_OUTPUT_ID not set")


def load_verdict(path: str) -> tuple[dict[str, bool], dict]:
    """Load a judgement file; return (boolean verdict fields, full JSON).

    Raises when the file is not a JSON object or holds no boolean field at
    all — then it is not a judge verdict.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON is not an object")
    flags = {k: v for k, v in data.items() if isinstance(v, bool)}
    if not flags:
        raise ValueError(f"{path}: no boolean verdict field found")
    return flags, data


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
    """Yield the latest run dir per (benchmark, model) of each method directory.

    "Latest" is the highest cluster id, decided per method directory by
    utils.walk_latest_runs — the same rule collect.py scores by. A missing
    root is a hard error. Method names starting with '_' are derived-artifact
    dirs (e.g. _aggregated/), never methods. A method name seen under an
    earlier root shadows later copies (warned, like collect.py).
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

            latest_runs = walk_latest_runs(method_path)
            for key in sorted(latest_runs):
                yield latest_runs[key]["path"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print absolute paths of the latest result dir per "
        "benchmark/model flagged by the given judge."
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=available_judges(),
        help="Judge whose verdicts to scan (directory name under src/judges/)",
    )
    parser.add_argument(
        "--justification",
        action="store_true",
        help="Also print each flagged dir's justification to stderr",
    )
    args = parser.parse_args()

    output_id = get_judge_output_id(args.judge)
    roots = get_all_roots()

    # (run_dir, verdict_file, {fired_field: justification})
    flagged: list[tuple[str, str, dict[str, str]]] = []
    with_verdict = 0
    without_verdict = 0
    field_names: set[str] = set()
    fired_counts: Counter[str] = Counter()

    for run_dir in iter_run_dirs(roots):
        verdict_path = optional_judgement_path(run_dir, output_id)
        if verdict_path is None:
            without_verdict += 1
            continue
        with_verdict += 1
        flags, data = load_verdict(verdict_path)
        field_names.update(flags)
        fired = [field for field, value in flags.items() if value]
        fired_counts.update(fired)
        if fired:
            justifications = {
                field: data.get(f"justification_{field}", "") for field in fired
            }
            flagged.append(
                (os.path.realpath(run_dir), verdict_path, justifications)
            )

    # stdout: pure absolute paths, one per line (pipeable).
    for run_dir, _, _ in flagged:
        print(run_dir)

    # stderr: summary (and optionally justifications).
    print("=" * 60, file=sys.stderr)
    print(f"Judge: {args.judge} (output id: {output_id})", file=sys.stderr)
    print(f"Results roots scanned:            {len(roots)}", file=sys.stderr)
    print(f"Latest run dirs with a verdict:   {with_verdict}", file=sys.stderr)
    print(f"... without one (ignored):        {without_verdict}", file=sys.stderr)
    print(f"Flagged:                          {len(flagged)}", file=sys.stderr)
    if len(field_names) > 1:
        for field in sorted(field_names):
            print(f"  {field + ':':<32}{fired_counts[field]}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if len(field_names) > 1 and flagged and not args.justification:
        print("", file=sys.stderr)
        for run_dir, _, justifications in flagged:
            print(f"[{', '.join(justifications)}] {run_dir}", file=sys.stderr)

    if args.justification and flagged:
        print("", file=sys.stderr)
        for run_dir, verdict_path, justifications in flagged:
            print(f"### {run_dir}", file=sys.stderr)
            print(f"    verdict file: {os.path.basename(verdict_path)}", file=sys.stderr)
            for field, justification in justifications.items():
                print(f"    [{field}]", file=sys.stderr)
                print(justification, file=sys.stderr)
            print("", file=sys.stderr)


if __name__ == "__main__":
    main()

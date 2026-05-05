#!/usr/bin/env python3
"""Aggregate per-run metrics JSON files into a single metrics_averaged.json.

Reads every file matching --runs-glob, treats top-level numeric keys as
per-run metric values, and writes mean/std/stderr/min/max per key plus the
raw per-run records and source file list.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys


def _numeric(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-glob", required=True,
                        help="Glob matching per-run metrics JSON files.")
    parser.add_argument("--output", required=True,
                        help="Path to write the aggregated metrics JSON.")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.runs_glob))
    if not paths:
        sys.exit(f"no run files matched {args.runs_glob}")

    runs: list[dict] = []
    for path in paths:
        with open(path, "r") as f:
            runs.append(json.load(f))

    keys = sorted({k for r in runs for k in r.keys()})

    aggregated: dict[str, dict[str, float | int]] = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and _numeric(r[k])]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        aggregated[k] = {
            "mean": mean,
            "std": std,
            "stderr": std / math.sqrt(len(vals)),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    out = {
        "n_runs": len(runs),
        "metrics": aggregated,
        "per_run": runs,
        "run_files": paths,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

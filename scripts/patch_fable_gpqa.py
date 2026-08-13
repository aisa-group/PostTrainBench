#!/usr/bin/env python3
"""Patch Fable 5 (Max)'s gpqamain cells with Opus-4.8 (Max)'s values.

Fable 5 (Max) has almost no gpqamain data (only 3 of 8 model×seed cells are
present with new-judge output, and one model has none at all). Until proper
gpqa runs finish, borrow Opus-4.8 (Max)'s gpqamain row so Fable's aggregated
numbers don't collapse to baseline_zeroshot for that benchmark.

Pipeline placement — run BETWEEN collect.py and aggregate.py:

    python scripts/collect.py
    python scripts/patch_fable_gpqa.py
    python scripts/aggregate.py

The patch modifies the per-seed final_*.csv files that collect.py writes
(one per Fable seed). aggregate.py then averages the patched cells naturally,
so aggregated_avg/std and single_metrics.csv all end up consistent.

Delete this script (and its call from your local workflow) once real Fable
gpqa runs land.
"""
import argparse
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from utils import get_aggregation_dir  # noqa: E402

# Per-seed final_*.csv basename (without the "final_" prefix / ".csv" suffix).
# Runs are paired 1:1 by index — Fable_run1 borrows from Opus_run1, etc. — so
# std stays honest instead of collapsing to zero.
FABLE_SEEDS = [
    "claude_non_api_max_claude-fable-5_1m__10h_run1",
    "claude_non_api_max_claude-fable-5_1m__10h_run2",
]
OPUS_SEEDS = [
    "claude_non_api_max_claude-opus-4-8_10h_run1",
    "claude_non_api_max_claude-opus-4-8_10h_run2",
]
BENCH_TO_PATCH = "gpqamain"


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def patch_pair(fable_seed: str, opus_seed: str, data_dir: str) -> None:
    fable_path = os.path.join(data_dir, f"final_{fable_seed}.csv")
    opus_path = os.path.join(data_dir, f"final_{opus_seed}.csv")

    if not os.path.exists(fable_path):
        raise FileNotFoundError(
            f"{fable_path} missing — run collect.py first."
        )
    if not os.path.exists(opus_path):
        raise FileNotFoundError(
            f"{opus_path} missing — Opus-4.8 (Max) source not collected."
        )

    fable_fields, fable_rows = read_csv(fable_path)
    _, opus_rows = read_csv(opus_path)

    # If Fable didn't run gpqamain at all in this seed, collect.py wrote a CSV
    # with no gpqamain column. Inject the column so the borrowed values can
    # land — otherwise aggregate.py's per-cell loop crashes on KeyError.
    if BENCH_TO_PATCH not in fable_fields:
        print(f"  ({BENCH_TO_PATCH} column absent, injecting)")
        fable_fields = list(fable_fields) + [BENCH_TO_PATCH]
        for row in fable_rows:
            row[BENCH_TO_PATCH] = ""

    opus_by_model = {row["model"]: row for row in opus_rows}

    changes = 0
    for row in fable_rows:
        model = row["model"]
        if model not in opus_by_model:
            raise KeyError(
                f"Model {model!r} in {fable_path} has no counterpart in "
                f"{opus_path}"
            )
        old = row[BENCH_TO_PATCH]
        new = opus_by_model[model][BENCH_TO_PATCH]
        if old != new:
            print(f"  {model:32s}  {BENCH_TO_PATCH}: {old} -> {new}")
            row[BENCH_TO_PATCH] = new
            changes += 1

    write_csv(fable_path, fable_fields, fable_rows)
    print(f"  patched {changes} cell(s) in {fable_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing final_*.csv files (from collect.py). "
        "Defaults to <POST_TRAIN_BENCH_RESULTS_DIR>/_aggregated.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir or get_aggregation_dir()

    if len(FABLE_SEEDS) != len(OPUS_SEEDS):
        raise RuntimeError("FABLE_SEEDS and OPUS_SEEDS must be the same length")

    for fable_seed, opus_seed in zip(FABLE_SEEDS, OPUS_SEEDS):
        print(f"Patching {fable_seed} <- {opus_seed}")
        patch_pair(fable_seed, opus_seed, data_dir)

    print("Done. Re-run aggregate.py to produce updated aggregated_/final_ CSVs.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Collect results from raw run directories into per-method CSVs.

For each method directory in the results dir, does a single pass:
  1. Finds the latest run per (benchmark, model)
  2. Reads metrics.json, the GPT-5.4 contamination judgement
     (judgement_gpt5_4_rerun.json if present, else judgement_gpt5_4.json),
     the API usage judgement (judgement_api_rerun.json if present, else
     judgement_api.json; absent for runs predating that judge), the
     PTB-lookup judgement (same rerun-over-original preference; archival —
     a True verdict raises instead of affecting scores), the general
     unknown-unknowns judgement (judgement_general[_rerun].json; archival,
     see below), and time_taken.txt
  3. Applies baseline fallback for cells flagged by the contamination or
     API judge, or with no run
  4. Writes final_{method}.csv, contamination_{method}.csv

Also writes a time_overview.csv summarising average time per method.

The general judge never affects any score. If it flagged any run, the
collection pass (steps 1-3) still completes for every method, but NOTHING is
written: collect.py raises an error listing every flagged run instead.
Double-check those runs; for each that looks fine, flip "general_anomaly" to
false in the verdict file named in the error, then re-run.

Any missing or malformed metrics.json / contamination judgement / time_taken.txt
inside an existing run directory is a hard error — there are no silent
fallbacks for broken runs. Cells with no run at all are filled from
baselines.json.

Usage:
    python collect.py
    python collect.py --data-dir /path/to/results --output-dir /path/to/output
    python collect.py --min-run-id 100 --max-run-id 200
"""
import argparse
import csv
import glob
import os

from utils import (
    get_results_dir,
    get_extra_results_dirs,
    get_aggregation_dir,
    get_baseline_fallback_data,
    walk_latest_runs,
    load_metrics,
    load_judgement,
    load_api_judgement,
    load_ptb_lookup_judgement,
    load_general_judgement,
    general_judgement_path,
    judgement_to_cell,
    load_time_taken,
    format_time_hms,
    BUDGET_SECONDS,
)

# Directories to skip (baselines are hardcoded in baselines.json)
SKIP_METHODS = {"baseline", "baseline_zeroshot"}


def collect_method(
    method_path: str,
    method_name: str,
    baseline_data: dict[str, dict[str, str]],
    min_run_id: int | None = None,
    max_run_id: int | None = None,
) -> dict | None:
    """
    Scan one method directory (no files are written here).

    Returns everything needed to write the method's CSVs, or None if no runs
    found:
      {"benchmarks", "models", "metrics_grid" (baseline fallback applied),
       "contamination_grid", "time_stats", "general_flagged"}

    Writing is deferred to write_method_csvs() so a general-judge flag in any
    method can abort aggregation before a single file is stored.
    """
    latest_runs = walk_latest_runs(method_path, min_run_id, max_run_id)
    if not latest_runs:
        return None

    benchmarks = sorted({b for b, m in latest_runs})
    models = sorted({m for b, m in latest_runs})

    # Collect metrics, contamination, and time in one pass
    metrics_grid = {}  # {model: {bench: str}}
    contamination_grid = {}  # {model: {bench: str}}
    general_flagged = []  # [(run_dir, verdict_path)]
    time_total_seconds = 0
    time_valid_count = 0

    for model in models:
        metrics_grid[model] = {}
        contamination_grid[model] = {}

        for bench in benchmarks:
            key = (bench, model)
            if key not in latest_runs:
                metrics_grid[model][bench] = ""
                contamination_grid[model][bench] = ""
                continue

            run_dir = latest_runs[key]["path"]

            try:
                # General-judge verdict first, so flagged runs are reported
                # even when the run is otherwise broken (e.g. missing
                # metrics.json). The verdict never affects scores; main()
                # aborts before writing anything when this list is non-empty.
                if load_general_judgement(run_dir):
                    general_flagged.append(
                        (run_dir, general_judgement_path(run_dir))
                    )
                metrics_grid[model][bench] = load_metrics(
                    os.path.join(run_dir, "metrics.json")
                )
                judgement = load_judgement(run_dir)
                api_usage = load_api_judgement(run_dir)
                # The PTB-lookup verdict is archival and never expected to
                # flag; a True verdict is a RuntimeError (not caught by the
                # broken-run handler below) so it cannot pass unnoticed.
                if load_ptb_lookup_judgement(run_dir):
                    raise RuntimeError(
                        f"PTB-lookup judge fired for {run_dir} "
                        f"(disallowed_ptb_lookup=true). Investigate this run "
                        f"before aggregating."
                    )
                contamination_grid[model][bench] = judgement_to_cell(
                    judgement, api_usage
                )
                _, seconds = load_time_taken(run_dir)
                time_total_seconds += seconds
                time_valid_count += 1
            except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
                # Broken run directory (missing/malformed metrics, judgement,
                # or time file). Fall through to baseline fallback. Skip the
                # warning when a final_eval_9.txt-style file exists — the
                # eval exhausted its retries, so a missing metrics.json is
                # expected. Matches both `final_eval_9.txt` and the rerun
                # naming `*_final_eval_9.txt` (e.g. `z_new_<id>_final_eval_9.txt`).
                if not glob.glob(os.path.join(run_dir, "*final_eval_9.txt")):
                    print(f"WARNING: skipping broken run {run_dir}: {e}")
                metrics_grid[model][bench] = ""
                contamination_grid[model][bench] = ""

    # Replace the cell with the baseline value if no run exists or the judge
    # flagged it. load_metrics() guarantees numeric strings when a run exists,
    # so the only non-numeric value here is "" for missing runs.
    for model in models:
        for bench in benchmarks:
            value = metrics_grid[model][bench]
            contamination_value = contamination_grid[model][bench]

            reasons = []
            if value == "":
                reasons.append("no run for this (benchmark, model)")
            if contamination_value:
                reasons.append(f"judge flagged ({contamination_value!r})")

            if not reasons:
                continue

            if model not in baseline_data or bench not in baseline_data[model]:
                raise KeyError(
                    f"baselines.json missing entry for model={model!r} "
                    f"benchmark={bench!r}; needed as fallback in method "
                    f"{method_name!r} (triggered by {', '.join(reasons)})"
                )
            metrics_grid[model][bench] = baseline_data[model][bench]

    return {
        "benchmarks": benchmarks,
        "models": models,
        "metrics_grid": metrics_grid,
        "contamination_grid": contamination_grid,
        "general_flagged": general_flagged,
        "time_stats": {
            "total_seconds": time_total_seconds,
            "valid_count": time_valid_count,
        },
    }


def write_method_csvs(method_name: str, collected: dict, output_dir: str):
    """Write contamination_{method}.csv and final_{method}.csv for one method
    from the grids collected by collect_method()."""
    benchmarks = collected["benchmarks"]
    models = collected["models"]

    contamination_path = os.path.join(
        output_dir, f"contamination_{method_name}.csv"
    )
    with open(contamination_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                row.append(collected["contamination_grid"][model][bench])
            writer.writerow(row)
    print(f"Written: {contamination_path}")

    final_path = os.path.join(output_dir, f"final_{method_name}.csv")
    with open(final_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                row.append(collected["metrics_grid"][model].get(bench, ""))
            writer.writerow(row)
    print(f"Written: {final_path}")


def write_time_overview(method_stats: dict[str, dict], output_dir: str):
    """Write time_overview.csv with average time per method."""
    csv_path = os.path.join(output_dir, "time_overview.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "average_time", "percentage"])

        for method_name in sorted(method_stats.keys()):
            stats = method_stats[method_name]
            total_secs = stats["total_seconds"]
            valid = stats["valid_count"]

            if valid > 0:
                avg_secs = total_secs // valid
                avg_str = format_time_hms(avg_secs)
                pct = (avg_secs / BUDGET_SECONDS) * 100
                pct_str = f"{pct:.1f}%"
            else:
                avg_str = "N/A"
                pct_str = "N/A"

            writer.writerow([method_name, avg_str, pct_str])

    print(f"Written: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect raw results into per-method CSVs."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing method subdirectories with raw run data. "
        "Defaults to POST_TRAIN_BENCH_RESULTS_DIR from the project's .env file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output CSVs. Defaults to "
        "<POST_TRAIN_BENCH_RESULTS_DIR>/_aggregated (kept out of the results "
        "root so it stays tidy).",
    )
    parser.add_argument(
        "--min-run-id",
        type=int,
        default=None,
        help="Inclusive lower bound for run IDs to consider.",
    )
    parser.add_argument(
        "--max-run-id",
        type=int,
        default=None,
        help="Exclusive upper bound for run IDs to consider.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir or get_results_dir()
    output_dir = args.output_dir or get_aggregation_dir()

    # Extras only apply to the env-driven primary; passing --data-dir means
    # "just this dir".
    extra_dirs = [] if args.data_dir else get_extra_results_dirs()
    all_roots = [data_dir] + extra_dirs

    # Load baseline data for fallback (hardcoded in baselines.json)
    baseline_data = get_baseline_fallback_data()

    collected_by_method: dict[str, dict] = {}
    seen_method_root: dict[str, str] = {}

    for root in all_roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"results root does not exist: {root}")

        for method_name in sorted(os.listdir(root)):
            method_path = os.path.join(root, method_name)
            if not os.path.isdir(method_path):
                continue

            # Skip derived-artifact dirs like _aggregated/. Method dirs never
            # start with an underscore.
            if method_name.startswith("_"):
                continue

            # Skip baseline directories — their values are hardcoded
            if method_name in SKIP_METHODS:
                continue

            if method_name in seen_method_root:
                print(
                    f"WARNING: method {method_name!r} found in {root} but "
                    f"already collected from {seen_method_root[method_name]}; "
                    f"skipping this copy"
                )
                continue
            seen_method_root[method_name] = root

            collected = collect_method(
                method_path,
                method_name,
                baseline_data,
                min_run_id=args.min_run_id,
                max_run_id=args.max_run_id,
            )
            if collected:
                collected_by_method[method_name] = collected

    # The general (unknown-unknowns) judge is archival: it never changes a
    # score, but a flagged run must not silently enter the aggregation. The
    # collection pass above still ran to completion; now refuse to store any
    # of it and list every flagged run for manual review.
    general_flagged = [
        flag
        for collected in collected_by_method.values()
        for flag in collected["general_flagged"]
    ]
    if general_flagged:
        listing = "\n".join(
            f"  {run_dir}\n    verdict: {verdict_path}"
            for run_dir, verdict_path in general_flagged
        )
        raise RuntimeError(
            f"General judge flagged {len(general_flagged)} run(s); no "
            f"aggregation files were written.\n{listing}\n"
            f"Double-check each run above (its judge_output_general*.txt "
            f"trace and the justification in the verdict file explain what "
            f"the judge saw). For every run that looks fine, flip "
            f'"general_anomaly" to false in the verdict file listed for it, '
            f"then re-run collect.py."
        )

    os.makedirs(output_dir, exist_ok=True)

    method_stats = {}
    for method_name, collected in collected_by_method.items():
        write_method_csvs(method_name, collected, output_dir)
        method_stats[method_name] = collected["time_stats"]

    if method_stats:
        write_time_overview(method_stats, output_dir)


if __name__ == "__main__":
    main()

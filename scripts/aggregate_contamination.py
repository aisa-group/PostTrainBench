#!/usr/bin/env python3
import os
import csv

OUTPUT_PREFIX = "contamination_"        # e.g. "contamination_method.csv"


def load_contamination(contamination_path: str):
    """
    Return True, False, "IMPORTANT ERR", or "ERR" based on contamination judgement.
    """
    if not os.path.exists(contamination_path):
        return "ERR"
    try:
        with open(contamination_path, "r") as f:
            content = f.read().strip()
    except Exception:
        return "ERR"
    if content == "contamination detected":
        return True
    elif content == "no contamination detected":
        return False
    else:
        return "IMPORTANT ERR"


def load_disallowed_model(disallowed_path: str):
    """
    Return True, False, "IMPORTANT ERR", or "ERR" based on disallowed model judgement.
    """
    if not os.path.exists(disallowed_path):
        return "ERR"
    try:
        with open(disallowed_path, "r") as f:
            content = f.read().strip()
    except Exception:
        return "ERR"
    if content == "disallowed use detected":
        return True
    elif content == "only allowed use detected":
        return False
    else:
        return "IMPORTANT ERR"


def combine_results(contamination, disallowed_model):
    """
    Combine contamination and disallowed model results into a single cell value.
    
    Returns:
        - "" if no illegal use detected (and no contamination)
        - "M" if disallowed model detected (but no contamination)
        - "MC" if disallowed model detected & contamination
        - "C" if only contamination detected (and no disallowed model)
        - Error string if either result is an error
    """
    # Handle error cases first
    if contamination in ("ERR", "IMPORTANT ERR") or disallowed_model in ("ERR", "IMPORTANT ERR"):
        errors = []
        if contamination in ("ERR", "IMPORTANT ERR"):
            errors.append(f"C:{contamination}")
        if disallowed_model in ("ERR", "IMPORTANT ERR"):
            errors.append(f"M:{disallowed_model}")
        return " ".join(errors)
    
    # Both are boolean now
    if disallowed_model and contamination:
        return "MC"
    elif disallowed_model and not contamination:
        return "M"
    elif not disallowed_model and contamination:
        return "C"
    else:  # not disallowed_model and not contamination
        return ""


def process_method(method_path: str, method_name: str):
    """
    For a single method dir (results/method_name), collect the newest run per
    (benchmark, model), then write a CSV.
    """
    # key: (benchmark, model) -> value: {"run_id": int, "path": str}
    latest_runs = {}

    for entry in os.listdir(method_path):
        entry_path = os.path.join(method_path, entry)
        if not os.path.isdir(entry_path):
            continue
        try: 
            benchmark, _, model, run_id = entry.split("_")
            key = (benchmark, model)
        except ValueError as e:
            print(entry)
            raise ValueError(f"{entry}, {method_path}")

        # keep only highest run_id per (benchmark, model)
        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {
                "run_id": run_id,
                "path": entry_path,
            }

    if not latest_runs:
        # nothing to do for this method
        return

    # Collect distinct benchmarks and models
    benchmarks = sorted({b for (b, m) in latest_runs.keys()})
    models = sorted({m for (b, m) in latest_runs.keys()})

    # Prepare CSV path (next to results/ or inside results/)
    csv_path = os.path.join(get_results_dir(), f"{OUTPUT_PREFIX}{method_name}.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # header
        writer.writerow(["model"] + benchmarks)

        # rows
        for model in models:
            row = [model]
            for bench in benchmarks:
                cell = ""
                key = (bench, model)
                if key in latest_runs:
                    run_dir = latest_runs[key]["path"]
                    contamination_path = os.path.join(run_dir, "contamination_judgement.txt")
                    disallowed_path = os.path.join(run_dir, "disallowed_model_judgement.txt")
                    
                    contamination = load_contamination(contamination_path)
                    disallowed_model = load_disallowed_model(disallowed_path)
                    cell = combine_results(contamination, disallowed_model)
                row.append(cell)
            writer.writerow(row)

    print(f"Written: {csv_path}")


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')


def main():
    results_dir = get_results_dir()
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue
        # treat every subdirectory of results/ as a "method"
        process_method(method_path, method_name)


if __name__ == "__main__":
    main()
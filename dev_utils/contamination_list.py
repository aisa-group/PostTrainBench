#!/usr/bin/env python3
import os

def load_contamination(contamination_path: str):
    """
    Return True if contamination is detected, otherwise False (or False on error).
    """
    if not os.path.exists(contamination_path):
        return False
    with open(contamination_path, "r") as f:
        content = f.read().strip()
        
    if content == "contamination detected":
        return True
    return False


def load_disallowed_model(disallowed_path: str):
    """
    Return True if disallowed use is detected, otherwise False (or False on error).
    """
    if not os.path.exists(disallowed_path):
        return False
    with open(disallowed_path, "r") as f:
        content = f.read().strip()

    if content == "disallowed use detected":
        return True
    return False


def get_latest_runs(method_path: str):
    """
    Scans a method directory and returns a list of paths corresponding 
    to the latest run_id for every (benchmark, model) pair.
    """
    # key: (benchmark, model) -> value: {"run_id": int, "path": str}
    latest_runs = {}

    for entry in os.listdir(method_path):
        entry_path = os.path.join(method_path, entry)
        if not os.path.isdir(entry_path):
            continue
        benchmark, _, model, run_id_str = entry.split("_")
        run_id = int(run_id_str)
        key = (benchmark, model)

        # keep only highest run_id per (benchmark, model)
        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {
                "run_id": run_id,
                "path": entry_path,
            }
            
    return [info["path"] for info in latest_runs.values()]


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')


def main():
    results_dir = get_results_dir()
    
    contaminated_list = []
    disallowed_list = []

    # 1. Iterate over all methods and collect paths
    for method_name in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(method_path):
            continue
        
        # Get only the latest runs for this method to avoid reporting old overwritten runs
        run_paths = get_latest_runs(method_path)

        for run_path in run_paths:
            # Check Contamination
            contam_path = os.path.join(run_path, "contamination_judgement.txt")
            if load_contamination(contam_path):
                contaminated_list.append(run_path)

            # Check Disallowed Model
            disallow_path = os.path.join(run_path, "disallowed_model_judgement.txt")
            if load_disallowed_model(disallow_path):
                disallowed_list.append(run_path)

    # 2. Output the lists
    print(f"=== CONTAMINATION DETECTED ({len(contaminated_list)}) ===")
    if contaminated_list:
        for path in sorted(contaminated_list):
            print(path)
    else:
        print("None")

    print("\n" + "-"*40 + "\n")

    print(f"=== DISALLOWED MODELS DETECTED ({len(disallowed_list)}) ===")
    if disallowed_list:
        for path in sorted(disallowed_list):
            print(path)
    else:
        print("None")


if __name__ == "__main__":
    main()
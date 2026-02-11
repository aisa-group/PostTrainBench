#!/usr/bin/env python3
"""
List all files with "safetensors" in their name within directories called "final_model".
"""
import os
import argparse


def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


def find_safetensors_in_final_model(root_dir: str) -> list[str]:
    """
    Recursively find all files with "safetensors" in their name
    that are located in directories called "final_model",
    but only if metrics.json exists in the parent directory.
    """
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if current directory is named "final_model"
        if os.path.basename(dirpath) == "final_model":
            # Check if metrics.json exists in the parent directory
            parent_dir = os.path.dirname(dirpath)
            metrics_path = os.path.join(parent_dir, "metrics.json")
            if not os.path.isfile(metrics_path):
                continue
            for filename in filenames:
                if filename.endswith(".safetensors"):
                    matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def main():
    parser = argparse.ArgumentParser(
        description="List safetensors files in final_model directories."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory to search. Defaults to results dir.",
    )
    args = parser.parse_args()

    root_dir = args.root or get_results_dir()

    if not os.path.isdir(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return 1

    files = find_safetensors_in_final_model(root_dir)

    if not files:
        print("No safetensors files found in final_model directories.")
        return 0

    print(f"Found {len(files)} safetensors file(s) in final_model directories:\n")
    for filepath in files:
        print(filepath)

    return 0


if __name__ == "__main__":
    exit(main())

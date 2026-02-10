#!/usr/bin/env python3
"""
Copy solve_parsed.txt (or solve_out.txt fallback) from result directories
to a new organized structure.
"""
import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Constants - modify these as needed
RESULTS_BASE = Path(os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
OUTPUT_DIR = os.path.join(RESULTS_BASE, "../collected_results")

# API key environment variables to check and redact
API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "MY_HF_TOKEN"
]


def get_api_keys() -> list[str]:
    """Get API key values from environment variables (non-empty only)."""
    keys = []
    for var in API_KEY_ENV_VARS:
        value = os.environ[var]
        keys.append(value)

    return keys


def sanitize_content(content: str, api_keys: list[str]) -> str:
    """Replace any API keys found in content with a placeholder."""
    for key in api_keys:
        content = content.replace(key, "<omitted-api-key>")
    return content


def copy_file_sanitized(src: Path, dest: Path, api_keys: list[str]) -> None:
    """Copy a file, sanitizing API keys from its content."""
    content = src.read_text(encoding="utf-8")
    sanitized = sanitize_content(content, api_keys)
    if content != sanitized:
        print(f"Sanitized API keys in file: {src}")
    dest.write_text(sanitized, encoding="utf-8")
    # Preserve file metadata
    shutil.copystat(src, dest)


def extract_model_name(dir_name: str) -> str:
    parts = dir_name.split("h_")
    if len(parts) > 2:
        raise ValueError(f"Unexpected directory name format: {dir_name}")
    if len(parts) == 1:
        return dir_name

    return parts[0] + "h"


def get_latest_subdirs(input_dir: Path) -> list[Path]:
    """
    Group subdirectories by their prefix (everything before the last _<id>)
    and return only the one with the highest numeric ID for each group.
    """
    grouped = defaultdict(list)
    
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        name = subdir.name
        parts = name.rsplit('_', 1)
        
        if len(parts) == 2 and parts[1].isdigit():
            prefix, id_str = parts
            grouped[prefix].append((int(id_str), subdir))
        else:
            # No numeric ID, treat the whole name as unique
            grouped[name].append((0, subdir))
    
    # For each group, keep only the one with the highest ID
    latest = []
    for prefix, entries in grouped.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest.append(entries[0][1])
    
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Copy solve_parsed.txt (or solve_out.txt fallback) from result directories to a new organized structure."
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Input directory names (relative to RESULTS_BASE) to process"
    )
    args = parser.parse_args()

    output_base = Path(OUTPUT_DIR)
    api_keys = get_api_keys()

    for input_dir_name in args.input_dirs:
        input_dir = RESULTS_BASE / input_dir_name
        
        if not input_dir.is_dir():
            print(f"Warning: Directory does not exist: {input_dir}")
            continue
        
        model_name = extract_model_name(input_dir_name)
        model_dir = output_base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate over only the latest subdirectories (highest ID per prefix)
        for subdir in get_latest_subdirs(input_dir):
            # Determine source file (prefer solve_parsed.txt)
            src_file = subdir / "solve_parsed.txt"
            solve_filename = "solve_parsed.txt"
            if not src_file.exists():
                src_file = subdir / "solve_out.txt"
                solve_filename = "solve_out.txt"
                if not src_file.exists():
                    print(f"Warning: No solve_parsed.txt or solve_out.txt in {subdir}")
                    continue
            
            # Create output directory with same name as original subdirectory
            task_name = subdir.name
            dest_dir = model_dir / task_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy solve file with original filename
            dest_file = dest_dir / solve_filename
            copy_file_sanitized(src_file, dest_file, api_keys)
            print(f"Copied: {src_file} -> {dest_file}")

            copy_other_files(subdir, dest_dir, 'metrics.json', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'contamination_judgement.txt', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'disallowed_model_judgement.txt', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'error.log', 'judgement.log', api_keys=api_keys)

def copy_other_files(subdir, dest_dir, filename, dest_filename=None, api_keys=None):
    if dest_filename is None:
        dest_filename = filename
    if api_keys is None:
        api_keys = []
    src_metrics = subdir / filename
    dest_metrics = dest_dir / dest_filename
    if src_metrics.exists():
        copy_file_sanitized(src_metrics, dest_metrics, api_keys)
    else:
        with open(dest_metrics, 'w') as f:
            f.write(f"No {filename} produced.")

if __name__ == "__main__":
    main()
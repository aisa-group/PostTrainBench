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
    "BEN_HF_TOKEN",
    "HARDIK_HF_TOKEN",
    "OPENCODE_API_KEY",
    "ZAI_API_KEY",
    "DASHSCOPE_API_KEY"

]

API_KEY_PATTERNS = [
    "sk-proj",      # OpenAI project keys
    "sk-ant",       # Anthropic keys
    "AIzaSy",       # Google/Gemini keys
    # "sk-",        # Generic OpenAI keys - too broad (matches "mask-in", etc). Covered by sk-proj/sk-ant.
    # "hf_",        # HuggingFace tokens - too broad (matches hf_cache, hf_home etc). Actual tokens redacted via env vars.
    # not needed
    # AWS
    # "AKIA",         # AWS access key IDs
    # GitHub
    # "ghp_",         # GitHub personal access tokens
    # "gho_",         # GitHub OAuth tokens
    # "ghs_",         # GitHub app installation tokens
    # "ghr_",         # GitHub refresh tokens
    # # GitLab
    # "glpat-",       # GitLab personal access tokens
    # AI services
    "sk-or-",       # OpenRouter
    # "r8_",          # Replicate
    # "xai-",         # xAI/Grok
    # "nvapi-",       # NVIDIA
    # Slack
    # "xoxb-",        # Slack bot tokens
    # "xoxp-",        # Slack user tokens
    # Stripe
    # "sk_live_",     # Stripe live secret keys
    # "sk_test_",     # Stripe test secret keys
    # "whsec_",       # Stripe webhook secrets
    # Other
    # "SG.",          # SendGrid API keys
]


def get_api_keys() -> list[str]:
    """Get API key values from environment variables (non-empty only)."""
    keys = []
    for var in API_KEY_ENV_VARS:
        if var not in os.environ:
            raise ValueError(f"Expected environment variable not set: {var}")
        value = os.environ[var]
        keys.append(value)

    return keys

_warnings = []

def warn_if_api_key_in_content(content: str, prefix: str, src_path: str = "") -> None:
    if prefix in content:
        idx = content.index(prefix)
        start = max(0, idx - 50)
        end = min(len(content), idx + 50)
        context = content[start:end].replace('\n', '\\n')
        _warnings.append({
            "pattern": prefix,
            "file": src_path,
            "context": context,
        })

def sanitize_content(content: str, api_keys: list[str], src_path: str = "") -> str:
    """Replace any API keys found in content with a placeholder."""
    import re
    for key in api_keys:
        content = content.replace(key, "<omitted-api-key>")
        # Also redact truncated versions (agent output may truncate keys)
        # Use the first 10 chars as an anchor and redact the full key-like token
        if len(key) >= 10:
            prefix = re.escape(key[:10])
            content = re.sub(prefix + r'[A-Za-z0-9_\-]+', '<omitted-api-key>', content)
    for pattern in API_KEY_PATTERNS:
        warn_if_api_key_in_content(content, pattern, src_path)

    return content


def copy_file_sanitized(src: Path, dest: Path, api_keys: list[str]) -> None:
    """Copy a file, sanitizing API keys from its content."""
    content = src.read_text(encoding="utf-8")
    sanitized = sanitize_content(content, api_keys, src_path=str(src))
    dest.write_text(sanitized, encoding="utf-8")
    # Preserve file metadata
    shutil.copystat(src, dest)
    return content != sanitized


def extract_model_name(dir_name: str) -> str:
    return dir_name


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

    copied_count = 0
    sanitized_count = 0
    missing_count = 0

    for input_dir_name in args.input_dirs:
        input_dir = RESULTS_BASE / input_dir_name

        if not input_dir.is_dir():
            print(f"  SKIP: {input_dir} does not exist")
            continue

        model_name = extract_model_name(input_dir_name)
        model_dir = output_base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{input_dir_name}]")

        # Iterate over only the latest subdirectories (highest ID per prefix)
        for subdir in sorted(get_latest_subdirs(input_dir)):
            # Determine source file (prefer solve_parsed.txt)
            src_file = subdir / "solve_parsed.txt"
            if not src_file.exists():
                src_file = subdir / "solve_out.txt"
                if not src_file.exists():
                    print(f"  MISS: {subdir.name} (no trace file)")
                    missing_count += 1
                    continue

            # Create output directory with same name as original subdirectory
            task_name = subdir.name
            dest_dir = model_dir / task_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy solve file with original filename
            dest_file = dest_dir / "trace.txt"
            was_sanitized = copy_file_sanitized(src_file, dest_file, api_keys)
            if was_sanitized:
                sanitized_count += 1

            copy_other_files(subdir, dest_dir, 'metrics.json', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'contamination_judgement.txt', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'disallowed_model_judgement.txt', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'error.log', 'judgement.log', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'system_monitor.log', api_keys=api_keys, optional=True)

            tag = " [sanitized]" if was_sanitized else ""
            print(f"  OK: {subdir.name}{tag}")
            copied_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {copied_count} copied, {sanitized_count} sanitized, {missing_count} missing")
    print(f"Output: {output_base}")

    if _warnings:
        print(f"\n--- {len(_warnings)} pattern warnings (review manually) ---")
        for w in _warnings:
            print(f"  [{w['pattern']}] {w['file']}")
            print(f"    ...{w['context']}...")

def copy_other_files(subdir, dest_dir, filename, dest_filename=None, api_keys=None, optional=False):
    if dest_filename is None:
        dest_filename = filename
    if api_keys is None:
        api_keys = []
    src_metrics = subdir / filename
    dest_metrics = dest_dir / dest_filename
    if src_metrics.exists():
        copy_file_sanitized(src_metrics, dest_metrics, api_keys)
    elif not optional:
        with open(dest_metrics, 'w') as f:
            f.write(f"No {filename} produced.")

if __name__ == "__main__":
    main()
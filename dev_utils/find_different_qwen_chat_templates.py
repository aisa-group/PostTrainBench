#!/usr/bin/env python3
"""
Find result folders containing Qwen3 models with chat templates that differ
from the original Qwen3 chat template.

Usage:
    python find_different_chat_templates.py [--verbose] [--group] [--count-by-method] [--dirs-only] [--sort-by-count]

Options:
    --verbose          Show template contents for different folders
    --group            Group folders by their template hash
    --count-by-method  Show count of different templates per method
    --dirs-only        Output only directory paths, sorted by scaffold
    --sort-by-count    With --dirs-only: within each scaffold group, sort by
                       number of folders sharing the same template hash (desc)
"""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# Original Qwen3 chat template from Qwen/Qwen3-4B-Base
# This is embedded in tokenizer_config.json of the base model
ORIGINAL_QWEN3_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""


def normalize_template(template: str) -> str:
    """Normalize a template for comparison by stripping whitespace."""
    return template.strip()


def get_chat_template_from_jinja(jinja_path: Path) -> str | None:
    """Read chat template from a .jinja file."""
    if not jinja_path.exists():
        return None
    return jinja_path.read_text()


def get_chat_template_from_tokenizer_config(config_path: Path) -> str | None:
    """Read chat template from tokenizer_config.json."""
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    return config.get("chat_template")


def get_chat_template(final_model_path: Path) -> str | None:
    """Get chat template from a final_model directory.

    Checks chat_template.jinja first, then tokenizer_config.json.
    """
    jinja_template = get_chat_template_from_jinja(final_model_path / "chat_template.jinja")
    if jinja_template is not None:
        return jinja_template

    return get_chat_template_from_tokenizer_config(final_model_path / "tokenizer_config.json")


@dataclass
class ResultFolderInfo:
    """Parsed information from a result folder path."""

    path: Path
    method: str  # e.g., "claude_claude-opus-4-5_10h_v5"
    benchmark: str  # e.g., "aime2025"
    model: str  # e.g., "Qwen_Qwen3-4B-Base"
    cluster_id: int  # e.g., 16675578


def parse_result_folder(folder: Path) -> ResultFolderInfo | None:
    """Parse a result folder path into its components.

    Folder structure: {results_dir}/{method}/{benchmark}_{model}_{cluster_id}
    Example: results/claude_claude-opus-4-5_10h_v5/aime2025_Qwen_Qwen3-4B-Base_16675578
    """
    method = folder.parent.name
    folder_name = folder.name

    # Match pattern: {benchmark}_{model}_{cluster_id}
    # The cluster_id is always the last part after the final underscore
    match = re.match(r"^(.+)_(\d+)$", folder_name)
    if not match:
        return None

    prefix, cluster_id_str = match.groups()
    cluster_id = int(cluster_id_str)

    # Split prefix into benchmark and model
    # Model always starts with the HF org (e.g., "Qwen_Qwen3-4B-Base", "HuggingFaceTB_SmolLM3-3B-Base")
    # Common patterns: Qwen_, google_, HuggingFaceTB_, meta-llama_
    model_orgs = ["Qwen_", "google_", "HuggingFaceTB_", "meta-llama_"]

    benchmark = None
    model = None
    for org in model_orgs:
        idx = prefix.find(org)
        if idx != -1:
            benchmark = prefix[:idx].rstrip("_")
            model = prefix[idx:]
            break

    if benchmark is None or model is None:
        return None

    return ResultFolderInfo(
        path=folder,
        method=method,
        benchmark=benchmark,
        model=model,
        cluster_id=cluster_id,
    )


def is_qwen3_folder(folder_name: str) -> bool:
    """Check if a folder name indicates it's a Qwen3 model result."""
    return "qwen3" in folder_name.lower()


def find_qwen3_result_folders(results_dir: Path) -> list[ResultFolderInfo]:
    """Find all Qwen3 result folders, deduplicated to keep only latest run.

    For each (method, benchmark, model) combination, keeps only the folder
    with the highest cluster_id.
    """
    # First pass: collect all Qwen3 folders with parsed info
    all_folders: dict[tuple[str, str, str], list[ResultFolderInfo]] = defaultdict(list)

    for experiment_dir in results_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        for result_folder in experiment_dir.iterdir():
            if not result_folder.is_dir():
                continue

            if not is_qwen3_folder(result_folder.name):
                continue

            info = parse_result_folder(result_folder)
            if info is None:
                continue

            key = (info.method, info.benchmark, info.model)
            all_folders[key].append(info)

    # Second pass: keep only the highest cluster_id for each combination
    deduplicated = []
    for folders in all_folders.values():
        latest = max(folders, key=lambda f: f.cluster_id)
        deduplicated.append(latest)

    return deduplicated


def find_folders_with_different_templates(
    results_dir: Path,
) -> list[tuple[ResultFolderInfo, str]]:
    """Find Qwen3 folders with chat templates different from the original.

    Returns list of (ResultFolderInfo, actual_template) tuples.
    Only considers the latest run (highest cluster_id) for each
    (method, benchmark, model) combination.
    """
    original_normalized = normalize_template(ORIGINAL_QWEN3_CHAT_TEMPLATE)
    different_folders = []

    qwen3_folders = find_qwen3_result_folders(results_dir)

    for info in qwen3_folders:
        final_model_path = info.path / "final_model"
        if not final_model_path.exists():
            continue

        template = get_chat_template(final_model_path)
        if template is None:
            print(f"Warning: No chat template found in {final_model_path}")
            continue

        template_normalized = normalize_template(template)
        if template_normalized != original_normalized:
            different_folders.append((info, template))

    return different_folders


def template_hash(template: str) -> str:
    """Generate a short hash for a template."""
    return hashlib.md5(normalize_template(template).encode()).hexdigest()[:8]


def template_summary(template: str) -> str:
    """Generate a brief summary describing key template features."""
    features = []

    if "tools" in template.lower() or "tool_call" in template.lower():
        features.append("tool_calling")
    if "<think>" in template or "reasoning_content" in template:
        features.append("thinking")
    if "multi_step_tool" in template:
        features.append("multi_step")

    if not features:
        features.append("basic")

    lines = template.strip().split("\n")
    return f"{len(lines)} lines, features: {', '.join(features)}"


def count_total_qwen3_folders(results_dir: Path) -> dict[str, int]:
    """Count total Qwen3 folders per method (after deduplication)."""
    qwen3_folders = find_qwen3_result_folders(results_dir)
    counts: dict[str, int] = defaultdict(int)
    for info in qwen3_folders:
        counts[info.method] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Find Qwen3 result folders with non-standard chat templates"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show template contents"
    )
    parser.add_argument(
        "--group", "-g", action="store_true", help="Group folders by template hash"
    )
    parser.add_argument(
        "--count-by-method",
        "-c",
        action="store_true",
        help="Show count of different templates per method",
    )
    parser.add_argument(
        "--dirs-only",
        "-d",
        action="store_true",
        help="Output only directory paths, sorted by scaffold",
    )
    parser.add_argument(
        "--sort-by-count",
        "-s",
        action="store_true",
        help="With --dirs-only: within each scaffold, sort by template hash frequency (descending)",
    )
    args = parser.parse_args()

    results_dir = Path(os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    different = find_folders_with_different_templates(results_dir)

    if not different:
        print("No Qwen3 folders with different chat templates found.")
        return

    if args.dirs_only:
        def scaffold_key(info: ResultFolderInfo) -> str:
            return info.method.split("_")[0]

        if args.sort_by_count:
            def hours_key(info: ResultFolderInfo) -> str:
                """Extract hours component (e.g. '10h') from method name."""
                match = re.search(r"_(\d+h)_", info.method)
                return match.group(1) if match else ""

            # Count how many folders share each (scaffold, hours, benchmark, model)
            # e.g. aime2025_Qwen_Qwen3-1.7B-Base across 3 gemini 10h methods → count 3
            # but 1h and 10h runs are counted separately
            group_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
            for info, _ in different:
                key = (scaffold_key(info), hours_key(info), info.benchmark, info.model)
                group_counts[key] += 1

            def sort_key(item: tuple[ResultFolderInfo, str]) -> tuple[str, str, int, str, str, str]:
                info, _ = item
                scaffold = scaffold_key(info)
                hours = hours_key(info)
                count = group_counts[(scaffold, hours, info.benchmark, info.model)]
                # Primary: scaffold, hours, then descending count,
                # then benchmark, model, method
                return (scaffold, hours, -count, info.benchmark, info.model, info.method)

            for info, template in sorted(different, key=sort_key):
                print(info.path)
        else:
            for info, _ in sorted(different, key=lambda x: (scaffold_key(x[0]), x[0].method, x[0].benchmark)):
                print(info.path)
        return

    print(f"Found {len(different)} Qwen3 folder(s) with different chat templates:\n")

    if args.count_by_method:
        # Get total counts per method for context
        total_counts = count_total_qwen3_folders(results_dir)

        # Count different templates per method
        diff_counts: dict[str, int] = defaultdict(int)
        for info, _ in different:
            diff_counts[info.method] += 1

        # Display sorted by count (descending)
        print(f"{'Method':<60} {'Different':>10} {'Total':>8} {'Pct':>8}")
        print("-" * 88)
        for method in sorted(diff_counts.keys(), key=lambda m: -diff_counts[m]):
            diff = diff_counts[method]
            total = total_counts.get(method, 0)
            pct = (diff / total * 100) if total > 0 else 0
            print(f"{method:<60} {diff:>10} {total:>8} {pct:>7.1f}%")

    elif args.group:
        # Group by template hash
        groups: dict[str, list[tuple[ResultFolderInfo, str]]] = defaultdict(list)
        for info, template in different:
            h = template_hash(template)
            groups[h].append((info, template))

        for h, items in sorted(groups.items(), key=lambda x: -len(x[1])):
            _, sample_template = items[0]
            summary = template_summary(sample_template)
            print(f"=== Template hash: {h} ({len(items)} folders) ===")
            print(f"    {summary}")
            for info, _ in items:
                print(f"  {info.path}")
            if args.verbose:
                print("\n--- Template content ---")
                print(sample_template[:1000])
                if len(sample_template) > 1000:
                    print(f"... ({len(sample_template) - 1000} more chars)")
            print()
    else:
        for info, template in different:
            h = template_hash(template)
            summary = template_summary(template)
            print(f"{info.path}")
            print(f"  hash={h}, {summary}")
            if args.verbose:
                print("  --- Template ---")
                for line in template.split("\n")[:10]:
                    print(f"  {line}")
                if template.count("\n") > 10:
                    print(f"  ... ({template.count(chr(10)) - 10} more lines)")
            print()


if __name__ == "__main__":
    main()

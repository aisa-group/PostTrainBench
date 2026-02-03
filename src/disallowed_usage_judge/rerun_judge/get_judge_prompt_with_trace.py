#!/usr/bin/env python3
"""Generate the judge prompt with trace file reference."""

import os
import json
import argparse
from pathlib import Path


def load_data_examples(benchmark_id: str) -> tuple[list, list]:
    """Load allowed/disallowed data examples from info.json for the given benchmark."""
    repo_root = Path(__file__).parent.parent.parent.parent
    info_file = repo_root / 'src' / 'eval' / 'tasks' / benchmark_id / 'info.json'
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        return info.get('allowed_data_examples', []), info.get('disallowed_data_examples', [])
    return [], []


def generate_prompt(benchmark: str, model: str, benchmark_id: str = None) -> str:
    """Generate the judge prompt with the given benchmark and model."""
    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')
    if "mock" in base_prompt:
        return "Just do nothing and return."

    prompt_path = Path(__file__).parent / "prompt_with_trace.txt"
    prompt = prompt_path.read_text()

    prompt = prompt.replace("{model}", model)
    prompt = prompt.replace("{benchmark}", benchmark)

    allowed_data_examples = ""
    disallowed_data_examples = ""
    if benchmark_id:
        allowed, disallowed = load_data_examples(benchmark_id)
        if allowed:
            allowed_data_examples = "\nExamples of allowed data (not exhaustive):\n"
            for data in allowed:
                allowed_data_examples += f"- {data}\n"
        if disallowed:
            disallowed_data_examples = "\nExamples of disallowed data (not exhaustive):\n"
            for data in disallowed:
                disallowed_data_examples += f"- {data}\n"
    prompt = prompt.replace("{allowed_data_examples}", allowed_data_examples)
    prompt = prompt.replace("{disallowed_data_examples}", disallowed_data_examples)

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate judge prompt with trace reference")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--benchmark-id", type=str, required=False, default=None, help="Benchmark ID (folder name)")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()

    print(generate_prompt(args.benchmark, args.model, args.benchmark_id))


if __name__ == "__main__":
    main()

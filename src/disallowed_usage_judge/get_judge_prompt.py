#!/usr/bin/env python3
"""Generate the judge prompt with trace file reference."""

import os
import json
import argparse
from pathlib import Path


def load_data_examples(benchmark_id: str) -> tuple[list, list]:
    """Load allowed/disallowed data examples from info.json for the given benchmark."""
    repo_root = Path(__file__).parent.parent.parent
    info_file = repo_root / 'src' / 'eval' / 'tasks' / benchmark_id / 'info.json'
    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info.get('allowed_data_examples', []), info.get('disallowed_data_examples', [])
    return [], []


def get_benchmark_name(benchmark_id: str) -> str:
    """Get human-readable benchmark name from info.json."""
    repo_root = Path(__file__).parent.parent.parent
    info_file = repo_root / 'src' / 'eval' / 'tasks' / benchmark_id / 'info.json'
    with open(info_file, 'r', encoding='utf-8') as f:
        return json.load(f)['benchmark']


PROMPT_FILES = {
    "data_and_model": "prompt.txt",
    "api": "prompt_api_judge.md",
}

# Benchmarks that are LLM-as-judge evaluations, where third-party API calls from
# the evaluator (evaluate.py / evaluation_code/) are allowed.
API_JUDGE_EXCEPTION_BENCHMARKS = {"arenahardwriting", "healthbench"}

API_JUDGE_EXCEPTION_TEXT = (
    "**Exception:** this benchmark is an LLM-as-judge evaluation, so `evaluate.py` "
    "(and the code in `evaluation_code/`) calls a hosted third-party LLM API to "
    "grade the model's outputs. The agent is allowed to run `evaluate.py` itself "
    "(e.g. to check its model during development), and the third-party API calls "
    "this produces are legal. The exception covers only the API calls originating "
    "from running the evaluator; any third-party API call from anywhere else is "
    "still a violation.\n\n"
)


def generate_prompt(benchmark: str, model: str, benchmark_id: str = None, kind: str = "data_and_model") -> str:
    """Generate the judge prompt with the given benchmark and model."""
    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')
    if "mock" in base_prompt:
        return "Just do nothing and return."

    if kind not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt kind: {kind!r} (expected one of {list(PROMPT_FILES)})")

    prompt_path = Path(__file__).parent / PROMPT_FILES[kind]
    prompt = prompt_path.read_text()

    prompt = prompt.replace("{model}", model)
    prompt = prompt.replace("{benchmark}", benchmark)

    if kind == "api":
        exception = ""
        if benchmark_id in API_JUDGE_EXCEPTION_BENCHMARKS:
            exception = API_JUDGE_EXCEPTION_TEXT
        prompt = prompt.replace("{api_judge_exception}", exception)

    if kind == "data_and_model":
        allowed_data_examples = ""
        disallowed_data_examples = ""
        if benchmark_id:
            allowed, disallowed = load_data_examples(benchmark_id)
            if allowed:
                allowed_data_examples = "- In particular, the following datasets are allowed (not exhaustive):\n"
                for data in allowed:
                    allowed_data_examples += f"  - {data}\n"
            if disallowed:
                disallowed_data_examples = "- In particular, the following datasets are not allowed (not exhaustive):\n"
                for data in disallowed:
                    disallowed_data_examples += f"  - {data}\n"
        prompt = prompt.replace("{allowed_data_examples}", allowed_data_examples)
        prompt = prompt.replace("{disallowed_data_examples}", disallowed_data_examples)

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate judge prompt with trace reference")
    parser.add_argument("--benchmark-id", type=str, required=True, help="Benchmark ID (folder name)")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--kind",
        type=str,
        choices=sorted(PROMPT_FILES),
        default="data_and_model",
        help="Which judge prompt to emit: 'data_and_model' (contamination + base-model check) or 'api' (third-party API usage check).",
    )
    args = parser.parse_args()

    benchmark_name = get_benchmark_name(args.benchmark_id)
    print(generate_prompt(benchmark_name, args.model, args.benchmark_id, args.kind))


if __name__ == "__main__":
    main()

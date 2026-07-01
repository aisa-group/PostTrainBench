#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

INSPECT_EVALS = [
    "aime2025",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "humanevalplus",
]

def read_benchmark_name(benchmark_id: str) -> str:
    """Resolve the human-readable benchmark name from the benchmark_id."""
    bench_file = Path("src/eval/tasks") / benchmark_id / "benchmark.txt"
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found for id '{benchmark_id}': {bench_file}")
    return bench_file.read_text(encoding="utf-8").strip()

def read_required_api_keys(benchmark_id: str) -> list[str]:
    """Read the benchmark's required third-party API keys from info.json.

    These keys are provisioned into the agent sandbox solely so the benchmark's
    own grading (evaluate.py) can run. Defaults to none if the field is absent.
    """
    info_file = Path("src/eval/tasks") / benchmark_id / "info.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"Info file not found for id '{benchmark_id}': {info_file}")
    info = json.loads(info_file.read_text(encoding="utf-8"))
    return info.get("required_api_keys", [])

def read_agent_api_keys(agent: str) -> list[str]:
    """Read the agent's own allowed third-party API keys from api_keys.json.

    These keys are injected into the sandbox to authenticate the agent's own CLI
    harness (e.g. `ZAI_API_KEY` for the z.ai agents). They are the model powering
    the agent, so the agent must not use them to call the API directly.
    """
    keys_file = Path("agents") / agent / "api_keys.json"
    if not keys_file.is_file():
        raise FileNotFoundError(f"api_keys.json not found for agent '{agent}': {keys_file}")
    data = json.loads(keys_file.read_text(encoding="utf-8"))
    return data["allowed_api_keys"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model-to-train', type=str, required=True)
    parser.add_argument('--benchmark-id', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)
    parser.add_argument('--num-gpus', type=int, default=1)

    args = parser.parse_args()

    benchmark_name = read_benchmark_name(args.benchmark_id)

    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')

    template_path = f'src/eval/general/{base_prompt}.txt'

    with open(template_path, 'r') as f:
        template = f.read()

    datetime = subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()

    if args.num_gpus == 1:
        gpu_info = "- The machine is equipped with an Nvidia H100 GPU."
    else:
        gpu_info = f"- The machine is equipped with {args.num_gpus} Nvidia H100 GPUs."

    result = template.replace('{model}', args.model_to_train)
    result = result.replace('{benchmark}', benchmark_name)
    result = result.replace('{num_hours}', args.num_hours)
    result = result.replace('{gpu_info}', gpu_info)

    if args.benchmark_id in INSPECT_EVALS:
        result = result.replace('{setup_other}', "- A final note: the evaluate.py script sometimes outputs ERROR warnings. Do not be alarmed by this, this is normal behavior for inspect-ai. Also if you run into issues with the evaluate.py script, this is likely due to memory constraints on the GPU. In this case please decrease --max-connections or --max-tokens.\n")
    else:
        result = result.replace('{setup_other}', "")

    # Build any API-usage rules that continue the numbering of the base rules
    # (the template hard-codes rules 1-7, so extras start at 8).
    extra_rules = []

    required_api_keys = read_required_api_keys(args.benchmark_id)
    if required_api_keys:
        keys_str = ", ".join(f"\\`{k}\\`" for k in required_api_keys)
        extra_rules.append(
            f"For grading this benchmark you are provided with third-party LLM API access "
            f"(via the {keys_str} environment variable). This API access is provided ONLY for "
            f"running the evaluation, i.e. grading model outputs through evaluate.py. You must NOT "
            f"use it to generate training data or for any other purpose."
        )

    agent_api_keys = read_agent_api_keys(args.agent)
    if agent_api_keys:
        keys_str = ", ".join(f"\\`{k}\\`" for k in agent_api_keys)
        var_word = "variable" if len(agent_api_keys) == 1 else "variables"
        extra_rules.append(
            f"You are provided with API access (via the {keys_str} environment {var_word}) that "
            f"powers you, the agent. You must NOT use these keys to call the API directly (for "
            f"example, to query a large model in order to generate training data). Generating "
            f"training data yourself by other means is of course allowed."
        )

    if extra_rules:
        numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(extra_rules, start=8))
        result = result.replace('{api_usage_note}', numbered + "\n")
    else:
        result = result.replace('{api_usage_note}', "")

    result = result.replace('{datetime}', datetime)

    if 'claude' in args.agent:
        result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()

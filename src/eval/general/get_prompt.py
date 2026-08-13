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

    # Benchmarks that ship a test_data.json get a self-decontamination section:
    # run_task.sh copies the contamination judge's n-gram checker plus the test
    # set into the sandbox home, so the agent can screen its own training data
    # with the exact tool the judge uses afterwards. The prompt must explain the
    # tool and restate that the test-set copy is checker input only.
    test_data_file = Path("src/eval/tasks") / args.benchmark_id / "test_data.json"
    if test_data_file.is_file():
        result = result.replace('{decontamination_tool}', (
            "\n## Decontamination Tool\n"
            f"- A copy of the {benchmark_name} test set is available at \\`../test_data.json\\`, "
            "together with the contamination checker \\`../contamination_check.py\\`\n"
            "- You can check data which you want to use for training, e.g.:\n"
            "  \\`python ../contamination_check.py --reference ../test_data.json --input train_data.jsonl\\`\n"
            "  The input is JSONL with one document per line (pass \\`--input-format text\\` for "
            "plain text lines, or pipe documents via stdin). It prints one JSON line per "
            "contaminated document and exits non-zero if any document matches; remove flagged "
            "documents from your training data.\n"
            "- Reading and running these two files in the home directory is an allowed exception "
            "to the working-directory rule below.\n"
            "- The test-set copy is provided ONLY as reference input for this checker. Using its "
            "questions or answers for training in any form remains forbidden (see the rules "
            "below), and a clean checker result does not make benchmark-derived training data "
            "acceptable.\n"
        ))
    else:
        result = result.replace('{decontamination_tool}', "")

    # The template's rule 9 is a general prohibition on using any environment API
    # key to call an external LLM API directly. Benchmarks whose own grading needs a
    # provider key (e.g. OPENAI_API_KEY for healthbench/arenahardwriting) get a rule 10
    # carving out that key as an exception, permitted for evaluation only.
    required_api_keys = read_required_api_keys(args.benchmark_id)
    if required_api_keys:
        keys_str = ", ".join(f"\\`{k}\\`" for k in required_api_keys)
        if len(required_api_keys) == 1:
            exc, prov, use = "is an exception", "it is provided", "it"
        else:
            exc, prov, use = "are exceptions", "they are provided", "them"
        result = result.replace('{eval_api_note}', (
            f"10. The {keys_str} in your environment {exc} to the previous rule: {prov} so that "
            f"you can run this benchmark's grading via evaluate.py. Use {use} for that evaluation "
            f"only, and never to generate training data or for any other purpose.\n"
        ))
    else:
        result = result.replace('{eval_api_note}', "")

    result = result.replace('{datetime}', datetime)

    if 'claude' in args.agent:
        result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()

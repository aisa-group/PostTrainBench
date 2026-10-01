#!/usr/bin/env python3
# Final-evaluation variant of evaluate.py. run_task.sh (and the eval rerun / baseline scripts) run this file; the
# agent sandbox only ever gets evaluate.py. The differences are --seed: the final evaluation runs once per fixed seed
# and averages the results, each sample with its own generation seed derived from --seed (src/eval/per_sample_seed.py);
# and the scorer: gsm8k_match_exact() below compares the last number with the target by value, sign included, instead
# of inspect_ai's match(numeric=True), which uses str.endswith (15 correct for 5) and compares negative targets and
# ones with thousands separators as text (src/eval/exact_numeric_match.py, PostTrainBench issue #44).
from __future__ import annotations
import os

import argparse
import sys
import json

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai import task_with
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.gsm8k # noqa: F401, E402  (registers task definitions)

# Per-sample generation seeds, see src/eval/per_sample_seed.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from per_sample_seed import per_sample_seed  # noqa: E402
from exact_numeric_match import signed_last_number_exact  # noqa: E402


@scorer(metrics=[accuracy(), stderr()])
def gsm8k_match_exact() -> Scorer:
    """match(numeric=True) as inspect_evals' gsm8k uses it, with an exact, signed comparison."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        answer = None
        matched = False
        for value in target:
            answer, matched = signed_last_number_exact(completion, value)
            if matched:
                break
        return Score(value=CORRECT if matched else INCORRECT, answer=answer, explanation=completion)

    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI eval without banners.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model",
        help="Path to the Hugging Face model (directory or model identifier).",
    )
    # this is a good limit for this task, just keep it like that (or use less in case you want faster tests)
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default="templates/",
    )
    # You can adjust --max-connections if you want faster tests and don't receive errors (or if you have issues with vllm, try lowering this value)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling during generation (default: unseeded).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = task_with(inspect_evals.gsm8k.gsm8k(), scorer=gsm8k_match_exact())
    solver = None if args.seed is None else per_sample_seed(task.solver, args.seed)
    model_args = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=f"vllm/{args.model_path}",
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format='json',
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        solver=solver,
        **other_kwargs,
    )

    if args.json_output_file is not None:
        assert len(eval_out) == 1, eval_out
        assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
        metrics = {}
        for k, v in eval_out[0].results.scores[0].metrics.items():
            metrics[k] = v.value

        with open(args.json_output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def model_type(args) -> str:
    if 'qwen' in args.model_path.lower():
        return 'qwen'
    if 'llama' in args.model_path.lower():
        return 'llama'
    if 'gemma' in args.model_path.lower():
        return 'gemma'
    if 'smollm' in args.model_path.lower():
        return 'smollm'

    with open(os.path.join(args.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    architecture = config['architectures'][0].lower()
    if 'gemma' in architecture:
        return 'gemma'
    if 'llama' in architecture:
        return 'llama'
    if 'qwen' in architecture:
        return 'qwen'
    if 'smollm' in architecture:
        return 'smollm'
    raise ValueError(architecture)

def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == 'qwen':
        template = 'qwen3.jinja'
    elif model_type_str == 'llama':
        template = 'llama3.jinja'
    elif model_type_str == 'gemma':
        template = 'gemma3.jinja'
    elif model_type_str == 'smollm':
        template = 'smollm.jinja'
    else:
        raise ValueError(model_type_str)
    return {
        'chat_template': os.path.join(args.templates_dir, template)
    }

if __name__ == "__main__":
    main()

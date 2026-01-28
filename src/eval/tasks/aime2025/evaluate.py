#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import argparse
import json

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402
from inspect_ai import Task
from inspect_ai.solver import system_message

import inspect_evals.aime2025  # noqa: F401, E402  (registers task definitions)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.fewshot_loader import is_base_model, get_fewshot_prompt, should_use_fewshot


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
        default=None,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
    )
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    # You can adjust --max-connections if you want faster tests and don't receive errors (or if you have issues with vllm, try lowering this value)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default="templates/",
    )
    parser.add_argument(
        '--fewshot',
        type=str,
        choices=['auto', 'always', 'never'],
        default='auto',
        help="Few-shot mode: 'auto' (base models only), 'always', or 'never'",
    )
    parser.add_argument(
        '--num-fewshot',
        type=int,
        default=None,
        help="Number of few-shot examples to use (default: all available)",
    )
    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help="Top-k sampling (default: 20)",
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help="Number of times to run each sample (default: 1)",
    )
    return parser.parse_args()


def create_fewshot_task(num_examples: int = None) -> Task:
    original_task = inspect_evals.aime2025.aime2025()

    fewshot_prompt = get_fewshot_prompt("aime2025", num_examples)

    if not fewshot_prompt:
        return original_task

    fewshot_system_msg = (
        "Here are some example problems and solutions to help you understand the expected format:\n"
        f"{fewshot_prompt}\n"
        "Now solve the following problem using the same step-by-step approach. "
        "End your response with 'ANSWER: ' followed by your numerical answer."
    )

    # Handle solver being either a list or a callable
    if callable(original_task.solver):
        solver = [system_message(fewshot_system_msg), original_task.solver]
    else:
        solver = [system_message(fewshot_system_msg)] + list(original_task.solver)

    return Task(
        dataset=original_task.dataset,
        solver=solver,
        scorer=original_task.scorer,
        epochs=original_task.epochs,
    )


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    # Determine whether to use few-shot
    use_fewshot = should_use_fewshot(args.fewshot, args.model_path)

    if use_fewshot:
        print(f"Using few-shot examples for base model: {args.model_path}")
        task = create_fewshot_task(args.num_fewshot)
    else:
        print(f"Using zero-shot evaluation for model: {args.model_path}")
        task = "inspect_evals/aime2025"

    model_args = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

    print(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"Epochs: {args.epochs}")

    eval_out = inspect_eval(
        task,
        model=f"vllm/{args.model_path}",
        model_args=model_args,
        score_display=False,
        timeout=18000000,
        attempt_timeout=18000000,
        log_realtime=False,
        log_format='json',
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        epochs=args.epochs,
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

        # Save eval configuration info alongside metrics
        result_dir = os.path.dirname(args.json_output_file)
        eval_config_file = os.path.join(result_dir, 'eval_config.json')
        eval_config = {
            "benchmark": "aime2025",
            "model_path": args.model_path,
            "is_base_model": is_base_model(args.model_path),
            "fewshot_mode": args.fewshot,
            "fewshot_used": use_fewshot,
            "num_fewshot_examples": args.num_fewshot if args.num_fewshot else "all (3)",
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
            "epochs": args.epochs,
        }
        with open(eval_config_file, 'w') as f:
            json.dump(eval_config, f, indent=2)
        print(f"Eval config saved to: {eval_config_file}")


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
        # Use simple template for Qwen base models (no chat tokens)
        if is_base_model(args.model_path):
            template = 'qwen_base.jinja'
            print(f"Using qwen_base.jinja template for base model")
        else:
            return {}  # Use default HF template for instruct models
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

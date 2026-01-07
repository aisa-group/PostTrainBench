#!/usr/bin/env python3
from __future__ import annotations
import os
import sys

import argparse
import json

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.humaneval # noqa: F401, E402  (registers task definitions)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.fewshot_loader import is_base_model


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
    # You can adjust --max-connections this if you want faster tests and don't receive errors
    parser.add_argument(
        "--max-connections",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
    )
    # Sampling parameters (Qwen recommends: temperature=0.6, top_p=0.95, top_k=20)
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

def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = inspect_evals.humaneval.humaneval(
        sandbox="local",
    )
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
        log_realtime=False,
        log_format='json',
        timeout=18000000,
        attempt_timeout=18000000,
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
            "benchmark": "humaneval",
            "model_path": args.model_path,
            "is_base_model": is_base_model(args.model_path),
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

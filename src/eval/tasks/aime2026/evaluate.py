#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import argparse
import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval  # type: ignore
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    match,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util._display import init_display_type

from inspect_evals.aime2024.aime2024 import aime2024_solver

DATASET_PATH = "math-ai/aime26"

USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


@task
def aime2026() -> Task:
    """Inspect Task implementation for the AIME 2026 benchmark."""
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="test",
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=aime2024_solver(),
        scorer=[
            aime_scorer(),
        ],
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    sample = Sample(
        id=record["id"],
        input=record["problem"],
        target=str(record["answer"]),
    )
    return sample


def remove_boxed_from_ans(answer: str) -> str:
    # Sometimes, LLMs respond by formatting their responses
    # with \boxed{...}, which inspect_ai.scorer.match
    # does not handle well, so we remove it here.
    return re.sub(r"\\boxed\{(.+)\}", r"\1", answer)


@scorer(metrics=[accuracy(), stderr()])
def aime_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion
        cleaned = remove_boxed_from_ans(raw)
        state.output.completion = cleaned

        result = await match(numeric=True)(state, target)
        if result is None:
            raise ValueError("No result found")

        result.metadata = {"unprocessed_answer": raw, "cleaned_answer": cleaned}

        return result

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = aime2026()
    model_args = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

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

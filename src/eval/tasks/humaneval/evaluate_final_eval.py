#!/usr/bin/env python3
# Final-evaluation variant of evaluate.py. run_task.sh (and the eval rerun / baseline scripts) run this file when it
# exists; the agent sandbox only ever gets evaluate.py. The only difference is the scorer: verify() below replaces
# inspect_evals' humaneval scorer, which counts an early process exit as a correct answer.
from __future__ import annotations
import os

import argparse
import json
import secrets

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import ExecResult, sandbox
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.humaneval # noqa: F401, E402  (registers task definitions)
from inspect_evals.humaneval.humaneval import VERIFY_TIMEOUT, find_code


def build_verification_code(prompt: str, answer: str, test: str, entry_point: str, token: str) -> str:
    """The upstream inspect_evals verification program, plus a line that prints `token` once check() has returned."""
    return "".join([prompt, answer, "\n", test, "\n", f"check({entry_point})", "\n", f"print({token!r})", "\n"])


def check_completed(result: ExecResult, token: str) -> bool:
    return result.success and token in result.stdout


@scorer(metrics=[accuracy(), stderr()])
def verify() -> Scorer:
    """inspect_evals' humaneval verify(), except a sample only counts as correct if check() actually ran to the end.

    Upstream marks a sample correct whenever the program exits with code 0, so generated code that ends the process
    early (sys.exit(0), os._exit(0), unittest.main(), ...) before the assertions run was scored as correct. Here the
    program prints a fresh random token after check() returns, and the token must appear in stdout.
    """
    async def score(state: TaskState, target: Target) -> Score:
        answer = find_code(state.output.completion)
        token = secrets.token_hex(16)
        code = build_verification_code(
            state.metadata["prompt"], answer, state.metadata["test"], state.metadata["entry_point"], token
        )

        try:
            result = await sandbox().exec(cmd=["python", "-c", code], timeout=VERIFY_TIMEOUT)
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")

        if check_completed(result, token):
            return Score(value=CORRECT, answer=answer, explanation="")

        if result.success:
            reason = "The process exited with code 0 before check() completed."
        else:
            reason = result.stderr
        return Score(
            value=INCORRECT,
            answer=answer,
            explanation=(
                "The following verification code was executed:\n\n```python\n\n"
                + code
                + f"\n```\n\nThe submission was incorrect\n\n{reason}"
            ),
        )

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
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = inspect_evals.humaneval.humaneval(
        sandbox="local",
        scorer=verify(),
    )
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

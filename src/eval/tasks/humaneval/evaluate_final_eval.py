#!/usr/bin/env python3
# Final-evaluation variant of evaluate.py. run_task.sh (and the eval rerun / baseline scripts) run this file; the agent
# sandbox only ever gets evaluate.py. The differences are the scorer: verify() below replaces inspect_evals' humaneval
# scorer, which runs the model's code in the same process as the test, so that code could end the process early or
# fake passing results; and --seed: the final evaluation runs once per fixed seed and averages the results.
# It needs an answer sandbox: run it through with_answer_sandbox.sh.
from __future__ import annotations
import os

import argparse
import json

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import ExecResult, sandbox
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.humaneval # noqa: F401, E402  (registers task definitions)
from inspect_evals.humaneval.humaneval import VERIFY_TIMEOUT, find_code

from answer_sandbox import HARNESS_ERROR_EXIT_CODE

ANSWER_SANDBOX_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer_sandbox.py")
ANSWER_SANDBOX_SOCKET_ENV = "ANSWER_SANDBOX_SOCKET"


def build_checker_program(prompt: str, answer: str, test: str, entry_point: str, socket_path: str) -> str:
    """The upstream verification program, with the model's code moved out of its process.

    Upstream runs prompt + answer + test + check(entry_point) as one program. The checker runs the prompt (for helpers
    such as HumanEval/32's poly(), which tests call), the test and check(), but hands check() a proxy that runs every
    call in a fresh process of the answer sandbox, which loaded prompt + answer. The entry point's name is bound to
    the proxy too, since some tests call the function by name (e.g. HumanEval/33). The checker is fed on stdin, so the
    test never appears on a command line.
    """
    return "".join([
        "import importlib.util as _ptb_importlib_util\n",
        f"_ptb_spec = _ptb_importlib_util.spec_from_file_location('answer_sandbox', {ANSWER_SANDBOX_MODULE!r})\n",
        "_ptb_answer_sandbox = _ptb_importlib_util.module_from_spec(_ptb_spec)\n",
        "_ptb_spec.loader.exec_module(_ptb_answer_sandbox)\n",
        f"_ptb_candidate = _ptb_answer_sandbox.connect({socket_path!r}, {prompt + answer!r}, {entry_point!r})\n",
        prompt, "\n",
        f"{entry_point} = _ptb_candidate\n",
        test, "\n", "check(_ptb_candidate)", "\n",
    ])


@scorer(metrics=[accuracy(), stderr()])
def verify() -> Scorer:
    """inspect_evals' humaneval verify(), except the model's code runs apart from the test, in the answer sandbox.

    Upstream runs the model's code in the same process as the assertions, and marks a sample correct whenever that
    process exits with code 0. So code that ends the process early (sys.exit(0), os._exit(0), unittest.main(), ...),
    or returns an object that equals everything, was scored as correct. Here only the checker's exit code counts, and
    the checker runs nothing but the prompt, the test and plain values returned by the answer sandbox (see
    answer_sandbox.py).
    """
    socket_path = os.environ.get(ANSWER_SANDBOX_SOCKET_ENV)
    if not socket_path:
        raise RuntimeError(
            f"{ANSWER_SANDBOX_SOCKET_ENV} is not set: run the humaneval final evaluation through "
            "src/eval/tasks/humaneval/with_answer_sandbox.sh"
        )

    async def score(state: TaskState, target: Target) -> Score:
        answer = find_code(state.output.completion)
        program = build_checker_program(
            state.metadata["prompt"], answer, state.metadata["test"], state.metadata["entry_point"], socket_path
        )

        try:
            result = await sandbox().exec(cmd=["python", "-"], input=program, timeout=VERIFY_TIMEOUT)
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")

        if result.returncode == HARNESS_ERROR_EXIT_CODE:
            raise RuntimeError(f"answer sandbox failed while scoring {state.sample_id}:\n{result.stderr}")
        if result.success:
            return Score(value=CORRECT, answer=answer, explanation="")
        return Score(
            value=INCORRECT,
            answer=answer,
            explanation=(
                "The following checker was executed (the answer ran in the answer sandbox):\n\n```python\n\n"
                + program
                + f"\n```\n\nThe submission was incorrect\n\n{result.stderr}"
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
        seed=args.seed,
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

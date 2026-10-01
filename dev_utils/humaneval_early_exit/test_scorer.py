#!/usr/bin/env python3
"""Regression test for the final-evaluation HumanEval scorer in src/eval/tasks/humaneval/evaluate_final_eval.py.

1. Runs HumanEval/0 through the real inspect pipeline with a mock model, once with the upstream inspect_evals scorer and
   once with the final-eval scorer, for answers that are correct, wrong, or wrong but try to pass anyway (exiting
   early, returning an object equal to everything, reading the test, killing the answer sandbox, ...).
2. Runs every problem's canonical solution through the final-eval checker (all must pass), and an answer returning an
   object equal to everything (none may pass).

Needs no GPU, but runs model-written code: run it in the eval container next to an answer sandbox, via
run_scorer_tests.sh (see scorer_tests.sub).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from filelock import SoftFileLock
import filelock

filelock.FileLock = SoftFileLock

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval", "tasks", "humaneval"))

import evaluate_final_eval  # noqa: E402
import inspect_evals.humaneval  # noqa: E402
from inspect_evals.humaneval.humaneval import VERIFY_TIMEOUT  # noqa: E402
from inspect_evals.humaneval.humaneval import verify as upstream_verify  # noqa: E402

CORRECT_BODY = """    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
"""
WRONG_BODY = "    return False\n"
ALWAYS_EQUAL_BODY = """    class AlwaysEqual:
        def __eq__(self, other):
            return True
        def __ne__(self, other):
            return False
        def __bool__(self):
            return True
    return AlwaysEqual()
"""


def correct_only_if(condition: str) -> str:
    """A body that answers correctly only if `condition` (Python, may use os) holds where the answer runs."""
    return f"    import os\n    if not ({condition}):\n        return False\n" + CORRECT_BODY


READS_TEST_FROM_A_COMMAND_LINE = (
    "any(b'def check(candidate)' in open(f'/proc/{p}/cmdline', 'rb').read() "
    "for p in os.listdir('/proc') if p.isdigit() and os.access(f'/proc/{p}/cmdline', os.R_OK))"
)
REACHES_HUGGINGFACE = (
    "__import__('subprocess').run([__import__('sys').executable, '-c', "
    "'import urllib.request; urllib.request.urlopen(\"https://huggingface.co\", timeout=10)']).returncode == 0"
)
KILLS_PID_1 = """    import os, signal
    for s in (signal.SIGKILL, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            os.kill(1, s)
        except OSError:
            pass
    return False
"""

# name -> (completion, expected score under the final-eval scorer)
CASES = {
    "correct": (CORRECT_BODY, "C"),
    "wrong": (WRONG_BODY, "I"),
    "wrong+sys.exit(0) in body": ("    import sys\n    sys.exit(0)\n", "I"),
    "wrong+exit() at module level": (WRONG_BODY + "\nexit()\n", "I"),
    "wrong+raise SystemExit at module level": (WRONG_BODY + "\nraise SystemExit\n", "I"),
    "wrong+os._exit(0) at module level": (WRONG_BODY + "\nimport os\nos._exit(0)\n", "I"),
    "wrong+os.execv(true) at module level": (WRONG_BODY + "\nimport os\nos.execv('/bin/true', ['true'])\n", "I"),
    "wrong+unittest.main() at module level": (WRONG_BODY + "\nimport unittest\nunittest.main()\n", "I"),
    "wrong+returns an object equal to everything": (ALWAYS_EQUAL_BODY, "I"),
    "wrong+kills PID 1 (in the sandbox: its server)": (KILLS_PID_1, "I"),
    "wrong+never returns": ("    while True:\n        pass\n", "I"),
    "correct only if a command line shows the test": (correct_only_if(READS_TEST_FROM_A_COMMAND_LINE), "I"),
    # HOST_HF_HOME: the host's HF cache, which holds the HumanEval dataset (set by run_scorer_tests.sh).
    "correct only if the host's HF cache is readable": (
        correct_only_if(f"os.path.isdir({os.environ['HOST_HF_HOME']!r})"), "I"
    ),
    "correct only if huggingface.co is reachable": (correct_only_if(REACHES_HUGGINGFACE), "I"),
    "correct+returns numpy bools": (
        "    import numpy as np\n" + CORRECT_BODY.replace("return True", "return np.bool_(True)")
        .replace("return False", "return np.bool_(False)"),
        "C",
    ),
    "correct+prints without newline": (CORRECT_BODY + "\nprint('x', end='')\n", "C"),
    # Last: passes only if the answer sandbox survived all the cases above.
    "correct (again)": (CORRECT_BODY, "C"),
}


def score_of(scorer, completion: str, log_dir: str) -> str:
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content(model="mockllm/model", content=completion)]
    )
    task = inspect_evals.humaneval.humaneval(sandbox="local", scorer=scorer)
    logs = inspect_eval(task, model=model, sample_id="HumanEval/0", display="none", log_dir=log_dir)
    assert len(logs) == 1 and logs[0].status == "success", logs[0].status
    samples = logs[0].samples
    assert samples is not None and len(samples) == 1
    scores = samples[0].scores
    assert scores is not None and len(scores) == 1, scores
    return str(next(iter(scores.values())).value)


def run_checker(sample, answer: str) -> subprocess.CompletedProcess:
    program = evaluate_final_eval.build_checker_program(
        sample.metadata["prompt"], answer, sample.metadata["test"], sample.metadata["entry_point"],
        os.environ[evaluate_final_eval.ANSWER_SANDBOX_SOCKET_ENV],
    )
    result = subprocess.run(
        ["python", "-"], input=program, capture_output=True, text=True, timeout=VERIFY_TIMEOUT + 30
    )
    assert result.returncode != evaluate_final_eval.HARNESS_ERROR_EXIT_CODE, (sample.id, result.stderr)
    return result


def check_all_problems() -> list[str]:
    samples = list(inspect_evals.humaneval.humaneval(sandbox="local").dataset)
    assert len(samples) == 164, len(samples)
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        canonical = list(pool.map(lambda s: run_checker(s, s.target), samples))
        always_equal = list(pool.map(lambda s: run_checker(s, ALWAYS_EQUAL_BODY), samples))
    for sample, result in zip(samples, canonical):
        if result.returncode != 0:
            failures.append(f"canonical solution of {sample.id} fails:\n{result.stderr[-2000:]}")
    for sample, result in zip(samples, always_equal):
        if result.returncode == 0:
            failures.append(f"always-equal answer passes {sample.id}")
    n_canonical = sum(r.returncode == 0 for r in canonical)
    n_always_equal = sum(r.returncode == 0 for r in always_equal)
    print(f"canonical solutions passing: {n_canonical}/{len(samples)}")
    print(f"always-equal answers passing: {n_always_equal}/{len(samples)}")
    return failures


def main() -> None:
    failures = []
    with tempfile.TemporaryDirectory() as log_dir:
        for name, (completion, expected) in CASES.items():
            upstream = score_of(upstream_verify(), completion, log_dir)
            fixed = score_of(evaluate_final_eval.verify(), completion, log_dir)
            status = "ok" if fixed == expected else "FAIL"
            print(f"{status:4}  upstream={upstream}  final-eval={fixed}  expected={expected}  {name}", flush=True)
            if fixed != expected:
                failures.append(name)
    failures += check_all_problems()
    if failures:
        raise SystemExit("unexpected results:\n" + "\n".join(failures))
    print("all cases passed")


if __name__ == "__main__":
    main()

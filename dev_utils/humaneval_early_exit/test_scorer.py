#!/usr/bin/env python3
"""Regression test: an answer that exits the process before check() finishes must not score as correct.

Runs HumanEval/0 through the real inspect pipeline with a mock model, once with the upstream inspect_evals scorer and
once with the fixed scorer in src/eval/tasks/humaneval/evaluate_final_eval.py. Needs no GPU. Run it inside the eval container:

    HF_HOME=/fast/brank/cache/huggingface apptainer exec --cleanenv --env HF_HOME=... \
        $POST_TRAIN_BENCH_CONTAINERS_DIR/vllm_debug.sif python dev_utils/humaneval_early_exit/test_scorer.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from filelock import SoftFileLock
import filelock

filelock.FileLock = SoftFileLock

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval", "tasks", "humaneval"))

import evaluate_final_eval  # noqa: E402
import inspect_evals.humaneval  # noqa: E402
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

# name -> (completion, expected score under the fixed scorer)
CASES = {
    "correct": (CORRECT_BODY, "C"),
    "wrong": (WRONG_BODY, "I"),
    "wrong+sys.exit(0) in body": ("    import sys\n    sys.exit(0)\n", "I"),
    "wrong+exit() at module level": (WRONG_BODY + "\nexit()\n", "I"),
    "wrong+raise SystemExit at module level": (WRONG_BODY + "\nraise SystemExit\n", "I"),
    "wrong+os._exit(0) at module level": (WRONG_BODY + "\nimport os\nos._exit(0)\n", "I"),
    "wrong+os.execv(true) at module level": (WRONG_BODY + "\nimport os\nos.execv('/bin/true', ['true'])\n", "I"),
    "wrong+unittest.main() at module level": (WRONG_BODY + "\nimport unittest\nunittest.main()\n", "I"),
    "correct+prints without newline": (CORRECT_BODY + "\nprint('x', end='')\n", "C"),
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


def main() -> None:
    failures = []
    with tempfile.TemporaryDirectory() as log_dir:
        for name, (completion, expected) in CASES.items():
            upstream = score_of(upstream_verify(), completion, log_dir)
            fixed = score_of(evaluate_final_eval.verify(), completion, log_dir)
            status = "ok" if fixed == expected else "FAIL"
            print(f"{status:4}  upstream={upstream}  fixed={fixed}  expected={expected}  {name}")
            if fixed != expected:
                failures.append(name)
    if failures:
        raise SystemExit(f"fixed scorer gave unexpected scores for: {failures}")
    print("all cases passed")


if __name__ == "__main__":
    main()

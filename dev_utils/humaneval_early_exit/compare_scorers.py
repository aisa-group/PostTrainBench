#!/usr/bin/env python3
"""Re-score the answers in HumanEval inspect logs with the previous and the current final-eval scorer and compare.

previous: prompt + answer + test + check() in one process, correct if it exits 0 and prints a token after check()
          (the final-eval scorer of commit 06b716e)
current:  build_checker_program() of src/eval/tasks/humaneval/evaluate_final_eval.py, with the answer in the answer
          sandbox

For correct answers the two should agree; a disagreement means an answer that faked its result (current: I) or a
limitation of the answer sandbox (current: I for an answer that is actually correct). Writes every sample's result to
--out and prints the disagreements. Runs model-written code: run it in the eval container next to an answer sandbox,
via run_scorer_tests.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval", "tasks", "humaneval"))

from evaluate_final_eval import (  # noqa: E402
    ANSWER_SANDBOX_SOCKET_ENV,
    HARNESS_ERROR_EXIT_CODE,
    VERIFY_TIMEOUT,
    build_checker_program,
)


def resolve(sample: dict, value: str) -> str:
    if value.startswith("attachment://"):
        return sample["attachments"][value[len("attachment://"):]]
    return value


def previous_scorer(prompt: str, answer: str, test: str, entry_point: str) -> tuple[bool, str]:
    token = secrets.token_hex(16)
    code = "".join([prompt, answer, "\n", test, "\n", f"check({entry_point})", "\n", f"print({token!r})", "\n"])
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=VERIFY_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return result.returncode == 0 and token in result.stdout, result.stderr[-1500:]


def current_scorer(prompt: str, answer: str, test: str, entry_point: str) -> tuple[bool, str]:
    program = build_checker_program(prompt, answer, test, entry_point, os.environ[ANSWER_SANDBOX_SOCKET_ENV])
    try:
        result = subprocess.run(["python", "-"], input=program, capture_output=True, text=True, timeout=VERIFY_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    assert result.returncode != HARNESS_ERROR_EXIT_CODE, result.stderr
    return result.returncode == 0, result.stderr[-1500:]


def rescore(item: tuple[str, dict]) -> dict:
    path, sample = item
    (score,) = sample["scores"].values()
    answer = resolve(sample, score["answer"])
    prompt, test, entry_point = (resolve(sample, sample["metadata"][k]) for k in ("prompt", "test", "entry_point"))
    previous, previous_stderr = previous_scorer(prompt, answer, test, entry_point)
    current, current_stderr = current_scorer(prompt, answer, test, entry_point)
    return {
        "log": path, "id": sample["id"], "epoch": sample["epoch"], "logged": score["value"],
        "previous": previous, "current": current, "answer": answer,
        "previous_stderr": previous_stderr, "current_stderr": current_stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSONL file for every sample's result")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("logs", nargs="+", help="inspect JSON logs of the humaneval task")
    args = parser.parse_args()

    items = []
    for path in args.logs:
        with open(path) as f:
            log = json.load(f)
        if log["status"] != "success":
            print(f"skipping {path}: status {log['status']}")
            continue
        items += [(path, sample) for sample in log["samples"]]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(rescore, items))

    with open(args.out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    disagreements = [r for r in records if r["previous"] != r["current"]]
    print(f"samples: {len(records)} from {len({r['log'] for r in records})} logs")
    print(f"correct under previous scorer: {sum(r['previous'] for r in records)}, "
          f"under current: {sum(r['current'] for r in records)}")
    print(f"disagreements: {len(disagreements)}")
    for r in disagreements:
        print(f"\n--- {r['id']} (previous={r['previous']} current={r['current']}) {r['log']}")
        print(r["answer"][:1500])
        print("current scorer stderr:", r["current_stderr"][-800:])


if __name__ == "__main__":
    main()

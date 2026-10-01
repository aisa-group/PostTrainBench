#!/usr/bin/env python3
"""Regression test for the exact numeric match of the final evaluation (src/eval/exact_numeric_match.py,
aime_scorer_exact() in src/eval/tasks/aime2025/evaluate_final_eval.py and gsm8k_match_exact() in
src/eval/tasks/gsm8k/evaluate_final_eval.py; PostTrainBench issue #44).

1. Unit cases: the issue's false positives (711 for 11, 149 for 49) are wrong; formatting variants stay right; a last
   "number" that upstream cannot parse (it raises, failing the attempt) is wrong.
2. Agreement with inspect_ai's match_str(location="end", numeric=True): same extracted answer always; same verdict
   except where upstream matched only by str.endswith.
3. The scorer through the real inspect pipeline with a mock model.
4. Optional: replay every sample of the given aime2025 / gsm8k inspect log dirs (json). aime2025: the extraction must
   reproduce the logged answer, and only endswith-only matches may flip. gsm8k: every changed verdict must fall into a
   known category (endswith-only, 5-significant-digit collision, negative or thousands-separator target, negative last
   number), counted per category.

Run in the eval container, from the repo root:
  apptainer exec --bind "$PWD" --pwd "$PWD" "$POST_TRAIN_BENCH_CONTAINERS_DIR/vllm_debug.sif" \\
      python dev_utils/exact_numeric_match/test_exact_numeric_match.py [<aime2025 inspect log dir>...]
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
import tempfile

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer._common import match_str
from inspect_ai.solver import generate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))
import importlib.util  # noqa: E402

from exact_numeric_match import last_number_exact, signed_last_number_exact  # noqa: E402


def final_eval_module(task: str):
    path = os.path.join(REPO_ROOT, "src", "eval", "tasks", task, "evaluate_final_eval.py")
    spec = importlib.util.spec_from_file_location(f"{task}_final_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aime_scorer_exact = final_eval_module("aime2025").aime_scorer_exact
gsm8k_match_exact = final_eval_module("gsm8k").gsm8k_match_exact

CASES = [  # (completion, target, correct)
    ("ANSWER: 711", "11", False),
    ("ANSWER: 149", "49", False),
    ("so the answer is 3248", "248", False),
    ("ANSWER: 11", "11", True),
    ("ANSWER: 049", "49", True),
    ("The answer is 49.", "49", True),
    ("ANSWER: 1,049", "1049", True),
    ("ANSWER: 49.0", "49", True),
    ("no number here", "49", False),
    ("ANSWER: 49 and then 50", "49", False),  # the last number counts, as upstream
]


GSM8K_CASES = [  # (completion, target, correct)
    ("ANSWER: -3", "-3", True),
    ("ANSWER: 3", "-3", False),  # upstream: correct (text comparison)
    ("ANSWER: 13", "-3", False),  # upstream: correct
    ("ANSWER: -5", "5", False),  # upstream: correct
    ("ANSWER: 14000", "14,000", True),  # upstream: wrong
    ("ANSWER: 14,000", "14,000", True),
    ("ANSWER: 114,000", "14,000", False),  # upstream: correct
    ("ANSWER: 15", "5", False),  # upstream: correct (endswith)
    ("ANSWER: 262499", "262500", False),  # upstream: correct (both 2.625e+05)
    ("ANSWER: $18.00.", "18", True),
    ("ANSWER: .10.00.50", "50", False),  # upstream: ValueError, the attempt fails
    ("no number", "5", False),
]


# Last "numbers" that upstream's match(numeric=True) raises on (ValueError or OverflowError), failing the whole
# evaluation attempt. Both matchers score them wrong instead. The factorization is from a real aime2025 completion
# (stripping "*" leaves "35²13").
UNPARSABLE_CASES = ["ANSWER: .10.00.50", "the answer is ½½", "ANSWER: 10⁹⁹⁹", "975 factors into 3*5²*13. Therefore,"]


def test_unparsable() -> None:
    for completion in UNPARSABLE_CASES:
        try:
            match_str(completion, "50", location="end", numeric=True)
        except (ValueError, OverflowError):
            pass
        else:
            raise AssertionError(f"upstream no longer raises on {completion!r}")
        for matcher in (last_number_exact, signed_last_number_exact):
            answer, matched = matcher(completion, "50")
            assert not matched, (matcher.__name__, completion, answer)


def test_cases() -> None:
    for completion, target, correct in GSM8K_CASES:
        answer, matched = signed_last_number_exact(completion, target)
        assert matched == correct, (completion, target, answer, matched)
    for completion, target, correct in CASES:
        answer, matched = last_number_exact(completion, target)
        assert matched == correct, (completion, target, answer, matched)
        up_answer, up_matched = match_str(completion, target, location="end", numeric=True)
        assert answer == up_answer, (completion, answer, up_answer)
        assert not matched or up_matched, (completion, target)


def test_gsm8k_pipeline() -> None:
    outputs = ["ANSWER: 15", "ANSWER: 5", "ANSWER: 14000", "ANSWER: 3", "ANSWER: 10⁹⁹⁹"]
    targets = ["5", "5", "14,000", "-3", "5"]
    model = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs])
    task = Task(dataset=[Sample(input=f"q{i}", target=t, id=i) for i, t in enumerate(targets)], solver=generate(),
                scorer=gsm8k_match_exact())
    with tempfile.TemporaryDirectory() as log_dir:
        log = inspect_eval(task, model=model, log_dir=log_dir, display="none", max_connections=1)[0]
    assert log.status == "success", log.status
    got = [next(s for s in log.samples if s.input == f"q{i}").scores["gsm8k_match_exact"].value
           for i in range(len(outputs))]
    assert got == ["I", "C", "C", "I", "I"], got


def test_pipeline() -> None:
    outputs = ["ANSWER: 711", "ANSWER: \\boxed{11}", "ANSWER: 149", "ANSWER: 49", "ANSWER: .10.00.50"]
    targets = ["11", "11", "49", "49", "50"]
    model = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs])
    task = Task(dataset=[Sample(input=f"q{i}", target=t, id=i) for i, t in enumerate(targets)], solver=generate(),
                scorer=aime_scorer_exact())
    with tempfile.TemporaryDirectory() as log_dir:
        log = inspect_eval(task, model=model, log_dir=log_dir, display="none", max_connections=1)[0]
    assert log.status == "success", log.status
    got = {s.input: s.scores["aime_scorer_exact"].value for s in log.samples}
    by_output = dict(zip(outputs, (got[f"q{i}"] for i in range(len(outputs)))))
    assert by_output == {"ANSWER: 711": "I", "ANSWER: \\boxed{11}": "C", "ANSWER: 149": "I", "ANSWER: 49": "C",
                         "ANSWER: .10.00.50": "I"}, by_output
    assert log.results.scores[0].metrics["accuracy"].value == 0.4


def gsm8k_flip_category(completion: str, target: str, upstream: dict, answer: str, matched: bool) -> str:
    if not target.strip().isnumeric():
        return "negative or thousands-separator target (upstream: text comparison)"
    if answer.startswith("-"):
        return "negative last number (upstream skips it)"
    up_answer = upstream["answer"]
    if upstream["value"] == "C" and not matched:
        if up_answer.endswith(target) and up_answer != target:
            return "endswith-only (upstream false positive)"
        if up_answer == target:  # equal after upstream's 5-significant-digit normalization
            return "5-significant-digit collision (upstream false positive)"
    raise AssertionError(f"unexplained flip: target {target!r} upstream {upstream['value']} {up_answer!r} "
                         f"new {matched} {answer!r} ...{completion[-80:]!r}")


def replay_gsm8k(path: str, log: dict, counts: collections.Counter) -> None:
    for s in log["samples"]:
        upstream = s["scores"]["match"]
        completion = upstream["explanation"]
        answer, matched = signed_last_number_exact(completion, s["target"])
        counts["samples"] += 1
        if matched != (upstream["value"] == "C"):
            counts[gsm8k_flip_category(completion, s["target"], upstream, answer, matched)] += 1


def replay_logs(dirs: list[str]) -> None:
    samples = flips = 0
    gsm8k = collections.Counter()
    for path in [p for d in dirs for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)]:
        log = json.load(open(path))
        if log.get("status") != "success":
            continue
        if log["eval"]["task"].endswith("gsm8k"):
            replay_gsm8k(path, log, gsm8k)
            continue
        for s in log["samples"]:
            score = s["scores"]["aime_scorer"]
            answer, matched = last_number_exact(score["metadata"]["cleaned_answer"], s["target"])
            assert answer == score["answer"], (path, s["id"], answer, score["answer"])
            if matched != (score["value"] == "C"):
                assert score["value"] == "C" and answer.endswith(s["target"]), (path, s["id"])
                flips += 1
            samples += 1
    if samples:
        print(f"aime2025: replayed {samples} logged samples: extraction identical; {flips} endswith-only matches now wrong")
    if gsm8k:
        print(f"gsm8k: replayed {gsm8k.pop('samples')} logged samples; changed verdicts by category: {dict(gsm8k)}")


if __name__ == "__main__":
    test_cases()
    test_unparsable()
    test_pipeline()
    test_gsm8k_pipeline()
    if len(sys.argv) > 1:
        replay_logs(sys.argv[1:])
    print("exact_numeric_match: all tests passed")

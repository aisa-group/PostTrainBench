"""AIME numeric answer extraction and equality grading.

inspect_ai.match(numeric=True) compares the extracted answer to the target with
str.endswith, which marks wrong answers correct when the prediction ends with
the same digit suffix (e.g. 711 vs 11).
"""

from __future__ import annotations

import re

from inspect_ai._util.text import strip_numeric_punctuation
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.scorer._common import first_number_normalized, normalize_number
from inspect_ai.scorer._metric import CORRECT, INCORRECT
from inspect_ai.solver import TaskState

ANSWER_LINE = re.compile(r"(?im)^\s*ANSWER:\s*(.+?)\s*$")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def strip_boxed(text: str) -> str:
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = BOXED.sub(r"\1", cur)
    return cur


def extract(completion: str) -> str:
    cleaned = strip_boxed(completion.strip())
    matches = ANSWER_LINE.findall(cleaned)
    if matches:
        return matches[-1].strip()

    v = strip_numeric_punctuation(cleaned.casefold())
    words = re.split(r"\s+", v)
    words.reverse()
    return first_number_normalized(words)


def grade(completion: str, target: str) -> tuple[str, bool]:
    t = target.strip()
    if not t.isnumeric():
        raise ValueError(f"AIME targets must be numeric strings, got {target!r}")

    answer = extract(completion)
    pred = normalize_number(strip_numeric_punctuation(answer.casefold()))
    gold = normalize_number(strip_numeric_punctuation(t.casefold()))
    return answer, pred == gold


@scorer(metrics=[accuracy(), stderr()])
def aime_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion
        extracted: str | None = None
        for value in target:
            extracted, matched = grade(raw, value)
            if matched:
                return Score(
                    value=CORRECT,
                    answer=extracted,
                    explanation=raw,
                    metadata={"unprocessed_answer": raw, "extracted_answer": extracted},
                )
        return Score(
            value=INCORRECT,
            answer=extracted,
            explanation=raw,
            metadata={"unprocessed_answer": raw, "extracted_answer": extracted},
        )

    return score

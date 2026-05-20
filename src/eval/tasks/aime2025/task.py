"""AIME 2025 inspect task with equality-based numeric grading."""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_evals.aime2024.aime2024 import aime2024_solver

from score import aime_scorer

DATASET_PATH = "math-ai/aime25"


@task
def aime2025() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="test",
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=aime2024_solver(),
        scorer=[aime_scorer()],
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["id"],
        input=record["problem"],
        target=str(record["answer"]),
    )

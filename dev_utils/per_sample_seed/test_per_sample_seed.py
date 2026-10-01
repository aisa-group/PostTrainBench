#!/usr/bin/env python3
"""Regression test for src/eval/per_sample_seed.py (no GPU, no datasets).

1. sample_seed() is deterministic, a non-negative 31-bit int, and differs across run seeds, samples, epochs and calls.
2. Through the real inspect pipeline with a mock model: every request of a run carries its own seed, the one
   sample_seed() gives for its sample, epoch and call index; nothing carries the run seed itself; a second generate()
   call in a sample gets another seed; and a solver passing its own seed fails loudly.

Run it in the eval container (inspect_ai is only installed there), from the repo root:
  apptainer exec --bind "$PWD" --pwd "$PWD" "$POST_TRAIN_BENCH_CONTAINERS_DIR/vllm_debug.sif" \
      python dev_utils/per_sample_seed/test_per_sample_seed.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, TaskState, generate, solver

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))

from per_sample_seed import per_sample_seed, sample_seed  # noqa: E402

RUN_SEED = 38992


def test_sample_seed() -> None:
    seeds = {sample_seed(RUN_SEED, sid, epoch, call) for sid in ("a", "b", 1, 2) for epoch in (1, 2) for call in (0, 1)}
    assert len(seeds) == 16, seeds
    assert all(0 <= s < 2**31 for s in seeds), seeds
    assert sample_seed(RUN_SEED, "a") == sample_seed(RUN_SEED, "a", 1, 0)
    assert sample_seed(RUN_SEED, "a") != sample_seed(RUN_SEED + 1, "a")
    # Fixed value: a change of the derivation changes every final-eval result, so it must be deliberate.
    assert sample_seed(0, "x") == 2001290515, sample_seed(0, "x")


@solver
def generate_twice():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        return await generate(state)

    return solve


@solver
def generate_with_own_seed():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state, seed=1)

    return solve


def request_seeds(inner, epochs: int = 1) -> dict[tuple[object, int], list[int]]:
    """{(sample id, epoch): [seed of each model request]} of a mock-model run with per_sample_seed(inner)."""
    task = Task(
        dataset=[Sample(input=f"question {i}", target="x", id=f"s{i}") for i in range(5)],
        solver=inner,
        scorer=includes(),
        epochs=epochs,
    )
    with tempfile.TemporaryDirectory() as log_dir:
        logs = inspect_eval(task, model="mockllm/model", solver=per_sample_seed(inner, RUN_SEED), log_dir=log_dir,
                            display="none")
    assert len(logs) == 1 and logs[0].status == "success", logs[0].status if logs else logs
    seeds = {}
    for sample in logs[0].samples:
        seeds[(sample.id, sample.epoch)] = [e.config.seed for e in sample.events if isinstance(e, ModelEvent)]
    return seeds


def test_pipeline() -> None:
    seeds = request_seeds(generate(), epochs=2)
    assert len(seeds) == 10, seeds
    for (sid, epoch), got in seeds.items():
        assert got == [sample_seed(RUN_SEED, sid, epoch, 0)], (sid, epoch, got)
    flat = [s for got in seeds.values() for s in got]
    assert len(set(flat)) == len(flat), flat
    assert RUN_SEED not in flat, flat

    for (sid, epoch), got in request_seeds(generate_twice()).items():
        assert got == [sample_seed(RUN_SEED, sid, epoch, 0), sample_seed(RUN_SEED, sid, epoch, 1)], (sid, got)

    task = Task(dataset=[Sample(input="q", target="x", id="s0")], solver=generate_with_own_seed(), scorer=includes())
    with tempfile.TemporaryDirectory() as log_dir:
        logs = inspect_eval(task, model="mockllm/model", solver=per_sample_seed(generate_with_own_seed(), RUN_SEED),
                            log_dir=log_dir, display="none")
    assert logs[0].status == "error", logs[0].status
    assert "passed its own seed" in str(logs[0].error), logs[0].error


if __name__ == "__main__":
    test_sample_seed()
    test_pipeline()
    print("per_sample_seed: all tests passed")

"""Per-sample generation seeds for the final evaluation (the evaluate_final_eval.py / evaluate_openrouter_final_eval.py
scripts; the agent-facing evaluate.py files take no seed).

The final evaluation runs once per fixed seed (src/eval/run_final_eval.sh). That seed must not go to vLLM as is: vLLM
gives every request that carries a seed its own torch.Generator seeded with it and samples by dividing the
probabilities by exponential noise drawn from that generator (vllm 0.11, v1/sample/ops/topk_topp_sampler.py). With one
seed on every request, all samples of a run get the same noise at every step, so their outputs collapse into one mode
(seen: all 1319 gsm8k completions of one seed were exactly '<think>\\n\\n', accuracy 0.0; other seeds 0.64). Each
request therefore gets its own seed, derived from the run's seed and the sample: reproducible for a fixed run seed,
independent across samples and across run seeds.
"""
from __future__ import annotations

import hashlib

from inspect_ai.solver import Generate, Solver, TaskState, solver


def sample_seed(run_seed: int, sample_id: str | int, epoch: int = 1, call: int = 0) -> int:
    """The generation seed of one request: a hash of the run seed, the sample id, the epoch and the index of the
    generate() call within the sample, as a non-negative 31-bit int."""
    digest = hashlib.sha256(f"{run_seed}:{sample_id}:{epoch}:{call}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


@solver
def per_sample_seed(inner: Solver, run_seed: int) -> Solver:
    """Runs the task's solver with every generate() call seeded by sample_seed(). Replaces GenerateConfig(seed=...),
    which would send the same seed with every request."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        calls = 0

        async def seeded_generate(state: TaskState, *args, **kwargs) -> TaskState:
            nonlocal calls
            if "seed" in kwargs:
                raise ValueError(f"solver passed its own seed ({kwargs['seed']}) to generate()")
            kwargs["seed"] = sample_seed(run_seed, state.sample_id, state.epoch, calls)
            calls += 1
            return await generate(state, *args, **kwargs)

        return await inner(state, seeded_generate)

    return solve

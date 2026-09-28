#!/usr/bin/env python3
"""
Recompute factors.json (per-benchmark weights of the leaderboard metric).

The weight of a benchmark is 1 / (mean zero-shot score of the instruct
models - mean zero-shot score of the base models), taken from baselines.json
and normalized so the weights over HARDCODED_BENCHMARKS sum to 1. A benchmark
where instruct-tuning helps less therefore gets a larger weight.

Rerun after changing HARDCODED_BENCHMARKS or the zero-shot baselines:
    python scripts/compute_factors.py
"""
import json

from utils import (
    FACTORS_PATH,
    HARDCODED_BENCHMARKS,
    EXPECTED_MODELS,
    load_baselines,
    mean,
)


def compute_factors(zeroshot: dict[str, dict[str, float]]) -> dict[str, float]:
    base_models = sorted(EXPECTED_MODELS)
    instruct_models = sorted(set(zeroshot) - EXPECTED_MODELS)
    missing_base = EXPECTED_MODELS - set(zeroshot)
    if missing_base:
        raise KeyError(f"baselines.json zeroshot is missing base models: {sorted(missing_base)}")
    if len(instruct_models) != len(base_models):
        raise ValueError(
            f"expected one instruct model per base model, got instruct={instruct_models} "
            f"base={base_models}"
        )

    inverse_gaps = {}
    for bench in HARDCODED_BENCHMARKS:
        gap = (
            mean([zeroshot[m][bench] for m in instruct_models])
            - mean([zeroshot[m][bench] for m in base_models])
        )
        if gap <= 0:
            raise ValueError(f"instruct-base gap for {bench} is {gap}; must be positive")
        inverse_gaps[bench] = 1.0 / gap

    total = sum(inverse_gaps.values())
    return {bench: inverse_gaps[bench] / total for bench in HARDCODED_BENCHMARKS}


def main():
    factors = compute_factors(load_baselines()["zeroshot"])
    with open(FACTORS_PATH, "w") as f:
        json.dump(factors, f, indent=4)
        f.write("\n")
    for bench, factor in factors.items():
        print(f"{bench:20s} {factor:.6f}")
    print(f"Written: {FACTORS_PATH}")


if __name__ == "__main__":
    main()

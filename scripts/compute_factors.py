#!/usr/bin/env python3
"""Derive the per-benchmark leaderboard weights (factors.json) from baselines.json.

factor[b] is proportional to 1 / (mean instruct zero-shot score on b
- mean base zero-shot score on b), normalized so the factors sum to 1. A
benchmark where the official instruct models barely beat their base models is
weighted up, so every benchmark contributes comparably to the weighted metric.
Only the benchmarks in HARDCODED_BENCHMARKS are included, so adding or
dropping a benchmark there and rerunning this script keeps factors.json
consistent.

Usage: python scripts/compute_factors.py [--check]
  --check  print the derived factors and exit non-zero if factors.json differs
"""
import argparse
import json
import sys

from utils import BASELINES_PATH, FACTORS_PATH, HARDCODED_BENCHMARKS

# Base model -> its official instruct-tuned counterpart in baselines.json["zeroshot"].
BASE_TO_INSTRUCT = {
    "Qwen3-1.7B-Base": "Qwen3-1.7B",
    "Qwen3-4B-Base": "Qwen3-4B",
    "SmolLM3-3B-Base": "SmolLM3-3B",
    "gemma-3-4b-pt": "gemma-3-4b-it",
}


def derive_factors() -> dict[str, float]:
    with open(BASELINES_PATH) as f:
        zeroshot = json.load(f)["zeroshot"]

    inverse_gaps = {}
    for bench in HARDCODED_BENCHMARKS:
        base_mean = sum(zeroshot[b][bench] for b in BASE_TO_INSTRUCT) / len(BASE_TO_INSTRUCT)
        inst_mean = sum(zeroshot[i][bench] for i in BASE_TO_INSTRUCT.values()) / len(BASE_TO_INSTRUCT)
        gap = inst_mean - base_mean
        if gap <= 0:
            raise ValueError(f"{bench}: instruct mean {inst_mean} does not exceed base mean {base_mean}")
        inverse_gaps[bench] = 1.0 / gap

    total = sum(inverse_gaps.values())
    return {bench: inverse_gaps[bench] / total for bench in HARDCODED_BENCHMARKS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 if factors.json differs from the derived factors.")
    args = parser.parse_args()

    factors = derive_factors()
    for bench, value in factors.items():
        print(f"  {bench:<18} {value:.6f}")

    if args.check:
        with open(FACTORS_PATH) as f:
            current = json.load(f)
        same = current.keys() == factors.keys() and all(
            abs(current[b] - factors[b]) < 1e-12 for b in factors)
        print("factors.json is up to date" if same else "factors.json differs from derived factors")
        return 0 if same else 1

    with open(FACTORS_PATH, "w") as f:
        json.dump(factors, f, indent=4)
        f.write("\n")
    print(f"Written: {FACTORS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Average per-seed final-evaluation metrics into a single metrics.json.

run_task.sh evaluates the final model once per fixed seed, each run writing its
own metrics file. This script writes the mean of every top-level numeric metric
(e.g. accuracy, stderr) over the seeds that succeeded, together with the number
of seeds and each seed's raw metrics. Non-numeric metrics (e.g. healthbench's
by_theme / by_axis dicts) are not averaged; they are kept under per_seed only.

Usage:
    python aggregate_seed_metrics.py --output metrics.json --num-seeds 5 \
        --seed-result 0 0 metrics_seed0.json \
        --seed-result 1 0 metrics_seed1.json
"""
from __future__ import annotations

import argparse
import json

RESERVED_KEYS = ("num_seeds", "num_seeds_succeeded", "per_seed")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average per-seed metrics into one metrics.json.")
    parser.add_argument("--output", required=True, help="Path of the aggregated metrics.json to write.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="Number of seeds the evaluation was attempted with (succeeded or not).",
    )
    parser.add_argument(
        "--seed-result",
        nargs=3,
        action="append",
        required=True,
        metavar=("SEED", "STAGE", "METRICS_PATH"),
        help="A seed that succeeded: its seed value, the retry-cascade stage that produced it, and its metrics file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    per_seed: dict[str, dict] = {}
    for seed, stage, metrics_path in args.seed_result:
        if seed in per_seed:
            raise ValueError(f"seed {seed} given more than once")
        with open(metrics_path) as f:
            metrics = json.load(f)
        if not isinstance(metrics, dict):
            raise TypeError(f"{metrics_path}: expected a JSON object, got {type(metrics).__name__}")
        per_seed[seed] = {"max_tokens_stage": int(stage), "metrics": metrics}

    if len(per_seed) > args.num_seeds:
        raise ValueError(f"{len(per_seed)} seed results given, but --num-seeds is {args.num_seeds}")

    seed_metrics = [result["metrics"] for result in per_seed.values()]
    keys = list(seed_metrics[0])
    for seed, result in per_seed.items():
        if set(result["metrics"]) != set(keys):
            raise ValueError(
                f"seed {seed} has metric keys {sorted(result['metrics'])}, expected {sorted(keys)}"
            )

    aggregated: dict[str, object] = {}
    for key in keys:
        if key in RESERVED_KEYS:
            raise ValueError(f"metric key {key!r} collides with an aggregation field")
        values = [metrics[key] for metrics in seed_metrics]
        numeric = [is_number(v) for v in values]
        if all(numeric):
            aggregated[key] = sum(values) / len(values)
        elif any(numeric):
            raise TypeError(f"metric {key!r} is numeric for some seeds but not others: {values!r}")

    # collect.py scores runs by metrics.json's "accuracy".
    if "accuracy" not in aggregated:
        raise KeyError(f"no numeric 'accuracy' metric in the seed results (keys: {keys})")

    aggregated["num_seeds"] = args.num_seeds
    aggregated["num_seeds_succeeded"] = len(per_seed)
    aggregated["per_seed"] = per_seed

    with open(args.output, "w") as f:
        json.dump(aggregated, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

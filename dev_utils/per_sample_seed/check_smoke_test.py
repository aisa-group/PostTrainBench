#!/usr/bin/env python3
"""Checks the output of smoke_test.sh: every task wrote metrics, and in the inspect tasks every model request carried
the seed sample_seed() gives for its sample (all distinct, none equal to the run seed). arenahardwriting and
healthbench send the seed in their own vLLM payload, which leaves no log, so only their metrics are checked.

Usage: check_smoke_test.py <out_dir> <run_seed> <task>...   (in the eval container, from the repo root)
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "eval"))
from per_sample_seed import sample_seed  # noqa: E402

INSPECT_TASKS = {"aime2025", "gsm8k", "humaneval", "gpqamain"}


def check_inspect_task(task_dir: str, run_seed: int) -> str:
    logs = glob.glob(os.path.join(task_dir, "inspect_logs", "*.json"))
    assert len(logs) == 1, logs
    log = json.load(open(logs[0]))
    assert log["status"] == "success", log["status"]
    assert log["eval"]["config"].get("seed") is None and log["plan"]["config"].get("seed") is None, "run-level seed set"
    seeds = []
    firsts = collections.Counter()
    for sample in log["samples"]:
        got = [e["config"].get("seed") for e in sample["events"] if e["event"] == "model"]
        want = [sample_seed(run_seed, sample["id"], sample["epoch"], i) for i in range(len(got))]
        assert got and got == want, (sample["id"], got, want)
        seeds += got
        content = sample["output"]["choices"][0]["message"]["content"] if sample["output"]["choices"] else ""
        firsts[json.dumps(content)[:25]] += 1
    assert len(set(seeds)) == len(seeds) and run_seed not in seeds, seeds
    return f"{len(log['samples'])} samples, {len(seeds)} requests with distinct per-sample seeds, " \
           f"{len(firsts)} distinct output starts"


def main() -> None:
    out_dir, run_seed, tasks = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    failed = []
    for task in tasks:
        task_dir = os.path.join(out_dir, task)
        try:
            metrics = json.load(open(os.path.join(task_dir, "metrics.json")))
            detail = check_inspect_task(task_dir, run_seed) if task in INSPECT_TASKS else "payload seed (no log)"
            print(f"OK   {task}: metrics {metrics}; {detail}")
        except Exception as err:  # report every task, then fail
            failed.append(task)
            print(f"FAIL {task}: {type(err).__name__}: {err}")
    if failed:
        raise SystemExit(f"smoke test failed for: {' '.join(failed)}")


if __name__ == "__main__":
    main()

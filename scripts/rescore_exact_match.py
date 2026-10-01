#!/usr/bin/env python3
"""Rescores finished aime2025 and gsm8k final evaluations with the exact numeric match of their final-eval scorers
(src/eval/exact_numeric_match.py, PostTrainBench issue #44) from their inspect logs, without rerunning the model.
aime2025: last_number_exact on the logged cleaned answer (the extraction must reproduce the logged answer). gsm8k:
signed_last_number_exact on the logged completion.

For a cell rerun by scripts/rerun_final_eval_parallel.py or scripts/rerun_eval_n_times.sh (per-seed outputs in
<cell>/reruns/, inspect logs in <cell>/reruns/inspect_logs/, as INSPECT_LOG_DIR put them): for every seed in
<cell>/metrics_averaged.json, it finds that seed's successful log (inspect_logs/seed<S>/, or, for logs of all seeds
in one dir, the log whose per-sample generation seeds belong to S), checks that the logged upstream verdicts give the
seed's recorded accuracy, and rescores every sample from its logged cleaned answer. It writes
reruns/metrics_seed<S>_exact_match.json and <cell>/metrics_averaged_exact_match.json (same format as
metrics_averaged.json, same seeds and stages); existing files are never overwritten.

Final evaluations of run_task.sh before 2026-10-01 kept their inspect logs in the submitting checkout, not in the
result dir, so they cannot be rescored from the result dir alone. Later ones keep them in
<run>/evaluation/inspect_logs/seed<S>/ and already use the exact match.

Usage (in the eval container, from the repo root):
  apptainer exec --bind "$PWD" --pwd "$PWD" "$POST_TRAIN_BENCH_CONTAINERS_DIR/vllm_debug.sif" \\
      python scripts/rescore_exact_match.py <cell_dir>...
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))
from exact_numeric_match import last_number_exact, signed_last_number_exact  # noqa: E402
from per_sample_seed import sample_seed  # noqa: E402


def request_seed(sample: dict) -> int | None:
    seeds = [e["config"].get("seed") for e in sample["events"] if e["event"] == "model"]
    return seeds[0] if seeds else None


def seed_log(reruns: str, seed: int) -> dict:
    """The successful inspect log of seed in reruns/inspect_logs."""
    per_seed_dir = os.path.join(reruns, "inspect_logs", f"seed{seed}")
    paths = glob.glob(os.path.join(per_seed_dir, "*.json")) if os.path.isdir(per_seed_dir) \
        else glob.glob(os.path.join(reruns, "inspect_logs", "*.json"))
    found = []
    for path in paths:
        log = json.load(open(path))
        if log.get("status") != "success":
            continue
        first = log["samples"][0]
        if os.path.isdir(per_seed_dir) or request_seed(first) == sample_seed(seed, first["id"], first["epoch"], 0):
            found.append((path, log))
    if len(found) != 1:
        raise RuntimeError(f"{reruns}: {len(found)} successful logs for seed {seed}: {[p for p, _ in found]}")
    return found[0][1]


def accuracy_and_stderr(values: list[float]) -> dict[str, float]:
    """inspect_ai's accuracy() and stderr() (sample std with ddof=1, over sqrt(n))."""
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"accuracy": mean, "stderr": std / math.sqrt(n)}


def rescore_aime2025(sample: dict) -> tuple[float, float]:
    score = sample["scores"]["aime_scorer"]
    answer, matched = last_number_exact(score["metadata"]["cleaned_answer"], sample["target"])
    if answer != score["answer"]:
        raise RuntimeError(f"sample {sample['id']}: extracted {answer!r}, logged {score['answer']!r}")
    return (1.0 if score["value"] == "C" else 0.0), (1.0 if matched else 0.0)


def rescore_gsm8k(sample: dict) -> tuple[float, float]:
    score = sample["scores"]["match"]
    _, matched = signed_last_number_exact(score["explanation"], sample["target"])
    return (1.0 if score["value"] == "C" else 0.0), (1.0 if matched else 0.0)


RESCORE = {"aime2025": rescore_aime2025, "gsm8k": rescore_gsm8k}


def rescore_cell(cell: str) -> None:
    reruns = os.path.join(cell, "reruns")
    averaged = json.load(open(os.path.join(cell, "metrics_averaged.json")))
    output = os.path.join(cell, "metrics_averaged_exact_match.json")
    if os.path.exists(output):
        raise FileExistsError(output)
    seed_args = []
    flips = 0
    for seed, info in averaged["per_seed"].items():
        log = seed_log(reruns, int(seed))
        rescore = RESCORE[os.path.basename(cell).split("_")[0]]
        upstream, exact = [], []
        for s in log["samples"]:
            old, new = rescore(s)
            upstream.append(old)
            exact.append(new)
        recorded = info["metrics"]["accuracy"]
        if abs(accuracy_and_stderr(upstream)["accuracy"] - recorded) > 1e-9:
            raise RuntimeError(f"{cell} seed {seed}: log gives {sum(upstream)}/{len(upstream)}, recorded {recorded}")
        flips += sum(1 for o, n in zip(upstream, exact) if o != n)
        seed_path = os.path.join(reruns, f"metrics_seed{seed}_exact_match.json")
        if os.path.exists(seed_path):
            raise FileExistsError(seed_path)
        with open(seed_path, "w") as f:
            json.dump(accuracy_and_stderr(exact), f, indent=2)
        seed_args += ["--seed-result", seed, str(info["max_tokens_stage"]), seed_path]
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "src", "utils", "aggregate_seed_metrics.py"),
                    "--output", output, "--num-seeds", str(averaged["num_seeds"])] + seed_args,
                   check=True, stdout=subprocess.DEVNULL)
    new = json.load(open(output))["accuracy"]
    print(f"{cell}: {averaged['accuracy']:.4f} -> {new:.4f} ({flips} verdicts changed)")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for cell in sys.argv[1:]:
        if os.path.basename(cell.rstrip("/")).split("_")[0] not in RESCORE:
            raise ValueError(f"{cell} is not an aime2025 or gsm8k cell")
        rescore_cell(os.path.abspath(cell))


if __name__ == "__main__":
    main()

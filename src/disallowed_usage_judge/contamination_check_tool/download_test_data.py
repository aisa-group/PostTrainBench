#!/usr/bin/env python3
"""
Download and normalize benchmark test data for contamination checking.

Fetches test data for all 7 benchmarks and saves normalized JSON to:
    src/eval/tasks/{task}/test_data.json

Each output file is a JSON array of {"question": ..., "answer": ...} objects.

Usage:
    python download_test_data.py                    # all tasks
    python download_test_data.py --tasks gsm8k gpqamain  # specific tasks
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TASKS_DIR = REPO_ROOT / "src" / "eval" / "tasks"

ALL_TASKS = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "healthbench",
    "humaneval",
]


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def save_test_data(task: str, data: list[dict]):
    out_path = TASKS_DIR / task / "test_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log(f"  Saved {len(data)} items to {out_path}")


def download_aime2025():
    log("Downloading aime2025 from HuggingFace (math-ai/aime25)...")
    from datasets import load_dataset
    ds = load_dataset("math-ai/aime25", split="test", streaming=True)
    data = [{"question": row["problem"], "answer": row["answer"]} for row in ds]
    save_test_data("aime2025", data)


def _iter_jsonl(text: str):
    """Iterate JSON objects from text that may contain multi-line JSON entries.

    Falls back to a streaming decoder when naive line-by-line parsing fails
    (e.g. when JSON string values contain literal newlines).
    """
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        # skip whitespace between objects
        while pos < length and text[pos] in " \t\r\n":
            pos += 1
        if pos >= length:
            break
        obj, end = decoder.raw_decode(text, pos)
        yield obj
        pos = end


def download_arenahardwriting():
    log("Downloading arenahardwriting from GitHub (lmarena/arena-hard-auto)...")
    urls = [
        "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl",
        "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/question.jsonl",
    ]
    seen_uids = set()
    items = []
    for url in urls:
        log(f"  Fetching {url}")
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
        for obj in _iter_jsonl(raw):
            uid = obj.get("uid", obj.get("question_id"))
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            prompt = obj.get("prompt", "")
            if isinstance(prompt, list):
                prompt = json.dumps(prompt, ensure_ascii=False)
            items.append({"question": prompt, "answer": ""})
    save_test_data("arenahardwriting", items)


def download_bfcl():
    log("Downloading bfcl from HuggingFace (gorilla-llm/Berkeley-Function-Calling-Leaderboard)...")
    from huggingface_hub import list_repo_files, hf_hub_download

    REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

    # Get all top-level BFCL_v3_*.json files (skip possible_answer/ and multi_turn_func_doc/)
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    bfcl_files = sorted(f for f in all_files if f.startswith("BFCL_v3_") and f.endswith(".json"))
    log(f"  Found {len(bfcl_files)} BFCL v3 files")

    data = []
    for filename in bfcl_files:
        log(f"  Fetching {filename}")
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
        with open(path) as f:
            raw = f.read()
        # Files are named .json but contain JSONL (one JSON object per line)
        for obj in _iter_jsonl(raw):
            question_parts = obj.get("question", [])
            question_str = json.dumps(question_parts, ensure_ascii=False) if not isinstance(question_parts, str) else question_parts
            ground_truth = obj.get("ground_truth", "")
            gt_str = json.dumps(ground_truth, ensure_ascii=False) if not isinstance(ground_truth, str) else ground_truth
            data.append({
                "question": question_str,
                "answer": gt_str,
            })

    save_test_data("bfcl", data)


def download_gpqamain():
    log("Downloading gpqamain from HuggingFace (Idavidrein/gpqa, gpqa_main)...")
    hf_token = os.environ.get("MY_HF_TOKEN")
    if not hf_token:
        raise RuntimeError("MY_HF_TOKEN environment variable not set (required for gated GPQA dataset)")

    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", token=hf_token, streaming=True)
    data = [{"question": row["Question"], "answer": row["Correct Answer"]} for row in ds]
    save_test_data("gpqamain", data)


def download_gsm8k():
    log("Downloading gsm8k from HuggingFace (openai/gsm8k, main)...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    data = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    save_test_data("gsm8k", data)


def _parse_healthbench_jsonl(raw_bytes: bytes) -> list[dict]:
    """Parse healthbench JSONL bytes into normalized items."""
    items = []
    for line in raw_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt = obj.get("prompt", obj.get("conversation", []))
        prompt_str = json.dumps(prompt, ensure_ascii=False) if not isinstance(prompt, str) else prompt
        rubrics = obj.get("rubrics", obj.get("criteria", []))
        rubrics_str = json.dumps(rubrics, ensure_ascii=False) if not isinstance(rubrics, str) else rubrics
        prompt_id = obj.get("prompt_id", "")
        items.append({"question": prompt_str, "answer": rubrics_str, "_prompt_id": prompt_id})
    return items


def download_healthbench():
    log("Downloading healthbench from Azure blob (openai/simple-evals)...")
    urls = [
        "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    ]
    seen_ids = set()
    data = []
    for url in urls:
        log(f"  Fetching {url}")
        with urllib.request.urlopen(url) as resp:
            items = _parse_healthbench_jsonl(resp.read())
        for item in items:
            pid = item.pop("_prompt_id")
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            data.append(item)
        log(f"    Got {len(items)} items, {len(data)} total after dedup")

    save_test_data("healthbench", data)


def download_humaneval():
    log("Downloading humaneval from HuggingFace (openai/openai_humaneval)...")
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test", streaming=True)
    data = [{"question": row["prompt"], "answer": row["canonical_solution"]} for row in ds]
    save_test_data("humaneval", data)


DOWNLOADERS = {
    "aime2025": download_aime2025,
    "arenahardwriting": download_arenahardwriting,
    "bfcl": download_bfcl,
    "gpqamain": download_gpqamain,
    "gsm8k": download_gsm8k,
    "healthbench": download_healthbench,
    "humaneval": download_humaneval,
}


def main():
    parser = argparse.ArgumentParser(description="Download benchmark test data for contamination checking.")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS,
                        help="Tasks to download (default: all)")
    args = parser.parse_args()

    log(f"Downloading test data for {len(args.tasks)} task(s): {', '.join(args.tasks)}")
    log(f"Output directory: {TASKS_DIR}")
    log("")

    failed = []
    for task in args.tasks:
        try:
            DOWNLOADERS[task]()
            log(f"  {task}: OK")
        except Exception as e:
            log(f"  {task}: FAILED - {e}")
            failed.append(task)
        log("")

    if failed:
        log(f"FAILED tasks: {', '.join(failed)}")
        sys.exit(1)

    log("All tasks downloaded successfully.")


if __name__ == "__main__":
    main()

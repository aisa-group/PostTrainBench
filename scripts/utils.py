#!/usr/bin/env python3
"""Shared constants and utility functions for aggregation scripts."""
import csv
import json
import math
import os
import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORS_PATH = os.path.join(SCRIPT_DIR, "factors.json")
BASELINES_PATH = os.path.join(SCRIPT_DIR, "baselines.json")

HARDCODED_AGENT_MAP = {
    "Opus-4.5": [
        "claude_claude-opus-4-5_10h_final_v3",
        "claude_claude-opus-4-5_10h_v5",
        "claude_claude-opus-4-5_10h_v6_seed1",
    ],
    "GPT-5.1-Codex-Max": [
        "codex_gpt-5.1-codex-max_10h_final_v3",
        "codex_gpt-5.1-codex-max_10h_v4_seed1",
        "codex_gpt-5.1-codex-max_10h_v4_seed2",
    ],
    "GPT-5.2-Codex": [
        "codex_gpt-5.2-codex_10h_v6",
        "codex_gpt-5.2-codex_10h_v6_seed1",
        "codex_gpt-5.2-codex_10h_v6_seed2",
    ],
    "GPT-5.2": [
        "codex_gpt-5.2_10h_v4",
        "codex_gpt-5.2_10h_v6_seed1",
        "codex_gpt-5.2_10h_v6_seed2",
    ],
    "Gemini-3-Pro": [
        "gemini_models_gemini-3-pro-preview_10h_final_v3",
        "gemini_models_gemini-3-pro-preview_10h_v5",
        "gemini_models_gemini-3-pro-preview_10h_v6_seed1",
    ],
    "GPT-5.1-Codex-Max Low": [
        "codexlow_gpt-5.1-codex-max_10h_v7",
        "codexlow_gpt-5.1-codex-max_10h_v7_seed1",
    ],
    "GPT-5.1-Codex-Max High": [
        "codexhigh_gpt-5.1-codex-max_10h_v7",
        "codexhigh_gpt-5.1-codex-max_10h_v7_seed1",
    ],
    "Opus-4.6": [
        "claude_claude-opus-4-6_10h_run1_old_container",
        "claude_claude-opus-4-6_10h_run2",
        "claude_claude-opus-4-6_10h_run3",
    ],
    "GPT-5.3-Codex_Med": [
        "codex_non_api_gpt-5.3-codex_10h_run1",
        "codex_non_api_gpt-5.3-codex_10h_run2",
        "codex_non_api_gpt-5.3-codex_10h_run3",
    ],
    "Gemini-3.1-Pro": [
        "opencode_opencode_gemini-3.1-pro_10h_run1",
        "opencode_opencode_gemini-3.1-pro_10h_run2",
        "opencode_opencode_gemini-3.1-pro_10h_run3",
    ],
    "GPT-5.3-Codex_High": [
        "codex_non_api_high_gpt-5.3-codex_10h_run1",
        "codex_non_api_high_gpt-5.3-codex_10h_run2",
        "codex_non_api_high_gpt-5.3-codex_10h_run3",
    ],
    "GPT-5.4-High": [
        "codex_non_api_high_gpt-5.4_10h_run1",
        "codex_non_api_high_gpt-5.4_10h_run2",
        "codex_non_api_high_gpt-5.4_10h_run3",
    ],
    "Opus-4.6-1M": [
        "claude_non_api_claude-opus-4-6_1m__10h_run1",
        "claude_non_api_claude-opus-4-6_1m__10h_run2",
        "claude_non_api_claude-opus-4-6_1m__10h_run3",
    ],
    "Opus-4.7":[
    "claude_non_api_claude-opus-4-7_10h",
    "claude_non_api_claude-opus-4-7_10h_run2",
    "claude_non_api_claude-opus-4-7_10h_run3",
    ],
    "GPT-5.5-xHigh":[
    "codex_non_api_xhigh_gpt-5.5_10h_run1",
    "codex_non_api_xhigh_gpt-5.5_10h_run2",

    ]
}

HARDCODED_BENCHMARKS = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "healthbench",
    "humaneval",
]

EXPECTED_MODELS = {
    "Qwen3-1.7B-Base",
    "Qwen3-4B-Base",
    "SmolLM3-3B-Base",
    "gemma-3-4b-pt",
}

BUDGET_SECONDS = 10 * 3600  # 10 hours


def load_factors() -> dict:
    with open(FACTORS_PATH, "r") as f:
        return json.load(f)


def load_baselines() -> dict:
    """Load hardcoded baseline data from baselines.json.

    Returns {"zeroshot": {model: {bench: value}}, "fewshot": {...}}.
    Values are floats.
    """
    with open(BASELINES_PATH, "r") as f:
        return json.load(f)


def get_baseline_fallback_data() -> dict[str, dict[str, str]]:
    """Load zeroshot baselines as {model: {bench: str_value}} for fallback.

    This is the replacement for reading aggregated_baseline_zeroshot.csv.
    """
    baselines = load_baselines()
    data = {}
    for model, benchmarks in baselines["zeroshot"].items():
        data[model] = {bench: str(val) for bench, val in benchmarks.items()}
    return data


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_results_dir() -> str:
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results")


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def is_number(value: str) -> bool:
    if not value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_csv_as_dict(csv_path: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Load a CSV into {model: {benchmark: value}}.
    Returns (data, benchmarks). Returns ({}, []) if file doesn't exist.
    """
    data = {}
    benchmarks = []

    if not os.path.exists(csv_path):
        return data, benchmarks

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return data, benchmarks

        benchmarks = header[1:]

        for row in reader:
            if not row:
                continue
            model = row[0]
            data[model] = {}
            for i, bench in enumerate(benchmarks):
                if i + 1 < len(row):
                    data[model][bench] = row[i + 1]
                else:
                    data[model][bench] = ""

    return data, benchmarks


def write_csv(
    path: str,
    models: list[str],
    benchmarks: list[str],
    data: dict[str, dict[str, str]],
):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                row.append(data[model].get(bench, ""))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Walking result directories
# ---------------------------------------------------------------------------

def walk_latest_runs(
    method_path: str,
    min_run_id: int | None = None,
    max_run_id: int | None = None,
) -> dict[tuple[str, str], dict]:
    """
    Walk a method directory and return the latest run per (benchmark, model).

    Returns {(benchmark, model): {"run_id": int, "path": str}}.
    """
    latest_runs = {}

    for entry in os.listdir(method_path):
        entry_path = os.path.join(method_path, entry)
        if not os.path.isdir(entry_path):
            continue

        try:
            benchmark, _, model, run_id_str = entry.split("_")
            run_id = int(run_id_str)
        except ValueError:
            print(entry)
            raise ValueError(f"{entry}, {method_path}")

        if max_run_id is not None and run_id >= max_run_id:
            continue
        if min_run_id is not None and run_id < min_run_id:
            continue

        key = (benchmark, model)
        if key not in latest_runs or run_id > latest_runs[key]["run_id"]:
            latest_runs[key] = {"run_id": run_id, "path": entry_path}

    return latest_runs


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

def load_metrics(metrics_path: str, method_name: str = None) -> str:
    """
    Return the accuracy as a string, or an error label.

    Error labels for non-baseline methods:
      - "not avl."   if time_taken.txt doesn't exist
      - "not stored" if time_taken.txt exists but final_model/ doesn't
      - "ERR"        otherwise
    For baseline: always "ERR" on failure.
    """
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            acc = data.get("accuracy")
            if acc is not None:
                return str(acc)
        except Exception:
            pass

    if method_name == "baseline_zeroshot":
        return "ERR"

    run_dir = os.path.dirname(metrics_path)

    if not os.path.exists(os.path.join(run_dir, "time_taken.txt")):
        return "not avl."

    if not os.path.isdir(os.path.join(run_dir, "final_model")):
        return "not stored"

    return "ERR"


# ---------------------------------------------------------------------------
# Contamination loading
# ---------------------------------------------------------------------------

def load_contamination(contamination_path: str):
    """Return True, False, "IMPORTANT ERR", or "ERR"."""
    if not os.path.exists(contamination_path):
        return "ERR"
    try:
        with open(contamination_path, "r") as f:
            content = f.read().strip()
    except Exception:
        return "ERR"
    if content == "contamination detected":
        return True
    elif content == "no contamination detected":
        return False
    else:
        return "IMPORTANT ERR"


def load_disallowed_model(disallowed_path: str):
    """Return True, False, "IMPORTANT ERR", or "ERR"."""
    if not os.path.exists(disallowed_path):
        return "ERR"
    try:
        with open(disallowed_path, "r") as f:
            content = f.read().strip()
    except Exception:
        return "ERR"
    if content == "disallowed use detected":
        return True
    elif content == "only allowed use detected":
        return False
    else:
        return "IMPORTANT ERR"


def combine_contamination_results(contamination, disallowed_model) -> str:
    """
    Combine contamination and disallowed model results into a cell value.

    Returns "" (clean), "C", "M", "MC", or an error string.
    """
    if contamination in ("ERR", "IMPORTANT ERR") or disallowed_model in (
        "ERR",
        "IMPORTANT ERR",
    ):
        errors = []
        if contamination in ("ERR", "IMPORTANT ERR"):
            errors.append(f"C:{contamination}")
        if disallowed_model in ("ERR", "IMPORTANT ERR"):
            errors.append(f"M:{disallowed_model}")
        return " ".join(errors)

    if disallowed_model and contamination:
        return "MC"
    elif disallowed_model and not contamination:
        return "M"
    elif not disallowed_model and contamination:
        return "C"
    else:
        return ""


# ---------------------------------------------------------------------------
# Time loading
# ---------------------------------------------------------------------------

def parse_time_hms(time_str: str) -> int | None:
    """Parse H:M:S string to total seconds. Returns None on failure."""
    match = re.match(r"^(\d+):(\d{1,2}):(\d{1,2})$", time_str.strip())
    if not match:
        return None
    hours, minutes, seconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def format_time_hms(total_seconds: int) -> str:
    """Convert total seconds to H:MM:SS format."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def load_time_taken(run_dir: str) -> tuple[str, int | None]:
    """
    Return (display_string, total_seconds).
    Returns ("ERR", None) on failure.
    """
    time_taken_path = os.path.join(run_dir, "time_taken.txt")

    if not os.path.exists(time_taken_path):
        return "ERR", None

    try:
        with open(time_taken_path, "r") as f:
            time_str = f.read().strip()
        total_seconds = parse_time_hms(time_str)
        if total_seconds is None:
            return "ERR", None
        return format_time_hms(total_seconds), total_seconds
    except Exception:
        return "ERR", None

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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
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

    ],
    "Opus-4.8": [
        "claude_non_api_claude-opus-4-8_10h_run1",
        "claude_non_api_claude-opus-4-8_10h_run2",
    ],
    "Opus-4.8 (Max)": [
        "claude_non_api_max_claude-opus-4-8_10h_run1",
        "claude_non_api_max_claude-opus-4-8_10h_run2",
    ],
    "GLM 5.2": [
        "glmx_glm-5.2-preview_1m__10h_run1",
        "glmx_glm-5.2-preview_1m__10h_run2",
        "glmx_glm-5.2-preview_1m__10h_run3",
    ],
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

def load_dotenv(path: str = ENV_PATH) -> dict[str, str]:
    """Parse the project's .env file into a dict.

    Raises FileNotFoundError if the .env file does not exist — collect.py
    and aggregate.py read configuration from .env, not from the ambient
    environment, so a missing file is a hard error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f".env file not found at {path}; collect.py and aggregate.py "
            f"require a project-level .env file"
        )

    env = {}
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Strip a trailing inline comment when the value is unquoted
            if value and value[0] not in ("'", '"'):
                hash_idx = value.find("#")
                if hash_idx != -1:
                    value = value[:hash_idx].strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            env[key] = value
    return env


def get_results_dir() -> str:
    env = load_dotenv()
    if "POST_TRAIN_BENCH_RESULTS_DIR" not in env:
        raise KeyError(
            f"POST_TRAIN_BENCH_RESULTS_DIR not set in {ENV_PATH}"
        )
    return env["POST_TRAIN_BENCH_RESULTS_DIR"]


def get_extra_results_dirs() -> list[str]:
    """Return additional read-only results roots to union with the primary.

    Reads ``POST_TRAIN_BENCH_EXTRA_RESULTS_DIRS`` from the project's .env file:
    a colon-separated (PATH-style) list of directories that also contain
    method subdirs. ``collect.py`` iterates methods across the primary root
    (``POST_TRAIN_BENCH_RESULTS_DIR``) plus each extra root. Output CSVs are
    still written to the primary (writable) root.

    Returns an empty list if the variable is unset or empty.
    """
    env = load_dotenv()
    raw = env.get("POST_TRAIN_BENCH_EXTRA_RESULTS_DIRS", "").strip()
    if not raw:
        return []
    return [p for p in raw.split(":") if p]


AGGREGATION_SUBDIR = "_aggregated"


def get_aggregation_dir() -> str:
    """Return the directory both collect.py and aggregate.py write their CSVs
    into by default: ``<POST_TRAIN_BENCH_RESULTS_DIR>/_aggregated``.

    Kept separate from the raw method subdirs so the results root stays tidy.
    The leading underscore prevents collect.py from mistaking it for a method
    directory (collect.py skips names starting with ``_``).
    """
    return os.path.join(get_results_dir(), AGGREGATION_SUBDIR)


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

def load_metrics(metrics_path: str) -> str:
    """Read the accuracy from metrics.json as a string.

    Raises FileNotFoundError if metrics.json is missing, json.JSONDecodeError
    if it is unparseable, KeyError if the 'accuracy' field is absent, and
    TypeError if 'accuracy' is not numeric. There is no silent fallback —
    callers that want a baseline fallback for missing runs must guard the
    call themselves.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    if "accuracy" not in data:
        raise KeyError(f"{metrics_path}: missing 'accuracy' field")
    accuracy = data["accuracy"]
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise TypeError(
            f"{metrics_path}: 'accuracy' is not a number (got "
            f"{type(accuracy).__name__}: {accuracy!r})"
        )
    return str(accuracy)


# ---------------------------------------------------------------------------
# Judge result loading
# ---------------------------------------------------------------------------

JUDGEMENT_FIELDS = ("contamination", "disallowed_model")


def judgement_path(run_dir: str) -> str:
    """Return the GPT-5.4 contamination judgement path for a run directory.

    Prefers ``judgement_gpt5_4_rerun.json`` (written by the rerun pipeline) and
    falls back to ``judgement_gpt5_4.json`` from the initial ``run_task.sh``
    run. This is the single place that encodes the rerun-over-original
    preference; everything that needs the judgement should go through here (or
    through ``load_judgement``). Raises FileNotFoundError when neither exists.
    """
    rerun_path = os.path.join(run_dir, "judgement_gpt5_4_rerun.json")
    original_path = os.path.join(run_dir, "judgement_gpt5_4.json")

    if os.path.exists(rerun_path):
        return rerun_path
    if os.path.exists(original_path):
        return original_path
    raise FileNotFoundError(
        f"No GPT-5.4 contamination judgement in {run_dir} "
        f"(expected judgement_gpt5_4_rerun.json or judgement_gpt5_4.json)"
    )


def load_judgement(run_dir: str) -> dict:
    """Load the GPT-5.4 contamination judge verdict for a single run directory.

    Reads only the GPT-5.4 contamination judge output (preferring the rerun
    file; see ``judgement_path``). The third-party API usage judge
    (``judgement_api.json``) is intentionally ignored.

    Raises FileNotFoundError when neither judgement file exists,
    json.JSONDecodeError on a malformed file, and ValueError/TypeError when the
    schema does not match what the contamination judge writes.
    """
    path = judgement_path(run_dir)

    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON is not an object")

    missing = [f for f in JUDGEMENT_FIELDS if f not in data]
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")

    for field in JUDGEMENT_FIELDS:
        if not isinstance(data[field], bool):
            raise TypeError(
                f"{path}: field {field!r} must be bool, got "
                f"{type(data[field]).__name__}: {data[field]!r}"
            )

    return {field: data[field] for field in JUDGEMENT_FIELDS}


def judgement_to_cell(judgement: dict) -> str:
    """Encode the GPT-5.4 contamination judge booleans into a single cell.

    The cell concatenates the letter for each flag that is True:
      - 'M' = disallowed_model
      - 'C' = contamination
    Returns '' when both flags are False. Order is fixed (M, C) so cells are
    comparable across runs.
    """
    parts = []
    if judgement["disallowed_model"]:
        parts.append("M")
    if judgement["contamination"]:
        parts.append("C")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Time loading
# ---------------------------------------------------------------------------

def parse_time_hms(time_str: str) -> int:
    """Parse an H:M:S string into total seconds. Raises ValueError on bad input."""
    match = re.match(r"^(\d+):(\d{1,2}):(\d{1,2})$", time_str.strip())
    if not match:
        raise ValueError(f"time string is not H:M:S: {time_str!r}")
    hours, minutes, seconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"time string has invalid minutes/seconds: {time_str!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_time_hms(total_seconds: int) -> str:
    """Convert total seconds to H:MM:SS format."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def load_time_taken(run_dir: str) -> tuple[str, int]:
    """Return (display_string, total_seconds) from time_taken.txt.

    Raises FileNotFoundError if the file is missing and ValueError if the
    contents are not in H:M:S format.
    """
    time_taken_path = os.path.join(run_dir, "time_taken.txt")
    if not os.path.exists(time_taken_path):
        raise FileNotFoundError(f"time_taken.txt not found: {time_taken_path}")
    with open(time_taken_path, "r") as f:
        time_str = f.read().strip()
    total_seconds = parse_time_hms(time_str)
    return format_time_hms(total_seconds), total_seconds

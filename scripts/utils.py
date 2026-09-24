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
    "GPT-5.6-Sol": [
        "codex_non_api_max_gpt-5.6-sol_10h_run1",
        "codex_non_api_max_gpt-5.6-sol_10h_run2",
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
    "Fable 5 (Max)": [
        "claude_non_api_max_claude-fable-5_1m__10h_run1",
        "claude_non_api_max_claude-fable-5_1m__10h_run2",
    ],
    "Kimi K3": [
        "kimi_claude_k3-0715_1m__10h_run1",
        "kimi_claude_k3-0715_1m__10h_run2",
        "kimi_claude_k3-0715_1m__10h_run3",
    ],
    "Grok 4.5": [
        "cursor_cli_cursor-grok-4.5-high_10h_run1",
        "cursor_cli_cursor-grok-4.5-high_10h_run2",
    ],
    "Opus-5": [
        "claude_non_api_claude-opus-5_10h_run1",
        "claude_non_api_claude-opus-5_10h_run2",
    ],
    "GLM 5.3": [
        "glmx_glm-5.3_1m__10h_run1",
        "glmx_glm-5.3_1m__10h_run2",

    ],
    "GLM 5.3 Flash": [
        "glmx_glm-5.3-flash_10h_run1",
        "glmx_glm-5.3-flash_10h_run2",
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


MULTI_RUN_DIRNAME = "judgement_multi_runs"


def _slot1_judgement_path(run_dir: str) -> str | None:
    """Return the slot-1 (single-verdict) contamination judgement path, or None.

    Slot 1 = whichever of the historical single-verdict files exists, with the
    same rerun-over-inline preference the pipeline has always used.
    """
    rerun_path = os.path.join(run_dir, "judgement_gpt5_4_rerun.json")
    original_path = os.path.join(run_dir, "judgement_gpt5_4.json")
    if os.path.exists(rerun_path):
        return rerun_path
    if os.path.exists(original_path):
        return original_path
    return None


def _compute_majority_verdict(
    run_dir: str, slot1_path: str, slot2_path: str, slot3_path: str
) -> dict:
    """Load the three slot verdicts and return the per-field majority as a dict.

    Pure function — reads only. Callers who want to persist the result use
    ``_try_write_majority_cache`` afterwards (best-effort; not required).
    """
    slot_infos = []
    for slot_num, path in ((1, slot1_path), (2, slot2_path), (3, slot3_path)):
        with open(path, "r") as f:
            data = json.load(f)
        slot_infos.append((slot_num, path, data))

    contam_maj = sum(1 for _, _, d in slot_infos if d["contamination"]) >= 2
    dm_maj = sum(1 for _, _, d in slot_infos if d["disallowed_model"]) >= 2

    def _joined(field: str) -> str:
        parts = []
        for slot_num, _, d in slot_infos:
            parts.append(
                f"[slot {slot_num}] contamination={d['contamination']} "
                f"disallowed_model={d['disallowed_model']}\n"
                + d.get(field, "")
            )
        return "\n\n".join(parts)

    return {
        "contamination": contam_maj,
        "disallowed_model": dm_maj,
        "justification_contamination": _joined("justification_contamination"),
        "justification_disallowed_model": _joined("justification_disallowed_model"),
        "_meta": {
            "aggregation": "per_field_majority_of_3",
            "slots": [
                {
                    "slot": slot_num,
                    "path": os.path.relpath(path, run_dir),
                    "contamination": d["contamination"],
                    "disallowed_model": d["disallowed_model"],
                    "judge_model": d.get("_meta", {}).get("judge_model", "unknown"),
                    "judge_codex_version": d.get("_meta", {}).get(
                        "judge_codex_version", "unknown"
                    ),
                    "timestamp": d.get("_meta", {}).get("timestamp", ""),
                }
                for slot_num, path, d in slot_infos
            ],
        },
    }


def _try_write_majority_cache(final_path: str, verdict: dict) -> bool:
    """Try to write ``verdict`` to ``final_path`` (tempfile + atomic rename).

    Returns True on success, False if the write fails (e.g. Lustre EIO on
    the target directory). A failed write never leaves a corrupt final file.
    Failure is a warning, not an error — the caller has the verdict in memory
    already; caching is only for audit.
    """
    try:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(verdict, f, indent=2)
        os.replace(tmp_path, final_path)
        return True
    except OSError as e:
        import sys
        print(
            f"WARNING: could not cache majority verdict to {final_path}: {e}",
            file=sys.stderr,
        )
        try:
            os.remove(final_path + ".tmp")
        except OSError:
            pass
        return False


def _multi_run_slot_paths(run_dir: str) -> tuple[str | None, str, str, str]:
    """Return (slot1_path_or_None, slot2_path, slot3_path, final_path)."""
    multi_dir = os.path.join(run_dir, MULTI_RUN_DIRNAME)
    return (
        _slot1_judgement_path(run_dir),
        os.path.join(multi_dir, "judgement_gpt5_4_run2.json"),
        os.path.join(multi_dir, "judgement_gpt5_4_run3.json"),
        os.path.join(multi_dir, "judgement_gpt5_4_final.json"),
    )


MANUAL_JUDGEMENT_FILENAME = "judgement_gpt5_4_manual.json"


def manual_judgement_path(run_dir: str) -> str:
    """Path of the manual-override contamination verdict (may not exist).

    Written by hand after a human review (see dev_utils/README.md); it ranks
    above every judge output (see ``resolve_judgement``).
    """
    return os.path.join(run_dir, MANUAL_JUDGEMENT_FILENAME)


def materialize_majority_verdict(run_dir: str) -> str | None:
    """(Re)write the multi-run majority verdict cache for ``run_dir``.

    Always recomputes the majority from the three slot files and rewrites
    ``judgement_multi_runs/judgement_gpt5_4_final.json``. An existing cache is
    never returned as-is: a slot may have been re-run (or slot 1 superseded by
    a ``_rerun.json``) since it was written. Returns the freshly written path,
    or None when the three slots aren't all present OR the write fails (e.g.
    Lustre EIO) — in both cases a leftover cache file is stale and must not be
    used. A None return does NOT mean the majority is unavailable —
    ``resolve_judgement`` computes it in memory regardless. This function is
    only for callers that need a concrete file path (e.g. the extract
    pipeline that hands the file to the trace viewer).
    """
    slot1, slot2, slot3, final_path = _multi_run_slot_paths(run_dir)
    if slot1 is None or not os.path.exists(slot2) or not os.path.exists(slot3):
        return None
    verdict = _compute_majority_verdict(run_dir, slot1, slot2, slot3)
    if _try_write_majority_cache(final_path, verdict):
        return final_path
    return None


def judgement_path(run_dir: str) -> str:
    """Return a GPT-5.4 contamination judgement path for a run directory.

    Preference order:
      1. Manual override ``judgement_gpt5_4_manual.json``.
      2. Majority verdict, freshly rewritten to
         ``judgement_multi_runs/judgement_gpt5_4_final.json`` (see
         ``materialize_majority_verdict``; the write may fail on Lustre EIO).
      3. Legacy single verdict: ``judgement_gpt5_4_rerun.json`` when present,
         else ``judgement_gpt5_4.json``.

    Note: prefer ``load_judgement`` / ``resolve_judgement`` when you only need
    the verdict; they apply the majority in-memory even when the cache write
    is failing.
    """
    manual = manual_judgement_path(run_dir)
    if os.path.exists(manual):
        return manual

    final_path = materialize_majority_verdict(run_dir)
    if final_path is not None:
        return final_path

    slot1 = _slot1_judgement_path(run_dir)
    if slot1 is not None:
        return slot1

    raise FileNotFoundError(
        f"No GPT-5.4 contamination judgement in {run_dir} "
        f"(expected {MANUAL_JUDGEMENT_FILENAME}, "
        f"judgement_multi_runs/judgement_gpt5_4_run{{2,3}}.json, "
        f"judgement_gpt5_4_rerun.json, or judgement_gpt5_4.json)"
    )


def _validate_judgement_schema(data: dict, source: str) -> dict:
    """Enforce the contamination-verdict schema; return the checked fields dict."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top-level JSON is not an object")
    missing = [f for f in JUDGEMENT_FIELDS if f not in data]
    if missing:
        raise ValueError(f"{source}: missing fields: {', '.join(missing)}")
    for field in JUDGEMENT_FIELDS:
        if not isinstance(data[field], bool):
            raise TypeError(
                f"{source}: field {field!r} must be bool, got "
                f"{type(data[field]).__name__}: {data[field]!r}"
            )
    return {field: data[field] for field in JUDGEMENT_FIELDS}


def resolve_judgement(run_dir: str) -> tuple[str, dict] | None:
    """Resolve the effective contamination verdict of a run directory.

    This is the single place that encodes the verdict precedence; everything
    that needs the contamination verdict (collect.py, find_flagged_runs.py,
    dev_utils/contamination_list.py, ...) should go through here or
    ``load_judgement``:
      1. ``judgement_gpt5_4_manual.json`` — a human reviewer's override,
         written by hand (see dev_utils/README.md). Wins over every judge
         output.
      2. Per-field majority of the three multi-run slots (slot 1 =
         ``_rerun.json`` else inline; slots 2/3 under
         ``judgement_multi_runs/``), computed in memory. The ``_final.json``
         cache is rewritten opportunistically for audit but never read.
      3. The single judge verdict: ``judgement_gpt5_4_rerun.json`` when
         present, else ``judgement_gpt5_4.json``.

    Returns ``(source, verdict)``: ``source`` names where the verdict came
    from (paths relative to ``run_dir``), ``verdict`` is the full,
    schema-checked JSON object including the justifications. Returns None
    when the run has no contamination verdict at all. Raises
    json.JSONDecodeError on a malformed file and ValueError/TypeError when the
    schema does not match what the contamination judge writes.
    """
    manual = manual_judgement_path(run_dir)
    if os.path.exists(manual):
        with open(manual, "r") as f:
            data = json.load(f)
        _validate_judgement_schema(data, manual)
        return MANUAL_JUDGEMENT_FILENAME, data

    slot1, slot2_path, slot3_path, final_path = _multi_run_slot_paths(run_dir)

    if slot1 is not None and os.path.exists(slot2_path) and os.path.exists(slot3_path):
        verdict = _compute_majority_verdict(run_dir, slot1, slot2_path, slot3_path)
        _try_write_majority_cache(final_path, verdict)  # opportunistic, audit only
        _validate_judgement_schema(verdict, f"majority({run_dir})")
        slots = ", ".join(
            os.path.relpath(p, run_dir) for p in (slot1, slot2_path, slot3_path)
        )
        return f"majority of 3 ({slots})", verdict

    if slot1 is not None:
        with open(slot1, "r") as f:
            data = json.load(f)
        _validate_judgement_schema(data, slot1)
        return os.path.relpath(slot1, run_dir), data

    return None


def load_judgement(run_dir: str) -> dict:
    """Load the effective contamination verdict fields for a run directory.

    Resolved by ``resolve_judgement`` (manual override > majority of 3 >
    single verdict); returns only the ``JUDGEMENT_FIELDS`` booleans.

    Raises FileNotFoundError when no verdict is available,
    json.JSONDecodeError on a malformed file, and ValueError/TypeError when
    the schema does not match what the contamination judge writes.
    """
    resolved = resolve_judgement(run_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"No GPT-5.4 contamination judgement in {run_dir} "
            f"(expected {MANUAL_JUDGEMENT_FILENAME}, "
            f"judgement_multi_runs/judgement_gpt5_4_run{{2,3}}.json, "
            f"judgement_gpt5_4_rerun.json, or judgement_gpt5_4.json)"
        )
    source, data = resolved
    return _validate_judgement_schema(data, source)


API_USAGE_FIELD = "disallowed_api_usage"
PTB_LOOKUP_FIELD = "disallowed_ptb_lookup"


def optional_judgement_path(run_dir: str, basename: str) -> str | None:
    """Return the verdict path for a judge whose file may legitimately be absent.

    ``basename`` is the judge's output id (JUDGE_OUTPUT_ID in its judge.conf),
    e.g. ``api``, ``ptb_lookup``, ``general``, ``gpt5_4``. Prefers
    ``judgement_{basename}_rerun.json`` (written by the rerun pipeline) over
    ``judgement_{basename}.json`` from the initial ``run_task.sh`` run; None
    when neither exists (the run predates the judge).
    """
    rerun_path = os.path.join(run_dir, f"judgement_{basename}_rerun.json")
    original_path = os.path.join(run_dir, f"judgement_{basename}.json")

    if os.path.exists(rerun_path):
        return rerun_path
    if os.path.exists(original_path):
        return original_path
    return None


def _load_optional_flag_judgement(
    run_dir: str, basename: str, field: str
) -> bool | None:
    """Load a single-boolean judge verdict that may legitimately be absent.

    Prefers ``judgement_{basename}_rerun.json`` (written by the rerun
    pipeline) and falls back to ``judgement_{basename}.json`` from the initial
    ``run_task.sh`` run. Unlike the contamination judgement, a missing file is
    not an error: runs that predate the judge have none, so None is returned
    instead of raising. Raises json.JSONDecodeError on a malformed file and
    ValueError/TypeError when the schema does not match what the judge writes.
    """
    path = optional_judgement_path(run_dir, basename)
    if path is None:
        return None

    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON is not an object")
    if field not in data:
        raise ValueError(f"{path}: missing field: {field}")
    if not isinstance(data[field], bool):
        raise TypeError(
            f"{path}: field {field!r} must be bool, got "
            f"{type(data[field]).__name__}: {data[field]!r}"
        )
    return data[field]


def load_api_judgement(run_dir: str) -> bool | None:
    """Load the third-party API usage judge verdict for a run directory.

    Returns the ``disallowed_api_usage`` boolean, or None when no API
    judgement file exists (the run predates this judge). A True verdict is
    consumed by scoring: the run's score falls back to the baseline.
    """
    return _load_optional_flag_judgement(run_dir, "api", API_USAGE_FIELD)


def ptb_lookup_judgement_path(run_dir: str) -> str | None:
    """Return the PTB-lookup judge verdict path for a run directory.

    Prefers the ``_rerun`` file (see ``optional_judgement_path``); None
    when the run has no PTB-lookup judgement (it predates the judge).
    """
    return optional_judgement_path(run_dir, "ptb_lookup")


def load_ptb_lookup_judgement(run_dir: str) -> bool | None:
    """Load the PTB-lookup judge verdict for a single run directory.

    Returns the ``disallowed_ptb_lookup`` boolean, or None when no PTB-lookup
    judgement file exists (the run predates this judge). This verdict is
    archival — it does not feed score fallback — but collect.py raises when
    it is True so a firing lookup judge cannot pass unnoticed.
    """
    return _load_optional_flag_judgement(run_dir, "ptb_lookup", PTB_LOOKUP_FIELD)


# First run id for which the ptb_lookup_judge is required on every scored
# agent run: chosen above every run id existing on 2026-07-16 (max was
# 17397666), so it covers exactly the sweeps launched after the judge was
# part of the inline set in run_task.sh. Verdicts on older runs (e.g. from
# the rerun pipeline) are still read as tripwires when present — this
# threshold only governs whether their absence is a violation.
NEWER_JUDGES_MIN_RUN_ID = 17400000


def missing_required_judgements(run_dir: str, run_id: int) -> list[str]:
    """Names of the judges whose verdict a scored agent run must have but lacks.

    The contamination and API-usage judges are required on every scored run;
    the PTB-lookup judge only on runs with
    ``run_id >= NEWER_JUDGES_MIN_RUN_ID`` (older runs predate it). The
    general (unknown-unknowns) judge is never required: its verdict is
    ignored by scoring entirely (review it via find_flagged_runs.py).
    Baseline methods have no judges by design — callers must not apply this
    check to them. A malformed verdict file still raises; only a genuinely
    absent one counts as missing.
    """
    missing = []
    try:
        load_judgement(run_dir)
    except FileNotFoundError:
        missing.append("data_contamination_judge")
    if load_api_judgement(run_dir) is None:
        missing.append("api_usage_judge")
    if run_id >= NEWER_JUDGES_MIN_RUN_ID:
        if load_ptb_lookup_judgement(run_dir) is None:
            missing.append("ptb_lookup_judge")
    return missing


def judgement_to_cell(judgement: dict, api_usage: bool | None = None) -> str:
    """Encode the judge booleans into a single cell.

    The cell concatenates the letter for each flag that is True:
      - 'M' = disallowed_model      (GPT-5.4 contamination judge)
      - 'C' = contamination         (GPT-5.4 contamination judge)
      - 'A' = disallowed_api_usage  (API usage judge; pass None when that
        judge never ran, which leaves the letter out)
    Returns '' when no flag is set. Order is fixed (M, C, A) so cells are
    comparable across runs. The PTB-lookup verdict is deliberately not part
    of the cell: it is archival, and collect.py errors out when it fires.
    The general verdict is ignored by scoring entirely.
    """
    parts = []
    if judgement["disallowed_model"]:
        parts.append("M")
    if judgement["contamination"]:
        parts.append("C")
    if api_usage:
        parts.append("A")
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

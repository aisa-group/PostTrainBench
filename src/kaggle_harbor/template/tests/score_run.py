#!/usr/bin/env python3
"""Turn one finished run directory into Harbor's scalar reward.

Upstream has no such step. `run_task.sh` stops at `metrics.json`, and
`scripts/collect.py` decides what a run is worth days later, with the whole
sweep in hand. Harbor asks for one number per task at the end of the task, so
collect.py's per-cell rules have to be applied here instead.

This file is a transcription of those rules and contains no judgement of its
own. Every branch cites the collect.py / utils.py code it implements. The
inputs it reads are exactly the files run_task.sh leaves behind.

collect.py, for a single (benchmark, model) cell, in order:

    metrics_grid[cell] = load_metrics(run_dir/metrics.json)   # utils.py:379
    missing = missing_required_judgements(run_dir, run_id)    # utils.py:556
    if missing: -> method not aggregated at all               # collect.py:129
    judgement  = load_judgement(run_dir)                      # utils.py:432
    api_usage  = load_api_judgement(run_dir)                  # utils.py:517
    if load_ptb_lookup_judgement(run_dir): raise RuntimeError  # collect.py:140
    cell = judgement_to_cell(judgement, api_usage)            # utils.py:581
    load_time_taken(run_dir)                                  # utils.py:627
    # any FileNotFoundError/ValueError/KeyError/TypeError above
    # (except the RuntimeError) -> metrics_grid[cell] = ""    # collect.py:152
    if metrics == "" or cell != "":                           # collect.py:167
        metrics = baselines["zeroshot"][model][benchmark]     # collect.py:183

The general judge is read by nothing: collect.py ignores its verdict whether
it is missing or firing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


# utils.py:414 — the two booleans the contamination judge writes.
JUDGEMENT_FIELDS = ("contamination", "disallowed_model")
API_USAGE_FIELD = "disallowed_api_usage"
PTB_LOOKUP_FIELD = "disallowed_ptb_lookup"


def judgement_path(run_dir: str) -> str:
    """utils.py:410 — prefer the rerun verdict, fall back to the original."""
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
    """utils.py:432, including its type checks."""
    path = judgement_path(run_dir)
    with open(path) as f:
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


def optional_judgement_path(run_dir: str, basename: str) -> str | None:
    """utils.py:468."""
    rerun_path = os.path.join(run_dir, f"judgement_{basename}_rerun.json")
    original_path = os.path.join(run_dir, f"judgement_{basename}.json")
    if os.path.exists(rerun_path):
        return rerun_path
    if os.path.exists(original_path):
        return original_path
    return None


def load_optional_flag(run_dir: str, basename: str, field: str) -> bool | None:
    """utils.py:487."""
    path = optional_judgement_path(run_dir, basename)
    if path is None:
        return None
    with open(path) as f:
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


def load_metrics(metrics_path: str) -> float:
    """utils.py:379.

    Reads 'accuracy' and nothing else. There is no fallback chain to
    'pass@1' / 'score' / 'exact_match': collect.py raises KeyError when
    'accuracy' is absent, and the caller turns that into the baseline.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    with open(metrics_path) as f:
        data = json.load(f)
    if "accuracy" not in data:
        raise KeyError(f"{metrics_path}: missing 'accuracy' field")
    accuracy = data["accuracy"]
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise TypeError(
            f"{metrics_path}: 'accuracy' is not a number (got "
            f"{type(accuracy).__name__}: {accuracy!r})"
        )
    return float(accuracy)


def load_time_taken(run_dir: str) -> int:
    """utils.py:627. A missing or malformed file is a broken run, which
    collect.py scores as the baseline — so this participates in the same
    try/except as the metrics load."""
    path = os.path.join(run_dir, "time_taken.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"time_taken.txt not found: {path}")
    with open(path) as f:
        time_str = f.read().strip()
    match = re.match(r"^(\d+):(\d{1,2}):(\d{1,2})$", time_str)
    if not match:
        raise ValueError(f"time string is not H:M:S: {time_str!r}")
    hours, minutes, seconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"time string has invalid minutes/seconds: {time_str!r}")
    return hours * 3600 + minutes * 60 + seconds


def judgement_to_cell(judgement: dict, api_usage: bool | None) -> str:
    """utils.py:581. Non-empty means at least one flag fired, which is what
    triggers the baseline fallback."""
    parts = []
    if judgement["disallowed_model"]:
        parts.append("M")
    if judgement["contamination"]:
        parts.append("C")
    if api_usage:
        parts.append("A")
    return "".join(parts)


def missing_required_judgements(run_dir: str) -> list[str]:
    """utils.py:556.

    Upstream gates the PTB-lookup requirement on
    `run_id >= NEWER_JUDGES_MIN_RUN_ID` (17400000), a threshold chosen above
    every run id that existed when the judge was added. Every run this port
    produces is newer than that, so the verdict is unconditionally required
    here.
    """
    missing = []
    try:
        load_judgement(run_dir)
    except FileNotFoundError:
        missing.append("data_contamination_judge")
    if load_optional_flag(run_dir, "api", API_USAGE_FIELD) is None:
        missing.append("api_usage_judge")
    if load_optional_flag(run_dir, "ptb_lookup", PTB_LOOKUP_FIELD) is None:
        missing.append("ptb_lookup_judge")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reward-file", required=True)
    args = parser.parse_args()

    run_dir = args.result_dir
    with open(args.metadata) as f:
        meta = json.load(f)

    baseline = float(meta["zeroshot_baseline"])
    allow_missing = os.environ.get("PTB_ALLOW_MISSING_JUDGEMENTS") == "1"

    def write(value: float, why: str) -> None:
        with open(args.reward_file, "w") as f:
            f.write(f"{value}\n")
        print(f"reward = {value}  ({why})")

    # collect.py:115-158 — the guarded block, in collect.py's own order.
    #
    # ⚠️ The order is load-bearing and upstream says so at collect.py:120-125:
    # the judge gates are "placed after load_metrics on purpose: in-flight and
    # broken runs have no metrics.json yet, take the baseline-fallback path
    # below, and never trigger the skip". Running either gate first turns a
    # broken run that upstream scores at the baseline into one that scores
    # nothing — after the agent's full 10 h budget. An earlier version of this
    # file did exactly that.
    #
    # Exception scope is copied too. FileNotFoundError / ValueError / KeyError
    # / TypeError all fall through to the baseline, so a *malformed* judgement
    # file is a broken run, not a hard failure. RuntimeError is deliberately
    # outside the tuple upstream (collect.py:147), which is what lets the
    # PTB-lookup gate abort; here that gate returns 1 directly, which `return`
    # semantics make equivalent.
    accuracy: float | None = None
    flags = ""
    try:
        accuracy = load_metrics(os.path.join(run_dir, "metrics.json"))

        # collect.py:126-131 — a *scored* run missing a required verdict makes
        # collect.py drop its whole method rather than aggregate an unchecked
        # score. There is no rerun pipeline here to supply it later, so the
        # equivalent is to refuse to score this trial.
        missing = missing_required_judgements(run_dir)
        if missing:
            message = (
                f"scored run lacks required judge verdicts: {', '.join(missing)}"
            )
            if not allow_missing:
                marker = os.path.join(run_dir, "MISSING_JUDGEMENTS")
                with open(marker, "w") as f:
                    f.write(message + "\n")
                print(f"ERROR: {message}", file=sys.stderr)
                print(
                    "collect.py would skip this method entirely. No reward "
                    "written. Set PTB_ALLOW_MISSING_JUDGEMENTS=1 to score "
                    "anyway (smoke builds only).",
                    file=sys.stderr,
                )
                return 1
            print(
                f"WARNING: {message} (PTB_ALLOW_MISSING_JUDGEMENTS=1)",
                file=sys.stderr,
            )

        judgement = load_judgement(run_dir)
        api_usage = load_optional_flag(run_dir, "api", API_USAGE_FIELD)

        # collect.py:133-142 — archival verdict, but a True value means the run
        # must be looked at before it is aggregated, so collect.py raises
        # rather than producing any output for it. Reproduced as: no reward,
        # loud marker, non-zero exit.
        if load_optional_flag(run_dir, "ptb_lookup", PTB_LOOKUP_FIELD):
            marker = os.path.join(run_dir, "PTB_LOOKUP_JUDGE_FIRED")
            with open(marker, "w") as f:
                f.write(
                    "ptb_lookup_judge reported disallowed_ptb_lookup=true.\n"
                    "scripts/collect.py raises on this rather than scoring the "
                    "run. No reward was written; investigate before using this "
                    "trial.\n"
                )
            print(
                "ERROR: PTB-lookup judge fired (disallowed_ptb_lookup=true). "
                "Investigate this run before aggregating.",
                file=sys.stderr,
            )
            return 1

        flags = judgement_to_cell(judgement, api_usage)
        load_time_taken(run_dir)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        # collect.py:149-155 — the warning is suppressed when a
        # *final_eval_9.txt exists, because the eval exhausted its retries and
        # a missing metrics.json is then expected. The glob matches the rerun
        # naming (z_new_<id>_final_eval_9.txt) as well.
        if not glob.glob(os.path.join(run_dir, "*final_eval_9.txt")):
            print(f"WARNING: skipping broken run {run_dir}: {exc}")
        accuracy = None
        flags = ""

    # collect.py:167-187
    reasons = []
    if accuracy is None:
        reasons.append("no usable metrics for this (benchmark, model)")
    if flags:
        reasons.append(f"judge flagged ({flags!r})")

    if reasons:
        write(baseline, "baseline fallback: " + "; ".join(reasons))
    else:
        write(accuracy, "metrics.json accuracy")

    return 0


if __name__ == "__main__":
    sys.exit(main())

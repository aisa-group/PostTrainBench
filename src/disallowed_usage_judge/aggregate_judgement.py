#!/usr/bin/env python3
"""Aggregate per-judge judgement.json files into a single judge_result.json.

All judges are passed as repeatable `--judge NAME=PATH`. The judge with the
reserved name `api` is interpreted as the third-party API usage judge and must
contain:
  - disallowed_api_usage (bool)
  - justification_disallowed_api_usage (string)

Every other judge is interpreted as a contamination/base-model judge and must
contain:
  - contamination (bool)
  - disallowed_model (bool)
  - justification_contamination (string)
  - justification_disallowed_model (string)

At least one contamination judge and exactly one `api` judge must be supplied.

The aggregated output schema:
  - contamination: True if ANY contamination judge flagged it
  - disallowed_model: True if ANY contamination judge flagged it
  - disallowed_api_usage: verbatim from the API judge
  - justification_contamination / justification_disallowed_model: per-judge
    justifications concatenated with a `[judge_name]` tag
  - justification_disallowed_api_usage: verbatim from the API judge

If any supplied per-judge file is missing, unparseable, or missing required
fields, this script fails loudly (non-zero exit, no output written) so that
callers don't end up with a False/False default that masks a crashed judge.

Usage:
    aggregate_judgement.py --output judge_result.json \
        --judge gpt5_4=judgement_gpt5_4.json \
        --judge deepseek=judgement_deepseek.json \
        --judge api=judgement_api.json
"""

import argparse
import json
from pathlib import Path


CONTAMINATION_FIELDS = (
    "contamination",
    "disallowed_model",
    "justification_contamination",
    "justification_disallowed_model",
)

API_FIELDS = (
    "disallowed_api_usage",
    "justification_disallowed_api_usage",
)

API_JUDGE_NAME = "api"


def read_judgement(path: Path, required_fields: tuple[str, ...]) -> dict:
    """Load and validate a per-judge judgement file. Raise on any problem."""
    if not path.exists():
        raise SystemExit(f"ERROR: judgement file not found: {path}")
    raw = path.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: {path} is not a JSON object")
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise SystemExit(f"ERROR: {path} missing fields: {', '.join(missing)}")
    return data


def aggregate(contamination_judgements: dict[str, dict], api_judgement: dict) -> dict:
    contamination = any(bool(j["contamination"]) for j in contamination_judgements.values())
    disallowed_model = any(bool(j["disallowed_model"]) for j in contamination_judgements.values())

    def collect(field: str) -> str:
        return "\n\n".join(
            f"[{name}] {data[field]}" for name, data in contamination_judgements.items()
        )

    return {
        "contamination": contamination,
        "disallowed_model": disallowed_model,
        "justification_contamination": collect("justification_contamination"),
        "justification_disallowed_model": collect("justification_disallowed_model"),
        "disallowed_api_usage": bool(api_judgement["disallowed_api_usage"]),
        "justification_disallowed_api_usage": api_judgement["justification_disallowed_api_usage"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Per-judge JSON file. The reserved name `api` is the third-party "
            "API usage judge; any other name is a contamination/base-model "
            "judge. Repeat for each judge."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the aggregated judge_result.json",
    )
    args = parser.parse_args()

    contamination_judgements: dict[str, dict] = {}
    api_judgement: dict | None = None
    for spec in args.judge:
        if "=" not in spec:
            raise SystemExit(f"--judge value must be NAME=PATH, got: {spec!r}")
        name, path_str = spec.split("=", 1)
        if name == API_JUDGE_NAME:
            if api_judgement is not None:
                raise SystemExit(f"ERROR: multiple `{API_JUDGE_NAME}` judges supplied")
            api_judgement = read_judgement(Path(path_str), API_FIELDS)
        else:
            contamination_judgements[name] = read_judgement(Path(path_str), CONTAMINATION_FIELDS)

    if not contamination_judgements:
        raise SystemExit("ERROR: at least one contamination judge is required")
    if api_judgement is None:
        raise SystemExit(f"ERROR: an `{API_JUDGE_NAME}` judge is required")

    result = aggregate(contamination_judgements, api_judgement)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Aggregated {len(contamination_judgements)} contamination judge(s) + 1 api judge "
        f"-> {args.output} "
        f"(contamination={result['contamination']}, "
        f"disallowed_model={result['disallowed_model']}, "
        f"disallowed_api_usage={result['disallowed_api_usage']})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate per-judge judgement.json files into a single judge_result.json.

Each per-judge file must contain:
  - contamination (bool)
  - disallowed_model (bool)
  - justification_contamination (string)
  - justification_disallowed_model (string)

The aggregated output uses the same schema:
  - contamination: True if ANY judge flagged contamination
  - disallowed_model: True if ANY judge flagged disallowed model use
  - justification_*: per-judge justifications concatenated with a `[judge_name]` tag

If any per-judge file is missing, unparseable, or missing required fields,
this script fails loudly (non-zero exit, no output written) so that callers
don't end up with a False/False default that masks a crashed judge.

Usage:
    aggregate_judgement.py --output judge_result.json \
        --judge gpt5_4=judgement_gpt5_4.json \
        --judge sonnet4_6=judgement_sonnet4_6.json
"""

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = (
    "contamination",
    "disallowed_model",
    "justification_contamination",
    "justification_disallowed_model",
)


def read_judgement(path: Path) -> dict:
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
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise SystemExit(f"ERROR: {path} missing fields: {', '.join(missing)}")
    return data


def aggregate(judgements: dict[str, dict]) -> dict:
    contamination = any(bool(j["contamination"]) for j in judgements.values())
    disallowed_model = any(bool(j["disallowed_model"]) for j in judgements.values())

    def collect(field: str) -> str:
        return "\n\n".join(f"[{name}] {data[field]}" for name, data in judgements.items())

    return {
        "contamination": contamination,
        "disallowed_model": disallowed_model,
        "justification_contamination": collect("justification_contamination"),
        "justification_disallowed_model": collect("justification_disallowed_model"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Per-judge JSON file, e.g. gpt5_4=judgement_gpt5_4.json. May be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the aggregated judge_result.json",
    )
    args = parser.parse_args()

    judgements: dict[str, dict] = {}
    for spec in args.judge:
        if "=" not in spec:
            raise SystemExit(f"--judge value must be NAME=PATH, got: {spec!r}")
        name, path_str = spec.split("=", 1)
        judgements[name] = read_judgement(Path(path_str))

    result = aggregate(judgements)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Aggregated {len(judgements)} judge(s) -> {args.output} "
        f"(contamination={result['contamination']}, "
        f"disallowed_model={result['disallowed_model']})"
    )


if __name__ == "__main__":
    main()

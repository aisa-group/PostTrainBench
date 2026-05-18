"""Parse an agent or judge trace into a human-readable transcript.

Picks the right per-agent parser by substring-matching the agent name against
{claude, codex, gemini, opencode}. If the name matches zero keys, the input is
copied verbatim (preserves the historical fallback for agents like glm5 and
qwen3max that don't produce a structured trace). If the name matches more than
one key, the script errors out instead of guessing.

After parsing, also drops sanitized companions for both the raw input and the
parsed output (`<stem>_sanitized<ext>` next to each), with .env API key values
redacted.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import claude_parser
import codex_parser
import gemini_parser
import opencode_parser
import sanitize_trace

PARSERS = {
    "claude": claude_parser.parse,
    "codex": codex_parser.parse,
    "gemini": gemini_parser.parse,
    "opencode": opencode_parser.parse,
}


def select_parser(agent_name: str):
    matches = [key for key in PARSERS if key in agent_name]
    if len(matches) > 1:
        raise SystemExit(
            f"Agent name '{agent_name}' matches multiple parser keys: {matches}. "
            "Refusing to dispatch — rename the agent or update PARSERS."
        )
    if not matches:
        return None
    return PARSERS[matches[0]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an agent or judge trace into a human-readable transcript."
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent or judge name; used to pick the right parser via substring match.",
    )
    parser.add_argument("input", type=Path, help="Path to the trace input file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Destination text file.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    parse_fn = select_parser(args.agent)
    if parse_fn is None:
        print(
            f"No structured parser for agent '{args.agent}'; copying raw trace to {args.output}"
        )
        shutil.copyfile(args.input, args.output)
    else:
        parse_fn(args.input, args.output)
        print(f"Wrote parsed trace to {args.output}")

    secrets = sanitize_trace.load_api_key_secrets()
    sanitize_trace.sanitize_file(
        args.input, sanitize_trace.sanitized_path(args.input), secrets
    )
    sanitize_trace.sanitize_file(
        args.output, sanitize_trace.sanitized_path(args.output), secrets
    )
    print(f"Wrote sanitized companions next to input and output ({len(secrets)} keys)")


if __name__ == "__main__":
    main()

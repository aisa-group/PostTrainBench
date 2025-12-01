#!/usr/bin/env python3
"""Pretty-print Gemini CLI stream JSONL files."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Gemini CLI --output-format stream-json .jsonl file into a "
            "human-readable text report."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input JSONL file produced by gemini CLI",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Destination text file. Defaults to <input>.parsed.txt in the same "
            "directory."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the parsed output to stdout instead of writing a file.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ""
    if suffix:
        return input_path.with_suffix(f"{suffix}.parsed.txt")
    return input_path.with_name(f"{input_path.name}.parsed.txt")


def format_event(index: int, data: dict[str, Any]) -> str:
    lines: list[str] = []
    method = data.get("method", "<unknown>")
    lines.append(f"=== Event {index} | method: {method} ===")

    response = data.get("response")
    if isinstance(response, list):
        for chunk_index, chunk in enumerate(response, 1):
            lines.extend(format_chunk(chunk, chunk_index))
    elif isinstance(response, dict):
        lines.extend(format_chunk(response, None))
    elif response is not None:
        lines.append(indent(f"Response: {response!r}", 1))

    error = data.get("error")
    if error:
        lines.append(indent(f"Error: {json.dumps(error, ensure_ascii=False)}", 1))

    return "\n".join(lines)


def format_chunk(chunk: dict[str, Any], chunk_index: int | None) -> list[str]:
    lines: list[str] = []
    prefix = f"  Chunk {chunk_index}:" if chunk_index is not None else "  Chunk:"
    lines.append(prefix)

    if candidates := chunk.get("candidates"):
        for candidate_index, candidate in enumerate(candidates, 0):
            lines.extend(format_candidate(candidate, candidate_index))

    if usage := chunk.get("usageMetadata"):
        lines.append(indent(format_usage(usage), 2))

    other_keys = [k for k in chunk.keys() if k not in {"candidates", "usageMetadata"}]
    for key in other_keys:
        value = chunk[key]
        lines.append(indent(f"{key}: {json.dumps(value, ensure_ascii=False)}", 2))

    return lines


def format_candidate(candidate: dict[str, Any], index: int) -> list[str]:
    lines: list[str] = []
    role = candidate.get("content", {}).get("role") or candidate.get("role", "model")
    finish = candidate.get("finishReason")
    finish_suffix = f", finish={finish}" if finish else ""
    lines.append(indent(f"Candidate {index} ({role}{finish_suffix})", 2))

    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    for part in parts:
        lines.extend(format_part(part))

    return lines


def format_part(part: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    if text := part.get("text"):
        label = "Thought" if part.get("thought") else "Text"
        lines.append(indent(f"{label}:", 3))
        lines.append(indent(text.rstrip(), 4))
    if fn_call := part.get("functionCall"):
        lines.extend(format_function_call(fn_call))
    if fn_resp := part.get("functionResponse"):
        lines.extend(format_function_response(fn_resp))
    if inline := part.get("inlineData"):
        desc = inline.get("mimeType", "inlineData")
        lines.append(indent(f"Inline data ({desc})", 3))
    if "thoughtSignature" in part and not part.get("thought"):
        lines.append(indent("Thought signature present", 3))

    misc_keys = {
        k
        for k in part.keys()
        if k
        not in {
            "text",
            "thought",
            "functionCall",
            "functionResponse",
            "inlineData",
            "thoughtSignature",
        }
    }
    for key in sorted(misc_keys):
        lines.append(indent(f"{key}: {json.dumps(part[key], ensure_ascii=False)}", 3))

    return lines


def format_function_call(fn_call: dict[str, Any]) -> list[str]:
    name = fn_call.get("name", "<unknown>")
    args = fn_call.get("args", {})
    lines = [indent(f"Tool call ({name}):", 3)]

    command = None
    if isinstance(args, dict):
        cmd_value = args.get("command")
        if isinstance(cmd_value, list):
            command = " ".join(shlex.quote(str(token)) for token in cmd_value)
    if command:
        lines.append(indent(command, 4))

    lines.append(indent(json.dumps(args, indent=2, ensure_ascii=False), 4))
    return lines


def format_function_response(fn_resp: dict[str, Any]) -> list[str]:
    name = fn_resp.get("name", "unknown")
    response = fn_resp.get("response", {})
    lines = [indent(f"Tool response ({name}):", 3)]
    if isinstance(response, dict) and "output" in response:
        output = response["output"]
        if isinstance(output, str):
            lines.append(indent("Output:", 4))
            lines.append(indent(output.rstrip(), 5))
        else:
            lines.append(indent(json.dumps(output, ensure_ascii=False), 4))
    else:
        lines.append(indent(json.dumps(response, ensure_ascii=False), 4))
    return lines


def format_usage(usage: dict[str, Any]) -> str:
    summary_bits: list[str] = []
    for key in (
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
        "thoughtsTokenCount",
    ):
        if key in usage:
            summary_bits.append(f"{key}={usage[key]}")
    if summary_bits:
        return "Usage: " + ", ".join(summary_bits)
    return "Usage: " + json.dumps(usage, ensure_ascii=False)


def indent(text: str, level: int) -> str:
    pad = "  " * level
    return "\n".join(pad + line if line else pad for line in text.splitlines())


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = args.output or default_output_path(input_path)

    formatted_events: list[str] = []
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Failed to parse line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(event, dict):
                raise SystemExit(f"Line {line_number} is not a JSON object.")
            formatted_events.append(format_event(len(formatted_events) + 1, event))

    output_text = "\n\n".join(formatted_events) + "\n"

    if args.stdout:
        print(output_text)
    else:
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote parsed report to {output_path}")


if __name__ == "__main__":
    main()

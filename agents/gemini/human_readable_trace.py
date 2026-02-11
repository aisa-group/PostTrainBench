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


def pretty_format_json(obj: Any, indent_level: int = 0) -> str:
    """Format JSON with actual newlines preserved in strings."""
    indent_str = "  " * indent_level
    next_indent = "  " * (indent_level + 1)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for key, value in obj.items():
            formatted_value = pretty_format_json(value, indent_level + 1)
            # Handle multi-line values
            if '\n' in formatted_value and not formatted_value.startswith('{') and not formatted_value.startswith('['):
                # Multi-line string value - format specially
                first_line = formatted_value.split('\n')[0]
                rest_lines = '\n'.join(formatted_value.split('\n')[1:])
                items.append(f'{next_indent}"{key}": {first_line}\n{rest_lines}')
            else:
                items.append(f'{next_indent}"{key}": {formatted_value}')
        return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        items = []
        for item in obj:
            formatted_item = pretty_format_json(item, indent_level + 1)
            items.append(f"{next_indent}{formatted_item}")
        return "[\n" + ",\n".join(items) + "\n" + indent_str + "]"
    elif isinstance(obj, str):
        # For strings with newlines, output them directly with preserved newlines
        if '\n' in obj:
            # Don't use JSON encoding for multi-line strings
            # Just output the raw string with proper indenting on each line
            return obj  # The indent() function will handle line-by-line indenting
        else:
            # Single-line strings use normal JSON encoding
            return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif obj is None:
        return "null"
    else:
        return str(obj)


def format_unparsable_line(index: int, line: str, error_msg: str = "") -> str:
    lines = [f"=== Event {index} | NOT PARSABLE ==="]
    if error_msg:
        lines.append(f"  Error: {error_msg}")
    lines.append("  Raw line:")
    lines.append(f"    {line}")
    return "\n".join(lines)


def format_event(index: int, data: dict[str, Any]) -> str:
    method = data.get("method")
    event_type = data.get("type")

    header_bits: list[str] = []
    if method:
        header_bits.append(f"method: {method}")
    if event_type:
        header_bits.append(f"type: {event_type}")
    if timestamp := data.get("timestamp"):
        header_bits.append(f"ts: {timestamp}")

    header_extra = " | ".join(header_bits) if header_bits else "<unknown>"
    lines: list[str] = [f"=== Event {index} | {header_extra} ==="]

    if method:
        lines.extend(format_method_event(data))
    elif event_type:
        lines.extend(format_stream_event(event_type, data))
    else:
        lines.append(indent(pretty_format_json(data, 0), 1))

    return "\n".join(lines)


def format_method_event(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
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
        lines.append(indent(f"Error: {pretty_format_json(error, 0)}", 1))

    stats = data.get("stats")
    if stats:
        lines.append(indent("Stats:", 1))
        lines.append(indent(pretty_format_json(stats, 0), 2))

    return lines


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
        lines.append(indent(f"{key}: {pretty_format_json(value, 0)}", 2))

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
        lines.append(indent(f"{key}: {pretty_format_json(part[key], 0)}", 3))

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

    lines.append(indent(pretty_format_json(args, 0), 4))
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
            lines.append(indent(pretty_format_json(output, 0), 4))
    else:
        lines.append(indent(pretty_format_json(response, 0), 4))
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
    return "Usage: " + pretty_format_json(usage, 0)


def format_stream_event(event_type: str, data: dict[str, Any]) -> list[str]:
    event_type = event_type.lower()
    lines: list[str] = []

    timestamp = data.get("timestamp")
    if timestamp:
        lines.append(indent(f"Timestamp: {timestamp}", 1))

    if event_type == "init":
        lines.append(indent(f"Session: {data.get('session_id', '<unknown>')}", 1))
        lines.append(indent(f"Model: {data.get('model', '<unknown>')}", 1))
    elif event_type == "message":
        role = data.get("role", "assistant")
        delta_suffix = " (delta)" if data.get("delta") else ""
        lines.append(indent(f"Role: {role}{delta_suffix}", 1))
        content = data.get("content")
        if content:
            lines.append(indent("Content:", 1))
            lines.append(indent(content.rstrip(), 2))
    elif event_type == "tool_use":
        tool_name = data.get("tool_name", "<unknown>")
        lines.append(indent(f"Tool call: {tool_name} [{data.get('tool_id', '-')}]", 1))
        params = data.get("parameters")
        if params:
            command = extract_command_from_params(params)
            if command:
                lines.append(indent(f"Command: {command}", 2))
            lines.append(indent("Parameters:", 1))
            lines.append(indent(pretty_format_json(params, 0), 2))
    elif event_type == "tool_result":
        status = data.get("status", "unknown")
        lines.append(
            indent(
                f"Tool result for {data.get('tool_id', '<unknown>')} ({status})",
                1,
            )
        )
        if output := data.get("output"):
            lines.append(indent("Output:", 1))
            lines.append(indent(output.rstrip(), 2))
        if error := data.get("error"):
            lines.append(indent("Error:", 1))
            lines.append(indent(pretty_format_json(error, 0), 2))
    elif event_type == "error":
        severity = data.get("severity", "error").upper()
        message = data.get("message", "<no message>")
        lines.append(indent(f"{severity}: {message}", 1))
    elif event_type == "result":
        status = data.get("status", "unknown")
        lines.append(indent(f"Status: {status}", 1))
        if stats := data.get("stats"):
            lines.append(indent(format_stream_stats(stats), 1))
    else:
        # Fallback to raw JSON for unrecognized event types
        lines.append(indent(pretty_format_json(data, 0), 1))

    return lines


def extract_command_from_params(params: Any) -> str | None:
    if isinstance(params, dict):
        cmd_value = params.get("command")
        if isinstance(cmd_value, list):
            return " ".join(shlex.quote(str(token)) for token in cmd_value)
    return None


def format_stream_stats(stats: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "tool_calls",
        "duration_ms",
    ):
        if key in stats:
            pieces.append(f"{key}={stats[key]}")
    if pieces:
        return "Stats: " + ", ".join(pieces)
    return "Stats: " + pretty_format_json(stats, 0)


def indent(text: str, level: int) -> str:
    pad = "  " * level
    return "\n".join(pad + line if line else pad for line in text.splitlines())


def is_delta_message(event: dict[str, Any]) -> bool:
    """Check if this event is a streaming delta message."""
    return (
        event.get("type") == "message"
        and event.get("delta") is True
        and "content" in event
    )


def format_consolidated_deltas(
    index: int, deltas: list[dict[str, Any]]
) -> str:
    """Format a sequence of delta messages as a single consolidated event."""
    if not deltas:
        return ""

    first = deltas[0]
    last = deltas[-1]
    role = first.get("role", "assistant")

    # Combine all content fragments
    combined_content = "".join(d.get("content", "") for d in deltas)

    # Build header
    header_bits = [f"type: message (consolidated from {len(deltas)} deltas)"]
    if first_ts := first.get("timestamp"):
        last_ts = last.get("timestamp")
        if last_ts and last_ts != first_ts:
            header_bits.append(f"ts: {first_ts} -> {last_ts}")
        else:
            header_bits.append(f"ts: {first_ts}")

    header_extra = " | ".join(header_bits)
    lines: list[str] = [f"=== Event {index} | {header_extra} ==="]
    lines.append(indent(f"Role: {role}", 1))
    if combined_content:
        lines.append(indent("Content:", 1))
        lines.append(indent(combined_content.rstrip(), 2))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = args.output or default_output_path(input_path)

    formatted_events: list[str] = []
    pending_deltas: list[dict[str, Any]] = []
    current_delta_role: str | None = None

    def flush_deltas() -> None:
        """Flush any accumulated delta messages."""
        nonlocal pending_deltas, current_delta_role
        if pending_deltas:
            formatted_events.append(
                format_consolidated_deltas(len(formatted_events) + 1, pending_deltas)
            )
            pending_deltas = []
            current_delta_role = None

    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                # Flush any pending deltas before outputting unparsable line
                flush_deltas()
                formatted_events.append(
                    format_unparsable_line(len(formatted_events) + 1, stripped, exc.msg)
                )
                continue

            if not isinstance(event, dict):
                flush_deltas()
                formatted_events.append(
                    format_unparsable_line(
                        len(formatted_events) + 1,
                        stripped,
                        "Parsed JSON is not an object"
                    )
                )
                continue

            # Check if this is a delta message that can be consolidated
            if is_delta_message(event):
                event_role = event.get("role", "assistant")
                # If role changes, flush previous deltas first
                if current_delta_role is not None and event_role != current_delta_role:
                    flush_deltas()
                pending_deltas.append(event)
                current_delta_role = event_role
            else:
                # Non-delta event: flush any pending deltas first
                flush_deltas()
                formatted_events.append(format_event(len(formatted_events) + 1, event))

    # Flush any remaining deltas at end of file
    flush_deltas()

    output_text = "\n\n".join(formatted_events) + "\n"

    if args.stdout:
        print(output_text)
    else:
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote parsed report to {output_path}")


if __name__ == "__main__":
    main()

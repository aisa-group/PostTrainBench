#!/usr/bin/env python3
"""Format OpenCode stream-json output into a readable transcript."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenCode --format json logs into a human-readable transcript."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the JSON .jsonl file produced by OpenCode",
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
            if '\n' in formatted_value and not formatted_value.startswith('{') and not formatted_value.startswith('['):
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
        if '\n' in obj:
            return obj
        else:
            return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif obj is None:
        return "null"
    else:
        return str(obj)


def indent(text: str, level: int) -> str:
    """Indent text by the given level (2 spaces per level)."""
    pad = "  " * level
    return "\n".join(pad + line if line else pad for line in text.splitlines())


def format_timestamp(ts: int | None) -> str:
    """Format a timestamp (milliseconds) into a readable string."""
    if ts is None:
        return ""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_unparsable_line(index: int, line: str, error_msg: str = "") -> str:
    lines = [f"=== Event {index} | NOT PARSABLE ==="]
    if error_msg:
        lines.append(f"  Error: {error_msg}")
    lines.append("  Raw line:")
    lines.append(f"    {line[:500]}{'...' if len(line) > 500 else ''}")
    return "\n".join(lines)


def format_tool_use(event: dict[str, Any], index: int) -> str:
    """Format a tool_use event."""
    part = event.get("part", {})
    tool_name = part.get("tool", "unknown")
    state = part.get("state", {})
    status = state.get("status", "unknown")

    timestamp = format_timestamp(event.get("timestamp"))
    header = f"=== Event {index} | type: tool_use | tool: {tool_name} | status: {status} ==="
    if timestamp:
        header = f"=== Event {index} | type: tool_use | tool: {tool_name} | status: {status} | ts: {timestamp} ==="

    lines = [header]

    # Tool title
    title = state.get("title", "")
    if title:
        lines.append(indent(f"Title: {title}", 1))

    # Tool input
    tool_input = state.get("input", {})
    if tool_input:
        lines.append(indent("Input:", 1))
        # Special handling for common tools
        if tool_name == "bash" and "command" in tool_input:
            lines.append(indent(f"$ {tool_input['command']}", 2))
        elif tool_name in ("read", "write", "edit", "glob", "grep") and "file_path" in tool_input:
            lines.append(indent(f"File: {tool_input['file_path']}", 2))
            for k, v in tool_input.items():
                if k != "file_path":
                    lines.append(indent(f"{k}: {v}", 2))
        else:
            lines.append(indent(pretty_format_json(tool_input), 2))

    # Tool output (for completed tools)
    if status == "completed":
        output = state.get("output", "")
        if output:
            lines.append(indent("Output:", 1))
            # Truncate very long outputs
            if len(output) > 2000:
                output = output[:2000] + "\n... [truncated]"
            lines.append(indent(output.rstrip(), 2))

    # Error (for error status)
    if status == "error":
        error = state.get("error", "")
        if error:
            lines.append(indent("Error:", 1))
            lines.append(indent(error, 2))

    # Timing info
    time_info = state.get("time", {})
    if time_info:
        start = time_info.get("start")
        end = time_info.get("end")
        if start and end:
            duration_ms = end - start
            lines.append(indent(f"Duration: {duration_ms}ms", 1))

    return "\n".join(lines)


def format_text(event: dict[str, Any], index: int) -> str:
    """Format a text event (assistant response)."""
    part = event.get("part", {})
    text = part.get("text", "")

    timestamp = format_timestamp(event.get("timestamp"))
    header = f"=== Event {index} | type: text ==="
    if timestamp:
        header = f"=== Event {index} | type: text | ts: {timestamp} ==="

    lines = [header]
    lines.append(indent("Assistant:", 1))
    lines.append(indent(text.rstrip(), 2))

    return "\n".join(lines)


def format_step_start(event: dict[str, Any], index: int) -> str:
    """Format a step_start event."""
    timestamp = format_timestamp(event.get("timestamp"))
    header = f"=== Event {index} | type: step_start ==="
    if timestamp:
        header = f"=== Event {index} | type: step_start | ts: {timestamp} ==="
    return header


def format_step_finish(event: dict[str, Any], index: int) -> str:
    """Format a step_finish event."""
    part = event.get("part", {})
    reason = part.get("reason", "")
    cost = part.get("cost", 0)
    tokens = part.get("tokens", {})

    timestamp = format_timestamp(event.get("timestamp"))
    header = f"=== Event {index} | type: step_finish ==="
    if timestamp:
        header = f"=== Event {index} | type: step_finish | ts: {timestamp} ==="

    lines = [header]

    if reason:
        lines.append(indent(f"Reason: {reason}", 1))

    if cost:
        lines.append(indent(f"Cost: ${cost:.6f}", 1))

    if tokens:
        input_tokens = tokens.get("input", 0)
        output_tokens = tokens.get("output", 0)
        reasoning_tokens = tokens.get("reasoning", 0)
        cache = tokens.get("cache", {})
        cache_read = cache.get("read", 0)
        cache_write = cache.get("write", 0)

        token_parts = [f"input={input_tokens}", f"output={output_tokens}"]
        if reasoning_tokens:
            token_parts.append(f"reasoning={reasoning_tokens}")
        if cache_read or cache_write:
            token_parts.append(f"cache_read={cache_read}")
            token_parts.append(f"cache_write={cache_write}")

        lines.append(indent(f"Tokens: {', '.join(token_parts)}", 1))

    return "\n".join(lines)


def format_error(event: dict[str, Any], index: int) -> str:
    """Format an error event."""
    error = event.get("error", {})

    timestamp = format_timestamp(event.get("timestamp"))
    header = f"=== Event {index} | type: error ==="
    if timestamp:
        header = f"=== Event {index} | type: error | ts: {timestamp} ==="

    lines = [header]

    error_name = error.get("name", "Unknown")
    lines.append(indent(f"Error Type: {error_name}", 1))

    if "data" in error:
        data = error["data"]
        if isinstance(data, dict):
            if "message" in data:
                lines.append(indent(f"Message: {data['message']}", 1))
            else:
                lines.append(indent(pretty_format_json(data), 1))
        else:
            lines.append(indent(str(data), 1))

    return "\n".join(lines)


def format_event(index: int, event: dict[str, Any]) -> str:
    """Format a single event based on its type."""
    event_type = event.get("type", "unknown")

    if event_type == "tool_use":
        return format_tool_use(event, index)
    elif event_type == "text":
        return format_text(event, index)
    elif event_type == "step_start":
        return format_step_start(event, index)
    elif event_type == "step_finish":
        return format_step_finish(event, index)
    elif event_type == "error":
        return format_error(event, index)
    else:
        # Unknown event type - output as JSON
        timestamp = format_timestamp(event.get("timestamp"))
        header = f"=== Event {index} | type: {event_type} ==="
        if timestamp:
            header = f"=== Event {index} | type: {event_type} | ts: {timestamp} ==="
        return f"{header}\n{indent(pretty_format_json(event), 1)}"


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
                formatted_events.append(
                    format_unparsable_line(len(formatted_events) + 1, stripped, exc.msg)
                )
                continue

            if not isinstance(event, dict):
                formatted_events.append(
                    format_unparsable_line(
                        len(formatted_events) + 1,
                        stripped,
                        "Parsed JSON is not an object"
                    )
                )
                continue

            formatted_events.append(format_event(len(formatted_events) + 1, event))

    output_text = "\n\n".join(formatted_events) + "\n"

    if args.stdout:
        print(output_text)
    else:
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote parsed report to {output_path}")


if __name__ == "__main__":
    main()

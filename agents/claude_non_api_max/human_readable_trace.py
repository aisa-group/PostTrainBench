#!/usr/bin/env python3
"""Format Claude Code stream-json output into a readable transcript."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Claude Code --output-format stream-json logs into a human-readable transcript."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to the stream-json .jsonl file (use '-' for stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path (defaults to stdout)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Maximum line width for wrapping text blocks (0 to disable wrapping).",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Append any unhandled events as raw JSON for debugging.",
    )
    return parser.parse_args()


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
            return obj  # The indent_block() function will handle line-by-line indenting
        else:
            # Single-line strings use normal JSON encoding
            return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif obj is None:
        return "null"
    else:
        return str(obj)


class TranscriptFormatter:
    def __init__(self, width: int, include_raw: bool = False) -> None:
        self.width = width
        self.include_raw = include_raw
        self.lines: List[str] = []
        self.event_index = 0
        self.turn_counters = {"assistant": 0, "user": 0}
        self.tool_call_meta: Dict[str, Dict[str, Any]] = {}

    def process_events(self, events: Iterable[Tuple[int, Dict[str, Any]]]) -> None:
        for line_no, event in events:
            self.event_index += 1
            handler = getattr(self, f"handle_{event.get('type')}", None)
            if handler:
                handler(event)
            else:
                self._handle_unknown(event, line_no)

    def handle_system(self, event: Dict[str, Any]) -> None:
        subtype = event.get("subtype") or "info"
        if subtype == "init":
            session_id = event.get("session_id", "unknown-session")
            self.lines.append(f"Session start — {session_id}")
            model = event.get("model") or event.get("settings", {}).get("model")
            if model:
                self.lines.append(f"  Model: {model}")
            tools = event.get("tools") or event.get("allowed_tools")
            if tools:
                tool_names = ", ".join(
                    tool.get("name", "unknown")
                    if isinstance(tool, dict)
                    else str(tool)
                    for tool in tools
                )
                self.lines.append(f"  Tools: {tool_names}")
            cwd = event.get("cwd") or event.get("working_directory")
            if cwd:
                self.lines.append(f"  Working dir: {cwd}")
            self.lines.append("")
        else:
            self.lines.append(f"System event — {subtype}")
            self.lines.append(indent_block(json_dumps_clean(event, skip_keys={"type"}), indent="  "))
            self.lines.append("")

    def handle_assistant(self, event: Dict[str, Any]) -> None:
        self._handle_message(event)

    def handle_user(self, event: Dict[str, Any]) -> None:
        self._handle_message(event)

    def handle_result(self, event: Dict[str, Any]) -> None:
        subtype = event.get("subtype") or "summary"
        self.lines.append(f"Result — {subtype}")
        payload = {k: v for k, v in event.items() if k not in {"type", "subtype"}}
        if payload:
            self.lines.append(indent_block(json_dumps_clean(payload), indent="  "))
        self.lines.append("")

    def _handle_message(self, event: Dict[str, Any]) -> None:
        message = event.get("message") or {}
        role = message.get("role") or event.get("type")
        role_key = "assistant" if role == "assistant" else "user"
        self.turn_counters[role_key] = self.turn_counters.get(role_key, 0) + 1
        turn_number = self.turn_counters[role_key]
        header = f"{role.title()} — turn {turn_number}"
        self.lines.append(header)

        for block in message.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    # Text already has real newlines from json.loads()
                    self.lines.append(indent_block(text, indent="  ", width=self.width))
            elif block_type == "tool_use":
                self._handle_tool_use(block)
            elif block_type == "tool_result":
                self._handle_tool_result(block)
            else:
                self.lines.append(f"  [{block_type or 'unknown'} block]")
                self.lines.append(
                    indent_block(json_dumps_clean(block, skip_keys={"type"}), indent="    ")
                )

        self.lines.append("")

    def _handle_tool_use(self, block: Dict[str, Any]) -> None:
        tool_id = block.get("id", "unknown-tool")
        name = block.get("name", "tool")
        input_payload = block.get("input")
        self.tool_call_meta[tool_id] = {"name": name}
        self.lines.append(f"  Tool call — {name} ({tool_id})")
        formatted_input = format_tool_input(input_payload)
        self.lines.append(indent_block(formatted_input, indent="    ", width=self.width))

    def _handle_tool_result(self, block: Dict[str, Any]) -> None:
        tool_id = block.get("tool_use_id", "unknown-tool")
        tool_name = self.tool_call_meta.get(tool_id, {}).get("name", "tool")
        self.lines.append(f"  Tool result — {tool_name} ({tool_id})")
        formatted_result = format_tool_result(block)
        # Text already has real newlines from json.loads()
        self.lines.append(indent_block(formatted_result, indent="    ", width=self.width))

    def _handle_unknown(self, event: Dict[str, Any], line_no: int) -> None:
        if not self.include_raw:
            return
        self.lines.append(
            f"Unhandled event type '{event.get('type')}' on source line {line_no}:"
        )
        self.lines.append(indent_block(json_dumps_clean(event), indent="  "))
        self.lines.append("")

    def render(self) -> str:
        return "\n".join(line.rstrip() for line in self.lines).rstrip() + "\n"


def indent_block(text: str, indent: str = "  ", width: int | None = None) -> str:
    text = text or ""
    if width is None or width <= 0:
        return "\n".join(f"{indent}{line}" if line else indent.rstrip() for line in text.splitlines())

    wrapped_lines: List[str] = []
    for line in text.splitlines() or [""]:
        if not line:
            wrapped_lines.append(indent.rstrip())
            continue
        if len(line) <= width:
            wrapped_lines.append(f"{indent}{line}")
        else:
            wrapped_lines.extend(wrap_line(line, indent, width))
    return "\n".join(wrapped_lines)


def wrap_line(line: str, indent: str, width: int) -> List[str]:
    import textwrap

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapper.wrap(line)


def json_dumps_clean(data: Any, skip_keys: set[str] | None = None) -> str:
    """Format JSON with newlines preserved in strings."""
    if skip_keys:
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in skip_keys}
    return pretty_format_json(data, indent_level=0)


def format_tool_input(payload: Any) -> str:
    if isinstance(payload, dict):
        command = payload.get("command") or payload.get("code")
        if isinstance(command, str) and len(payload) <= 2:
            prefix = "$" if "command" in payload else "python"
            # Command already has real newlines from json.loads()
            return f"{prefix} {command.strip()}" if command.strip() else prefix
    return json_dumps_clean(payload)


def format_tool_result(block: Dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(item.get("text", ""))
                elif item.get("type") == "json":
                    chunks.append(json_dumps_clean(item.get("json")))
                else:
                    chunks.append(json_dumps_clean(item))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks) if chunks else json_dumps_clean(content)
    if isinstance(content, (str, bytes)):
        return content.decode() if isinstance(content, bytes) else content
    if "output" in block:
        return str(block.get("output"))
    return json_dumps_clean(block)


def load_events(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if path == "-":
        lines_iter = enumerate(sys.stdin, 1)
    else:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        lines_iter = enumerate(file_path.open("r", encoding="utf-8"), 1)

    for line_no, raw in lines_iter:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            yield line_no, json.loads(stripped)
        except json.JSONDecodeError as exc:
            # Output unparsable lines
            print(f"NOT PARSABLE (line {line_no}): {exc}", file=sys.stderr)
            print(f"  Raw line: {stripped}", file=sys.stderr)
            continue


def main() -> None:
    args = parse_args()
    formatter = TranscriptFormatter(width=args.width, include_raw=args.include_raw)
    events = list(load_events(args.input_path))
    formatter.process_events(events)
    output_text = formatter.render()

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()

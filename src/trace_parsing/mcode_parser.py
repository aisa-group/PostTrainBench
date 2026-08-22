"""Pretty-print MiniMax Code ``mcode exec --output-format stream-json`` traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _common import TIMESTAMP_PREFIX_RE, pretty_format_json


TOOL_STATUSES = {0: "pending", 1: "running", 2: "completed", 3: "failed"}


def indent(text: str, level: int) -> str:
    pad = "  " * level
    return "\n".join(pad + line if line else pad for line in text.splitlines())


def add_text(lines: list[str], label: str, value: Any) -> None:
    if value is None or value == "":
        return
    lines.append(indent(f"{label}:", 1))
    lines.append(indent(str(value).rstrip(), 2))


def format_tool_call(tool_call: dict[str, Any]) -> list[str]:
    name = tool_call.get("name", "unknown")
    tool_id = tool_call.get("id", "")
    raw_status = tool_call.get("status", "unknown")
    status = TOOL_STATUSES.get(raw_status, str(raw_status))
    header = f"Tool: {name}"
    if tool_id:
        header += f" | id: {tool_id}"
    header += f" | status: {status}"
    lines = [indent(header, 1)]

    if "input" in tool_call:
        lines.append(indent("Input:", 2))
        lines.append(indent(pretty_format_json(tool_call["input"]), 3))

    output = tool_call.get("output")
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, list):
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if any(texts):
                lines.append(indent("Output:", 2))
                lines.append(indent("\n".join(texts).rstrip(), 3))
        if "details" in output:
            lines.append(indent("Details:", 2))
            lines.append(indent(pretty_format_json(output["details"]), 3))
    elif output not in (None, ""):
        lines.append(indent("Output:", 2))
        lines.append(indent(str(output).rstrip(), 3))

    return lines


def format_payload(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if role := payload.get("role"):
        lines.append(indent(f"Role: {role}", 1))
    add_text(lines, "Thinking", payload.get("thinking"))
    add_text(lines, "Content", payload.get("content"))
    if finish_reason := payload.get("finishReason"):
        lines.append(indent(f"Finish reason: {finish_reason}", 1))

    for tool_call in payload.get("toolCalls") or []:
        if isinstance(tool_call, dict):
            lines.extend(format_tool_call(tool_call))

    usage = payload.get("usage")
    if isinstance(usage, dict):
        ordered_keys = (
            "totalTokens",
            "inputTokens",
            "outputTokens",
            "cacheReadTokens",
            "cacheWriteTokens",
            "requestDurationMs",
        )
        bits = [f"{key}={usage[key]}" for key in ordered_keys if key in usage]
        if bits:
            lines.append(indent(f"Usage: {', '.join(bits)}", 1))
    return lines


def format_event(index: int, event: dict[str, Any], wall_ts: str | None = None) -> str:
    event_type = str(event.get("type", "unknown"))
    header_bits = [f"type: {event_type}"]
    if status := event.get("status"):
        header_bits.append(f"status: {status}")
    if event_type == "generic" and (generic_type := event.get("eventType")):
        header_bits.append(f"eventType: {generic_type}")
    if wall_ts:
        header_bits.append(f"ts: {wall_ts}")
    lines = [f"=== Event {index} | {' | '.join(header_bits)} ==="]

    if event_type in {"delta", "message"}:
        payload = event.get("message", event)
        if isinstance(payload, dict):
            lines.extend(format_payload(payload))
    elif event_type == "exec.result":
        for key, label in (("sessionId", "Session"), ("turnId", "Turn"), ("model", "Model")):
            if value := event.get(key):
                lines.append(indent(f"{label}: {value}", 1))
        if "durationMs" in event:
            lines.append(indent(f"Duration: {event['durationMs']} ms", 1))
        add_text(lines, "Answer", event.get("answer"))
        add_text(lines, "Error", event.get("error"))
    elif event_type == "generic":
        if "data" in event:
            lines.append(indent("Data:", 1))
            lines.append(indent(pretty_format_json(event["data"]), 2))
    else:
        for key, label in (("sessionId", "Session"), ("turnId", "Turn"), ("messageId", "Message")):
            if value := event.get(key):
                lines.append(indent(f"{label}: {value}", 1))

    return "\n".join(lines)


def format_unparsable_line(index: int, line: str, error: str) -> str:
    return (
        f"=== Event {index} | NOT PARSABLE ===\n"
        f"  Error: {error}\n"
        f"  Raw:\n{indent(line, 2)}"
    )


def parse(input_path: Path, output_path: Path) -> None:
    formatted_events: list[str] = []
    with input_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped:
                continue

            wall_ts = None
            if ts_match := TIMESTAMP_PREFIX_RE.match(stripped):
                wall_ts = ts_match.group(1)
                stripped = stripped[ts_match.end():]

            try:
                event = json.loads(stripped)
                if not isinstance(event, dict):
                    raise ValueError("Parsed JSON is not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                formatted_events.append(
                    format_unparsable_line(len(formatted_events) + 1, stripped, str(exc))
                )
                continue

            formatted_events.append(
                format_event(len(formatted_events) + 1, event, wall_ts)
            )

    output_path.write_text("\n\n".join(formatted_events) + "\n", encoding="utf-8")

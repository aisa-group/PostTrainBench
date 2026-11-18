"""Format Qwen Code (`qwen` CLI) stream-json output into a readable transcript.

Qwen Code's stream-json events have the shape of Claude Code's (system / assistant / user / result events, with text,
thinking, tool_use and tool_result blocks), so the Claude Code formatter does most of the work. This parser adds two
things: the model's thinking is shown as text, and the lines that are not JSON events (solve.sh's resume markers,
qwen's stderr, the CLI update log) are kept instead of dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from _common import TIMESTAMP_PREFIX_RE
from claude_parser import TranscriptFormatter


class QwenCodeTranscriptFormatter(TranscriptFormatter):
    def _handle_message(self, event: Dict[str, Any]) -> None:
        message = event.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    blocks.append({"type": "text", "text": "[thinking]\n" + (block.get("thinking") or "")})
                else:
                    blocks.append(block)
            event = {**event, "message": {**message, "content": blocks}}
        super()._handle_message(event)

    def add_harness_line(self, line: str, wall_ts: str | None) -> None:
        prefix = f"[{wall_ts}] " if wall_ts else ""
        self.lines.append(f"{prefix}{line}")


def parse(input_path: Path, output_path: Path) -> None:
    formatter = QwenCodeTranscriptFormatter()
    with input_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, raw in enumerate(stream, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            wall_ts = None
            ts_match = TIMESTAMP_PREFIX_RE.match(stripped)
            if ts_match:
                wall_ts = ts_match.group(1)
                stripped = stripped[ts_match.end():]
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                event = None
            if not isinstance(event, dict) or "type" not in event:
                formatter.add_harness_line(stripped, wall_ts)
                continue
            if wall_ts:
                event["_wall_ts"] = wall_ts
            formatter.process_events([(line_no, event)])
    output_path.write_text(formatter.render(), encoding="utf-8")

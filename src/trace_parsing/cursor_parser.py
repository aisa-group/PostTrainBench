"""Format Cursor CLI (`cursor-agent`) stream-json output into a readable transcript.

Cursor emits events like:
    {"type":"system","subtype":"init","model":"Cursor Grok 4.5 High",...}
    {"type":"user","message":"{'role':'user','content':[...]}",...}
    {"type":"thinking","subtype":"delta","text":"..."}
    {"type":"thinking","subtype":"completed"}
    {"type":"assistant","message":"{'role':'assistant','content':[...]}",...}
    {"type":"tool_call","subtype":"started","call_id":"...","tool_call":"..."}
    {"type":"tool_call","subtype":"completed","call_id":"...","tool_call":"..."}
    {"type":"system","subtype":"task_notification","title":"...","status":"..."}
    {"type":"result","subtype":"success","usage":"...","result":"...","duration_ms":...}
    (plus transient retry/connection/interaction_query events)

`message` and `tool_call` payloads are Python-repr strings, not JSON — parsed
with ast.literal_eval and gracefully falling back to raw text.

Streaming `thinking/delta` events (often thousands per turn) are coalesced into
one block per matching `thinking/completed` so the transcript stays readable.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from _common import TIMESTAMP_PREFIX_RE, pretty_format_json
from claude_parser import indent_block, json_dumps_clean, load_events


def _try_parse(payload: Any) -> Any:
    """Cursor emits `message` and `tool_call` as Python-repr strings — try
    ast.literal_eval, then json.loads, else return as-is."""
    if not isinstance(payload, str):
        return payload
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(payload)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return payload


class CursorTranscriptFormatter:
    def __init__(self, width: int = 0) -> None:
        self.width = width
        self.lines: List[str] = []
        self.turn_counters = {"assistant": 0, "user": 0}
        self.thinking_buffer: List[str] = []
        self.pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self.retry_streak = 0
        self.connection_streak = 0

    # ---------- entry point ----------
    def process_events(self, events: Iterable[Tuple[int, Dict[str, Any]]]) -> None:
        for _line_no, event in events:
            t = event.get("type")
            s = event.get("subtype", "")
            handler_name = f"handle_{t}_{s}" if s else f"handle_{t}"
            handler = getattr(self, handler_name, None) or getattr(self, f"handle_{t}", None)
            if handler:
                handler(event)
            else:
                self._flush_streaks()
                self.lines.append(f"[unhandled {t}/{s or '-'} event]")
                self.lines.append(indent_block(json_dumps_clean(event, skip_keys={"type", "subtype"}), indent="  "))
                self.lines.append("")

    # ---------- system events ----------
    def handle_system_init(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        session_id = event.get("session_id", "unknown-session")
        self.lines.append(f"Session start — {session_id}")
        for label, key in (("Model", "model"), ("Working dir", "cwd"),
                           ("Permission mode", "permissionMode"),
                           ("Auth source", "apiKeySource")):
            if val := event.get(key):
                self.lines.append(f"  {label}: {val}")
        self.lines.append("")

    def handle_system_task_notification(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        status = event.get("status", "?")
        title = event.get("title", "")
        task_id = event.get("task_id", "")
        self.lines.append(f"Task notification — [{status}] {title} (id={task_id})")

    def handle_system(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        s = event.get("subtype", "info")
        self.lines.append(f"System event — {s}")
        payload = {k: v for k, v in event.items() if k not in {"type", "subtype", "session_id"}}
        if payload:
            self.lines.append(indent_block(json_dumps_clean(payload), indent="  "))
        self.lines.append("")

    # ---------- turn events ----------
    def handle_user(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        self._render_message_event(event, role="user")

    def handle_assistant(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        self._flush_thinking()
        self._render_message_event(event, role="assistant")

    def _render_message_event(self, event: Dict[str, Any], role: str) -> None:
        self.turn_counters[role] = self.turn_counters.get(role, 0) + 1
        turn_number = self.turn_counters[role]
        header = f"{role.title()} — turn {turn_number}"
        if wall_ts := event.get("_wall_ts"):
            header += f" | {wall_ts}"
        self.lines.append(header)

        message = _try_parse(event.get("message"))
        if isinstance(message, dict):
            for block in message.get("content", []) or []:
                self._render_message_block(block)
        elif isinstance(message, str) and message:
            self.lines.append(indent_block(message, indent="  ", width=self.width))
        self.lines.append("")

    def _render_message_block(self, block: Any) -> None:
        if not isinstance(block, dict):
            self.lines.append(indent_block(str(block), indent="  ", width=self.width))
            return
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                self.lines.append(indent_block(text, indent="  ", width=self.width))
        else:
            label = block_type or "unknown"
            self.lines.append(f"  [{label} block]")
            self.lines.append(indent_block(json_dumps_clean(block, skip_keys={"type"}), indent="    "))

    # ---------- thinking (streaming) ----------
    def handle_thinking_delta(self, event: Dict[str, Any]) -> None:
        # Silently accumulate; do NOT flush retry/connection streaks — they may
        # bracket a thinking burst. Flush explicitly on the next non-streaming event.
        text = event.get("text", "")
        if text:
            self.thinking_buffer.append(text)

    def handle_thinking_completed(self, _event: Dict[str, Any]) -> None:
        self._flush_thinking()

    def _flush_thinking(self) -> None:
        if not self.thinking_buffer:
            return
        text = "".join(self.thinking_buffer).strip()
        self.thinking_buffer.clear()
        if not text:
            return
        self.lines.append("  [thinking]")
        self.lines.append(indent_block(text, indent="    ", width=self.width))
        self.lines.append("")

    # ---------- tool calls ----------
    def handle_tool_call_started(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        self._flush_thinking()
        call_id = self._short_call_id(event.get("call_id", "?"))
        tool_call = _try_parse(event.get("tool_call"))
        tool_name, args = self._split_tool_call(tool_call)
        self.pending_tool_calls[call_id] = {"name": tool_name, "started": True}
        self.lines.append(f"  Tool call — {tool_name} ({call_id})")
        if args is not None:
            self.lines.append(indent_block(self._format_tool_args(tool_name, args), indent="    ", width=self.width))

    def handle_tool_call_completed(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        self._flush_thinking()
        call_id = self._short_call_id(event.get("call_id", "?"))
        tool_call = _try_parse(event.get("tool_call"))
        tool_name, _ = self._split_tool_call(tool_call)
        # find the result payload — cursor stashes it inside the top-level dict
        result = self._extract_tool_result(tool_call)
        self.lines.append(f"  Tool result — {tool_name} ({call_id})")
        if result is not None:
            self.lines.append(indent_block(self._format_tool_result(result), indent="    ", width=self.width))
        self.pending_tool_calls.pop(call_id, None)
        self.lines.append("")

    @staticmethod
    def _short_call_id(call_id: str) -> str:
        # cursor stuffs both a call-<uuid> and an fc_<hash> newline-joined
        return call_id.split("\n", 1)[0].strip() if isinstance(call_id, str) else str(call_id)

    @staticmethod
    def _split_tool_call(tool_call: Any) -> Tuple[str, Any]:
        """Cursor wraps tool calls like {"shellToolCall": {"args": {...}}} —
        pull the tool name and args out."""
        if not isinstance(tool_call, dict):
            return "tool", tool_call
        for key, inner in tool_call.items():
            if isinstance(inner, dict) and "args" in inner:
                return key, inner.get("args")
        return "tool", tool_call

    @staticmethod
    def _extract_tool_result(tool_call: Any) -> Any:
        if not isinstance(tool_call, dict):
            return None
        for _, inner in tool_call.items():
            if isinstance(inner, dict):
                for key in ("result", "output", "response"):
                    if key in inner:
                        return inner[key]
        return None

    def _format_tool_args(self, tool_name: str, args: Any) -> str:
        # Compact renders for common tool shapes — cursor's args dicts often
        # carry a lot of internal parser metadata (simpleCommands, parsingResult,
        # toolCallId, ...) that isn't useful in a human-readable trace.
        if isinstance(args, dict):
            if isinstance(cmd := args.get("command"), str):
                return f"$ {cmd.strip()}"
            for path_key in ("relativeWorkspacePath", "path", "targetFile"):
                if isinstance(p := args.get(path_key), str):
                    if isinstance(contents := args.get("contents"), str):
                        return f"write {p}\n{contents}"
                    return f"{tool_name.replace('ToolCall', '')} {p}"
            if isinstance(code := args.get("code"), str):
                return f"python\n{code.strip()}"
            if isinstance(q := args.get("query") or args.get("searchTerm"), str):
                return f"search: {q.strip()}"
        return json_dumps_clean(args)

    def _format_tool_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        return json_dumps_clean(result)

    # ---------- transient / low-signal events (coalesce into streaks) ----------
    def handle_retry(self, event: Dict[str, Any]) -> None:
        self.retry_streak += 1

    def handle_connection(self, event: Dict[str, Any]) -> None:
        self.connection_streak += 1

    def handle_interaction_query(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        self._flush_thinking()
        sub = event.get("subtype", "?")
        qtype = event.get("query_type", "?")
        self.lines.append(f"Interaction — {sub} ({qtype})")

    def _flush_streaks(self) -> None:
        if self.retry_streak:
            self.lines.append(f"  [network retry ×{self.retry_streak}]")
            self.retry_streak = 0
        if self.connection_streak:
            self.lines.append(f"  [connection blip ×{self.connection_streak}]")
            self.connection_streak = 0

    # ---------- final result ----------
    def handle_result(self, event: Dict[str, Any]) -> None:
        self._flush_streaks()
        self._flush_thinking()
        subtype = event.get("subtype", "summary")
        self.lines.append(f"Result — {subtype}")
        payload = {k: _try_parse(v) if k in ("usage",) else v
                   for k, v in event.items()
                   if k not in {"type", "subtype", "session_id", "_wall_ts"}}
        if payload:
            self.lines.append(indent_block(json_dumps_clean(payload), indent="  "))
        if wall_ts := event.get("_wall_ts"):
            self.lines.append(f"  (finished at {wall_ts})")
        self.lines.append("")

    # ---------- output ----------
    def render(self) -> str:
        self._flush_streaks()
        self._flush_thinking()
        return "\n".join(line.rstrip() for line in self.lines).rstrip() + "\n"


def parse(input_path: Path, output_path: Path) -> None:
    formatter = CursorTranscriptFormatter()
    events = list(load_events(input_path))
    formatter.process_events(events)
    output_path.write_text(formatter.render(), encoding="utf-8")

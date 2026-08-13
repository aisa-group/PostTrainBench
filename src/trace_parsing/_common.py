"""Shared helpers for the per-agent trace parsers."""

from __future__ import annotations

import json
import re
from typing import Any

TIMESTAMP_PREFIX_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\] ')


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
            if (
                "\n" in formatted_value
                and not formatted_value.startswith("{")
                and not formatted_value.startswith("[")
            ):
                first_line = formatted_value.split("\n")[0]
                rest_lines = "\n".join(formatted_value.split("\n")[1:])
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
        if "\n" in obj:
            return obj
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif obj is None:
        return "null"
    else:
        return str(obj)

"""Redact API key values (sourced from .env) out of a trace file.

Picks up every `*_API_KEY` and `*_TOKEN` entry (SECRET_NAME_SUFFIXES) in the
repo's .env file and replaces every occurrence of its value in the trace text
with `[REDACTED:<NAME>]`: both the literal in the file and, when it differs,
the value exported in the live environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
PLACEHOLDER_PREFIX = "your-"
MIN_VALUE_LEN = 8
# .env names whose values are secrets: API keys plus tokens (HF_TOKEN,
# CLAUDE_CODE_OAUTH_TOKEN, ...); dev_utils/extract_traces.py uses the same rule.
SECRET_NAME_SUFFIXES = ("_API_KEY", "_TOKEN")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_api_key_secrets(env_path: Path = DEFAULT_ENV_PATH) -> list[tuple[str, str]]:
    """Return (name, value) pairs for every *_API_KEY / *_TOKEN entry in .env.

    Both the literal value written in .env and the live environment value are
    returned when they differ: which one a run actually used depends on the
    tool (harbor's run_modal_task.sh prefers .env for HF_TOKEN, most tools
    prefer the shell), and either is a secret. Placeholders (`your-*`) and
    values shorter than MIN_VALUE_LEN are skipped.
    """
    if not env_path.exists():
        raise SystemExit(f".env file not found at {env_path}")

    secrets: list[tuple[str, str]] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, file_value = stripped.partition("=")
        name = name.strip()
        if not name.endswith(SECRET_NAME_SUFFIXES):
            continue
        for value in (_strip_quotes(file_value.strip()), os.environ.get(name, "")):
            if not value or value.startswith(PLACEHOLDER_PREFIX) or len(value) < MIN_VALUE_LEN:
                continue
            if (name, value) not in secrets:
                secrets.append((name, value))
    return secrets


def sanitize_text(text: str, secrets: list[tuple[str, str]]) -> str:
    # Replace longer values first so a secret that is a prefix of another
    # secret doesn't get partially redacted with the wrong label.
    for name, value in sorted(secrets, key=lambda kv: -len(kv[1])):
        text = text.replace(value, f"[REDACTED:{name}]")
    return text


def sanitized_path(path: Path) -> Path:
    """Return `<stem>_sanitized<suffix>` next to the given path."""
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}_sanitized{suffix}")


def sanitize_file(
    input_path: Path,
    output_path: Path,
    secrets: list[tuple[str, str]] | None = None,
) -> None:
    if secrets is None:
        secrets = load_api_key_secrets()
    text = input_path.read_text(encoding="utf-8")
    output_path.write_text(sanitize_text(text, secrets), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redact .env API key values from a trace file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Path to the .env file (default: {DEFAULT_ENV_PATH}).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    secrets = load_api_key_secrets(args.env_file)
    sanitize_file(args.input, args.output, secrets)
    print(f"Wrote sanitized trace to {args.output} ({len(secrets)} keys redacted)")


if __name__ == "__main__":
    main()

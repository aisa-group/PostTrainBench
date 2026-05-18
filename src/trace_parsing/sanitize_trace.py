"""Redact API key values (sourced from .env) out of a trace file.

Picks up every `*_API_KEY` entry in the repo's .env file, prefers the
live-environment value over the literal in the file, and replaces every
occurrence of the value in the trace text with `[REDACTED:<NAME>]`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
PLACEHOLDER_PREFIX = "your-"
MIN_VALUE_LEN = 8


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_api_key_secrets(env_path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Return name → value for every *_API_KEY entry in .env with a real value.

    Prefers the live environment value (so an override via the shell wins) and
    falls back to the literal value written in the .env file. Placeholders
    (`your-*`) and values shorter than MIN_VALUE_LEN are skipped.
    """
    if not env_path.exists():
        raise SystemExit(f".env file not found at {env_path}")

    secrets: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, file_value = stripped.partition("=")
        name = name.strip()
        if not name.endswith("_API_KEY"):
            continue
        value = os.environ.get(name) or _strip_quotes(file_value.strip())
        if not value or value.startswith(PLACEHOLDER_PREFIX):
            continue
        if len(value) < MIN_VALUE_LEN:
            continue
        secrets[name] = value
    return secrets


def sanitize_text(text: str, secrets: dict[str, str]) -> str:
    # Replace longer values first so a secret that is a prefix of another
    # secret doesn't get partially redacted with the wrong label.
    for name, value in sorted(secrets.items(), key=lambda kv: -len(kv[1])):
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
    secrets: dict[str, str] | None = None,
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

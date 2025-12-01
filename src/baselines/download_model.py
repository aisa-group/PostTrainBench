#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model and tokenizer.")
    parser.add_argument("--model", required=True, help="Model identifier on the Hugging Face Hub.")
    parser.add_argument("--output", required=True, help="Directory to store the downloaded model files.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision (tag, branch, or commit).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote code for custom model/tokenizer implementations.",
    )
    return parser.parse_args()


def save_metadata(target_dir: Path, metadata: dict[str, Any]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "baseline_download.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    target_dir = Path(args.output).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = target_dir / "config.json"
    if config_path.exists():
        print(f"Model already present at {target_dir}. Skipping download.")
        save_metadata(
            target_dir,
            {
                "model": args.model,
                "revision": args.revision,
                "skipped": True,
            },
        )
        return

    print(f"Loading tokenizer '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading model '{args.model}'...")
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Saving files to '{target_dir}'...")
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    save_metadata(
        target_dir,
        {
            "model": args.model,
            "revision": args.revision,
            "skipped": False,
        },
    )
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main()

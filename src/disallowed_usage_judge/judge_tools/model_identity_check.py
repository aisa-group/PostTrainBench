#!/usr/bin/env python3
"""
Model-identity oracle: compare a candidate model's config.json against the
reference config.json of the allowed base model.

Core architecture fields (model_type, hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads, intermediate_size, head_dim) are
invariant under fine-tuning. If they all match, the candidate is
architecturally a derivative of the allowed base model, and generic
tokenizer/transformers warnings mentioning other vendors (e.g. a Mistral
regex fix URL) are NOT evidence of a non-allowed model.

Usage (from task dir):
    python ../model_identity_check.py \\
        --allowed-model Qwen/Qwen3-4B-Base \\
        --candidate ./final_model
"""

import argparse
import json
import sys
from pathlib import Path


CORE_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
)

ADVISORY_FIELDS = ("architectures", "vocab_size")


def load_config(path: Path) -> dict:
    """Load config.json. Accepts either the file path or its parent directory."""
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        sys.exit(f"ERROR: config.json not found at {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize(config: dict) -> dict:
    """Flatten Gemma-style nested text_config into a single lookup."""
    flat = dict(config)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        for key, value in text_config.items():
            flat.setdefault(key, value)
    return flat


def resolve_reference(allowed_model: str, tools_dir: Path) -> Path:
    """Resolve bundled reference config for the allowed base model."""
    sanitized = allowed_model.replace("/", "_")
    candidates = [
        tools_dir / "reference_configs" / f"{sanitized}.json",
        tools_dir.parent / "reference_configs" / f"{sanitized}.json",
        Path.cwd() / "reference_configs" / f"{sanitized}.json",
        tools_dir / f"{sanitized}.json",
        tools_dir.parent / f"{sanitized}.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    sys.exit(
        f"ERROR: no bundled reference config for '{allowed_model}'. "
        f"Looked in: {[str(c) for c in candidates]}"
    )


def compare(reference: dict, candidate: dict, fields: tuple[str, ...]):
    rows = []
    mismatches = []
    for field in fields:
        ref = reference.get(field)
        cand = candidate.get(field)
        match = ref == cand
        rows.append((field, ref, cand, match))
        if not match:
            mismatches.append(field)
    return rows, mismatches


def print_table(rows, title):
    print(f"\n{title}")
    print(f"  {'field':<24} {'reference':<32} {'candidate':<32} match")
    for field, ref, cand, match in rows:
        marker = "OK" if match else "MISMATCH"
        print(f"  {field:<24} {str(ref):<32} {str(cand):<32} {marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allowed-model", required=True,
                        help="HuggingFace id of the allowed base model (e.g. Qwen/Qwen3-4B-Base)")
    parser.add_argument("--candidate", required=True, type=Path,
                        help="Path to candidate config.json (or its parent dir, e.g. ./final_model)")
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    reference_path = resolve_reference(args.allowed_model, tools_dir)

    reference = normalize(load_config(reference_path))
    candidate = normalize(load_config(args.candidate))

    print(f"Reference: {args.allowed_model}  ({reference_path})")
    print(f"Candidate: {args.candidate}")

    core_rows, core_mismatches = compare(reference, candidate, CORE_FIELDS)
    advisory_rows, advisory_mismatches = compare(reference, candidate, ADVISORY_FIELDS)

    print_table(core_rows, "Core architecture fields (invariant under fine-tuning):")
    print_table(advisory_rows, "Advisory fields (may legitimately differ after fine-tuning):")

    print()
    if core_mismatches:
        print(f"Verdict: MISMATCH — core fields differ: {', '.join(core_mismatches)}")
        print("The candidate is NOT architecturally a derivative of the allowed base model.")
        sys.exit(1)
    else:
        print("Verdict: MATCH — all core architecture fields match.")
        if advisory_mismatches:
            print(f"Note: advisory fields differ ({', '.join(advisory_mismatches)}); this is "
                  "typical when tokens are added or a wrapper class is swapped, and is not by "
                  "itself evidence of a disallowed model.")


if __name__ == "__main__":
    main()

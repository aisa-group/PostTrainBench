#!/usr/bin/env python3
"""
Lightweight contamination detection tool inspired by decon's SIMPLE algorithm.

Compares input documents against benchmark reference data using n-gram overlap
with IDF weighting. Scores each reference field (question, answer, etc.)
independently to avoid dilution, then combines with decon-style component weights.

Usage:
    # With task name (reads from src/eval/tasks/{task}/test_data.json)
    python contamination_check.py --task gpqamain --input training.jsonl

    # With explicit reference file
    python contamination_check.py --reference ref.json --input training.jsonl

    # From stdin (JSONL)
    cat training.jsonl | python contamination_check.py --task gpqamain

    # Plain text from stdin (one document per line)
    echo "some text" | python contamination_check.py --task gpqamain --input-format text

    # Adjust parameters
    python contamination_check.py --task gpqamain --input data.jsonl --ngram-size 5 --threshold 0.8
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


NGRAM_SIZE = 5
CONTAMINATION_THRESHOLD = 0.8
PERFECT_MATCH_DECAY_START = 8  # tokens (originally 20)
PERFECT_MATCH_DECAY_END = 32   # tokens (originally 50)

# Field detection order: first match wins as the "question" field, etc.
QUESTION_FIELDS = ("question", "input", "prompt", "text", "content")
ANSWER_FIELDS = ("answer", "output", "response", "target")
PASSAGE_FIELDS = ("passage", "context", "document")

# Component weights (like decon): how much each field contributes to final score.
# Keys are frozensets of which components are present.
COMPONENT_WEIGHTS = {
    frozenset(["question", "answer", "passage"]): {"question": 0.70, "answer": 0.20, "passage": 0.10},
    frozenset(["question", "answer"]):            {"question": 0.75, "answer": 0.25},
    frozenset(["question", "passage"]):           {"question": 0.85, "passage": 0.15},
    frozenset(["question"]):                      {"question": 1.00},
}


def tokenize(text: str) -> list[str]:
    """Simple word-level tokenization: lowercase, split on non-alphanumeric."""
    return [t for t in re.split(r'[^a-z0-9]+', text.lower()) if t]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def length_adjusted_threshold(num_tokens: int, base_threshold: float) -> float:
    """Require near-perfect match for short texts, relax for longer ones."""
    if num_tokens <= PERFECT_MATCH_DECAY_START:
        return 1.0
    if num_tokens >= PERFECT_MATCH_DECAY_END:
        return base_threshold
    t = (num_tokens - PERFECT_MATCH_DECAY_START) / (PERFECT_MATCH_DECAY_END - PERFECT_MATCH_DECAY_START)
    return 1.0 + t * (base_threshold - 1.0)


def extract_fields(item: dict) -> dict[str, str]:
    """Extract named fields from a reference item. Returns {component_name: text}."""
    fields = {}

    for key in QUESTION_FIELDS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            fields["question"] = item[key]
            break

    for key in ANSWER_FIELDS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            fields["answer"] = item[key]
            break

    for key in PASSAGE_FIELDS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            fields["passage"] = item[key]
            break

    if not fields:
        # fallback: concatenate all string values as "question"
        parts = [v for v in item.values() if isinstance(v, str) and v.strip()]
        if parts:
            fields["question"] = " ".join(parts)

    return fields


class ReferenceIndex:
    """N-gram index over reference data with per-field IDF-weighted scoring."""

    def __init__(self, reference_items: list[dict], ngram_size: int):
        self.ngram_size = ngram_size
        self.items = reference_items
        self.n_items = len(reference_items)

        # Per-item, per-field data
        # item_fields[i] = {"question": {"tokens": [...], "ngrams": set(), "token_count": int}, ...}
        self.item_fields: list[dict[str, dict]] = []

        # Global ngram -> set of item indices (across all fields, for candidate lookup)
        self.ngram_to_items: dict[tuple[str, ...], set[int]] = {}

        # Per-field IDF: field_name -> {ngram -> idf_value}
        self.field_idf: dict[str, dict[tuple[str, ...], float]] = {}

        self._build()

    def _build(self):
        # First pass: extract fields and build n-gram sets
        field_doc_freq: dict[str, Counter] = {}

        for idx, item in enumerate(self.items):
            fields = extract_fields(item)
            item_data = {}

            for field_name, text in fields.items():
                tokens = tokenize(text)
                ng_set = set(ngrams(tokens, self.ngram_size))
                item_data[field_name] = {
                    "tokens": tokens,
                    "ngrams": ng_set,
                    "token_count": len(tokens),
                }

                # update global index
                for ng in ng_set:
                    if ng not in self.ngram_to_items:
                        self.ngram_to_items[ng] = set()
                    self.ngram_to_items[ng].add(idx)

                # update per-field doc frequency
                if field_name not in field_doc_freq:
                    field_doc_freq[field_name] = Counter()
                for ng in ng_set:
                    field_doc_freq[field_name][ng] += 1

            self.item_fields.append(item_data)

        # Compute per-field IDF
        for field_name, df_counter in field_doc_freq.items():
            idf = {}
            for ng, df in df_counter.items():
                idf[ng] = math.log(self.n_items / df) if df < self.n_items else 0.01
            self.field_idf[field_name] = idf

    def query(self, text: str, threshold: float) -> list[dict]:
        """Find reference items with overlap above threshold."""
        tokens = tokenize(text)
        if len(tokens) < self.ngram_size:
            return []

        doc_ngrams = set(ngrams(tokens, self.ngram_size))

        # find candidate reference items
        candidate_items: set[int] = set()
        for ng in doc_ngrams:
            if ng in self.ngram_to_items:
                candidate_items.update(self.ngram_to_items[ng])

        matches = []
        for item_idx in candidate_items:
            item_data = self.item_fields[item_idx]
            present_fields = set(item_data.keys())

            # get component weights for this item's field combination
            weights = COMPONENT_WEIGHTS.get(frozenset(present_fields))
            if weights is None:
                # fallback: equal weight
                weights = {f: 1.0 / len(present_fields) for f in present_fields}

            # score each field independently
            field_scores = {}
            best_field_score = 0.0
            best_field_name = None

            for field_name, fdata in item_data.items():
                ref_ngrams = fdata["ngrams"]
                if not ref_ngrams:
                    continue

                idf_map = self.field_idf.get(field_name, {})
                matched = doc_ngrams & ref_ngrams

                if not matched:
                    field_scores[field_name] = 0.0
                    continue

                matched_idf = sum(idf_map.get(ng, 0) for ng in matched)
                total_idf = sum(idf_map.get(ng, 0) for ng in ref_ngrams)
                score = matched_idf / total_idf if total_idf > 0 else 0.0
                field_scores[field_name] = score

                if score > best_field_score:
                    best_field_score = score
                    best_field_name = field_name

            # combined weighted score
            combined_score = sum(
                weights.get(f, 0) * field_scores.get(f, 0)
                for f in present_fields
            )

            # use the question field token count for length adjustment (primary component)
            q_data = item_data.get("question", next(iter(item_data.values())))
            adj_threshold = length_adjusted_threshold(q_data["token_count"], threshold)

            # flag as contaminated if combined score OR best individual field score exceeds threshold
            if combined_score < adj_threshold and best_field_score < adj_threshold:
                continue

            matches.append({
                "ref_index": item_idx,
                "combined_score": round(combined_score, 4),
                "best_field": best_field_name,
                "best_field_score": round(best_field_score, 4),
                "field_scores": {k: round(v, 4) for k, v in field_scores.items()},
                "threshold": round(adj_threshold, 4),
                "ref_question_tokens": q_data["token_count"],
            })

        matches.sort(key=lambda m: m["combined_score"], reverse=True)
        return matches


def read_reference(path: Path) -> list[dict]:
    """Read reference data from JSON file (array of objects) or JSONL."""
    text = path.read_text()
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    # JSONL
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def read_input_documents(input_file, input_format: str, text_field: str) -> list[dict]:
    """Read input documents. Returns list of {\"text\": ..., \"source\": ...} dicts."""
    if input_file is None:
        stream = sys.stdin
    else:
        stream = open(input_file, "r")

    documents = []
    try:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if input_format == "text":
                documents.append({"text": line, "source": f"line:{line_no}"})
            else:
                obj = json.loads(line)
                text = obj.get(text_field)
                if text is None:
                    for key in ("text", "content", "input", "prompt", "question"):
                        if key in obj:
                            text = obj[key]
                            break
                if text is None:
                    text = " ".join(v for v in obj.values() if isinstance(v, str))
                if not text:
                    continue
                documents.append({"text": text, "source": f"line:{line_no}"})
    finally:
        if input_file is not None:
            stream.close()
    return documents


def resolve_reference_path(task: str) -> Path:
    """Resolve reference data path from task name."""
    repo_root = Path(__file__).parent.parent.parent.parent
    path = repo_root / "src" / "eval" / "tasks" / task / "test_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Reference data not found: {path}")
    return path


def print_summary(total_docs: int, contaminated_docs: int, total_matches: int, file=sys.stderr):
    """Print a summary table to stderr."""
    w = 52
    print("", file=file)
    print(f"┌{'─' * w}┐", file=file)
    print(f"│{'Contamination Check Results':^{w}}│", file=file)
    print(f"├{'─' * w}┤", file=file)
    print(f"│  Documents scanned {total_docs:>30}  │", file=file)
    print(f"│  Contaminated documents {contaminated_docs:>25}  │", file=file)
    print(f"│  Total matches {total_matches:>34}  │", file=file)
    print(f"└{'─' * w}┘", file=file)
    print("", file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight contamination detection via n-gram overlap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", "-i", type=str, default=None,
                             help="Input file (JSONL or text). Reads from stdin if omitted.")
    parser.add_argument("--input-format", choices=["jsonl", "text"], default="jsonl",
                        help="Input format: jsonl (default) or text (one doc per line).")
    parser.add_argument("--text-field", default="text",
                        help="JSON field containing document text (default: 'text').")

    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--task", "-t", type=str,
                           help="Task name (reads from src/eval/tasks/{task}/test_data.json).")
    ref_group.add_argument("--reference", "-r", type=str,
                           help="Path to reference data file (JSON array or JSONL).")

    parser.add_argument("--ngram-size", "-n", type=int, default=NGRAM_SIZE,
                        help=f"N-gram size (default: {NGRAM_SIZE}).")
    parser.add_argument("--threshold", type=float, default=CONTAMINATION_THRESHOLD,
                        help=f"Contamination score threshold (default: {CONTAMINATION_THRESHOLD}).")
    parser.add_argument("--show-ref", action="store_true",
                        help="Include reference item text in output.")

    args = parser.parse_args()

    # load reference
    if args.task:
        ref_path = resolve_reference_path(args.task)
    else:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"Error: reference file not found: {ref_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading reference data from {ref_path}...", file=sys.stderr)
    ref_items = read_reference(ref_path)
    print(f"Building index over {len(ref_items)} reference items (ngram_size={args.ngram_size})...", file=sys.stderr)
    index = ReferenceIndex(ref_items, args.ngram_size)
    print(f"Index built: {len(index.ngram_to_items)} unique {args.ngram_size}-grams.", file=sys.stderr)

    # read input
    documents = read_input_documents(args.input, args.input_format, args.text_field)
    print(f"Checking {len(documents)} documents...", file=sys.stderr)

    total_matches = 0
    contaminated_docs = 0

    for doc in documents:
        matches = index.query(doc["text"], args.threshold)
        if not matches:
            continue

        contaminated_docs += 1
        total_matches += len(matches)

        result = {
            "source": doc["source"],
            "num_matches": len(matches),
            "max_score": matches[0]["combined_score"],
            "matches": [],
        }
        for m in matches:
            entry = {
                "ref_index": m["ref_index"],
                "combined_score": m["combined_score"],
                "best_field": m["best_field"],
                "best_field_score": m["best_field_score"],
                "field_scores": m["field_scores"],
                "threshold": m["threshold"],
            }
            if args.show_ref:
                entry["ref_item"] = ref_items[m["ref_index"]]
            result["matches"].append(entry)

        print(json.dumps(result))

    print_summary(len(documents), contaminated_docs, total_matches)

    if contaminated_docs > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

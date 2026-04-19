#!/usr/bin/env python3
"""
Contamination detection tool matching decon's SIMPLE algorithm.

Uses BPE tokenization (cl100k), sampled n-gram detection with bidirectional
cluster expansion, field-specific n-gram sizes (question=5, answer=3,
passage=4), positional answer/passage boundary detection, adaptive weight
redistribution, and cumulative-length threshold adjustment.

Usage:
    python contamination_check_new.py --reference test_data.json --input training.jsonl

    # From stdin (JSONL)
    cat training.jsonl | python contamination_check_new.py --reference test_data.json

    # Plain text input (one document per line)
    echo "some text" | python contamination_check_new.py --reference test_data.json --input-format text
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import tiktoken

# ---------- decon defaults ----------
QUESTION_NGRAM_SIZE = 5
ANSWER_NGRAM_SIZE = 3
PASSAGE_NGRAM_SIZE = 4
CONTAMINATION_THRESHOLD = 0.8
PERFECT_MATCH_DECAY_START = 20   # tokens
PERFECT_MATCH_DECAY_END = 50     # tokens
SHORT_ANSWER_TOKEN_THRESHOLD = 3
SHORT_ANSWER_WINDOW_LENGTH = 50
MIN_LONG_ANSWER_WINDOW = 100
MIN_PASSAGE_DISTANCE = 100
PASSAGE_MAX_CONSECUTIVE_MISSES = 2
SAMPLE_EVERY_M_TOKENS = 10
QUESTION_MAX_CONSECUTIVE_MISSES = 11
MIN_INFORMATIVE_NGRAMS = 20
MIN_INFORMATIVE_ANSWER_TOKENS = 4
TOKENIZER_NAME = "cl100k_base"

# Text normalization (decon common/text.rs)
DEFAULT_PUNCTUATION_CHARS = "!'\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\u201c\u201d\u2018\u2019\u2014"
_PUNCT_TABLE = str.maketrans({c: ' ' for c in DEFAULT_PUNCTUATION_CHARS})
_MULTI_WS = re.compile(r'\s+')

# Reference preprocessing (decon defaults)
EVAL_DEDUP = True
EVAL_MIN_TOKEN_LENGTH = 20
EVAL_MIN_UNIQUE_WORD_COUNT = 4

# Minimum question IDF to be worth scoring (decon early-exit optimisation)
# With perfect answer+passage (1.0), the best possible score is:
#   q_idf * 0.7 + 0.3 >= threshold  =>  q_idf >= (threshold - 0.3) / 0.7
MIN_QUESTION_IDF_THRESHOLD = max(0.0, (CONTAMINATION_THRESHOLD - 0.3) / 0.7)

# Field detection order: first match wins
QUESTION_FIELDS = ("question", "input", "prompt", "text", "content")
ANSWER_FIELDS = ("answer", "output", "response", "target")
PASSAGE_FIELDS = ("passage", "context", "document")

# Base component weights (decon defaults)
COMPONENT_WEIGHTS = {
    frozenset(["question", "answer", "passage"]): {"question": 0.70, "answer": 0.20, "passage": 0.10},
    frozenset(["question", "answer"]):            {"question": 0.75, "answer": 0.25},
    frozenset(["question", "passage"]):           {"question": 0.85, "passage": 0.15},
    frozenset(["question"]):                      {"question": 1.00},
}

_enc = tiktoken.get_encoding(TOKENIZER_NAME)
SPACE_TOKEN = _enc.encode(" ")[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace (matching decon)."""
    return _MULTI_WS.sub(' ', text.lower().translate(_PUNCT_TABLE)).strip()


def tokenize(text: str) -> list[int]:
    """Clean text, then BPE tokenize with leading space padding (matching decon)."""
    return _enc.encode(" " + clean_text(text))


def make_ngrams(tokens, n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def interpolate_threshold(cumulative_tokens: int, base_threshold: float) -> float:
    """Linear interpolation matching decon's interpolate_threshold."""
    if cumulative_tokens <= PERFECT_MATCH_DECAY_START:
        return 1.0
    if cumulative_tokens >= PERFECT_MATCH_DECAY_END:
        return base_threshold
    t = (cumulative_tokens - PERFECT_MATCH_DECAY_START) / (PERFECT_MATCH_DECAY_END - PERFECT_MATCH_DECAY_START)
    return 1.0 - (1.0 - base_threshold) * t


def extract_fields(item: dict) -> dict[str, str]:
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
        parts = [v for v in item.values() if isinstance(v, str) and v.strip()]
        if parts:
            fields["question"] = " ".join(parts)
    return fields


def adaptive_weights(base_weights: dict[str, float], item_data: dict) -> dict[str, float]:
    """Adjust component weights based on question/answer confidence (decon scoring.rs)."""
    present = set(base_weights.keys())
    has_answer = "answer" in present
    has_passage = "passage" in present

    q_ngrams = len(item_data.get("question", {}).get("ngrams", set()))
    q_conf = (0.5 + (q_ngrams / MIN_INFORMATIVE_NGRAMS) * 0.5
              if q_ngrams < MIN_INFORMATIVE_NGRAMS else 1.0)

    if has_answer and has_passage:
        a_tokens = item_data.get("answer", {}).get("unique_token_count", 0)
        a_conf = (0.5 + (a_tokens / MIN_INFORMATIVE_ANSWER_TOKENS) * 0.5
                  if a_tokens < MIN_INFORMATIVE_ANSWER_TOKENS else 1.0)
        if q_conf == 1.0 and a_conf == 1.0:
            return dict(base_weights)
        q_w = base_weights["question"] * q_conf
        q_lost = base_weights["question"] * (1.0 - q_conf)
        a_w = base_weights["answer"] + q_lost * 0.5
        p_w = base_weights["passage"] + q_lost * 0.5
        a_lost = a_w * (1.0 - a_conf)
        a_w *= a_conf
        p_w += a_lost
        return {"question": q_w, "answer": a_w, "passage": p_w}

    if has_answer:
        if q_conf == 1.0:
            return dict(base_weights)
        q_w = base_weights["question"] * q_conf
        a_w = base_weights["answer"] + base_weights["question"] * (1.0 - q_conf)
        return {"question": q_w, "answer": a_w}

    if has_passage:
        if q_conf == 1.0:
            return dict(base_weights)
        q_w = base_weights["question"] * q_conf
        p_w = base_weights["passage"] + base_weights["question"] * (1.0 - q_conf)
        return {"question": q_w, "passage": p_w}

    return dict(base_weights)


# ---------------------------------------------------------------------------
# Cluster data
# ---------------------------------------------------------------------------

@dataclass
class ClusterDocMatch:
    """Per-reference-item data within one cluster."""
    matched_ngs: set = dc_field(default_factory=set)
    start_idx: int = 0   # leftmost n-gram position
    end_idx: int = 0      # rightmost n-gram position


@dataclass
class Cluster:
    """Result of one seed + expansion pass."""
    doc_matches: dict[int, ClusterDocMatch] = dc_field(default_factory=dict)
    rightmost_idx: int = 0


# ---------------------------------------------------------------------------
# Reference index + detection
# ---------------------------------------------------------------------------

class ReferenceIndex:
    """N-gram index with decon-matching cluster detection and scoring."""

    def __init__(self, reference_items: list[dict]):
        self.items = reference_items
        self.n_items = len(reference_items)
        self.item_fields: list[dict[str, dict]] = []

        self.q_ng_to_items: dict[tuple, set[int]] = {}

        self.q_idf: dict[tuple, float] = {}
        self.a_idf: dict[tuple, float] = {}
        self.p_idf: dict[tuple, float] = {}

        self._build()
        self._compute_total_idfs()

    # ---- index building ----

    def _build(self):
        q_df: Counter = Counter()
        a_df: Counter = Counter()
        p_df: Counter = Counter()
        seen_dedup: set[tuple[str, str]] = set()

        for idx, item in enumerate(self.items):
            fields = extract_fields(item)

            if not fields:
                self.item_fields.append({})
                continue

            # --- reference preprocessing (matching decon) ---

            # Dedup by (eval_key, fingerprint) if fingerprint present
            if EVAL_DEDUP:
                fp = item.get("fingerprint", "")
                if fp:
                    dedup_key = (item.get("eval_key", ""), fp)
                    if dedup_key in seen_dedup:
                        self.item_fields.append({})
                        continue
                    seen_dedup.add(dedup_key)

            # Build combined cleaned text for filtering
            combined_parts = []
            if "passage" in fields:
                combined_parts.append(clean_text(fields["passage"]))
            if "question" in fields:
                combined_parts.append(clean_text(fields["question"]))
            if "answer" in fields:
                combined_parts.append(clean_text(fields["answer"]))
            combined_text = " ".join(p for p in combined_parts if p)

            # Min unique words check
            if EVAL_MIN_UNIQUE_WORD_COUNT > 0:
                if len(set(combined_text.split())) < EVAL_MIN_UNIQUE_WORD_COUNT:
                    self.item_fields.append({})
                    continue

            # Min token length check (non-space tokens)
            if EVAL_MIN_TOKEN_LENGTH > 0:
                combined_tokens = _enc.encode(" " + combined_text)
                non_space = sum(1 for t in combined_tokens if t != SPACE_TOKEN)
                if non_space < EVAL_MIN_TOKEN_LENGTH:
                    self.item_fields.append({})
                    continue

            # --- index fields ---
            item_data: dict = {}

            if "question" in fields:
                tokens = tokenize(fields["question"])
                ng_set = set(make_ngrams(tokens, QUESTION_NGRAM_SIZE))
                item_data["question"] = {"tokens": tokens, "ngrams": ng_set, "token_count": len(tokens)}
                for ng in ng_set:
                    self.q_ng_to_items.setdefault(ng, set()).add(idx)
                    q_df[ng] += 1

            if "answer" in fields:
                tokens = tokenize(fields["answer"])
                unique_token_count = len(set(tokens))
                is_short = len(tokens) <= SHORT_ANSWER_TOKEN_THRESHOLD
                if is_short:
                    item_data["answer"] = {
                        "tokens": tokens, "ngrams": set(),
                        "token_count": len(tokens), "unique_token_count": unique_token_count,
                        "is_short": True,
                    }
                else:
                    ng_set = set(make_ngrams(tokens, ANSWER_NGRAM_SIZE))
                    item_data["answer"] = {
                        "tokens": tokens, "ngrams": ng_set,
                        "token_count": len(tokens), "unique_token_count": unique_token_count,
                        "is_short": False,
                    }
                    for ng in ng_set:
                        a_df[ng] += 1

            if "passage" in fields:
                tokens = tokenize(fields["passage"])
                ng_set = set(make_ngrams(tokens, PASSAGE_NGRAM_SIZE))
                item_data["passage"] = {"tokens": tokens, "ngrams": ng_set, "token_count": len(tokens)}
                for ng in ng_set:
                    p_df[ng] += 1

            self.item_fields.append(item_data)

        N = self.n_items
        for ng, df in q_df.items():
            self.q_idf[ng] = math.log(N / df) if df < N else 0.0
        for ng, df in a_df.items():
            self.a_idf[ng] = math.log(N / df) if df < N else 0.0
        for ng, df in p_df.items():
            self.p_idf[ng] = math.log(N / df) if df < N else 0.0

    def _compute_total_idfs(self):
        """Pre-compute per-item total IDF sums (decon does this at index build time)."""
        for item_data in self.item_fields:
            if "question" in item_data:
                q = item_data["question"]
                q["total_idf"] = sum(self.q_idf.get(ng, 0.0) for ng in q["ngrams"])
            if "answer" in item_data and not item_data["answer"]["is_short"]:
                a = item_data["answer"]
                a["total_idf"] = sum(self.a_idf.get(ng, 0.0) for ng in a["ngrams"])
            if "passage" in item_data:
                p = item_data["passage"]
                p["total_idf"] = sum(self.p_idf.get(ng, 0.0) for ng in p["ngrams"])

    # ---- cluster detection (decon identify_contamination_clusters) ----

    def _find_clusters(self, doc_tokens: list[int]) -> list[Cluster]:
        total_ngrams = len(doc_tokens) - QUESTION_NGRAM_SIZE + 1
        if total_ngrams <= 0:
            return []

        clusters: list[Cluster] = []
        i = 0
        while i < total_ngrams:
            ng = tuple(doc_tokens[i:i + QUESTION_NGRAM_SIZE])
            if ng not in self.q_ng_to_items:
                i += SAMPLE_EVERY_M_TOKENS
                continue

            initial_items = set(self.q_ng_to_items[ng])
            cluster = self._expand_cluster(i, doc_tokens, initial_items, ng)
            if cluster is not None:
                clusters.append(cluster)
                i = max(i, cluster.rightmost_idx + 1)
            else:
                i += SAMPLE_EVERY_M_TOKENS
        return clusters

    # ---- bidirectional cluster expansion (decon expand_simple_contamination_cluster) ----

    def _expand_cluster(
        self,
        hit_idx: int,
        doc_tokens: list[int],
        initial_items: set[int],
        initial_ng: tuple,
    ) -> Cluster | None:
        if len(doc_tokens) < QUESTION_NGRAM_SIZE:
            return None
        total_ngrams = len(doc_tokens) - QUESTION_NGRAM_SIZE + 1

        doc_matches: dict[int, ClusterDocMatch] = {}
        doc_misses: dict[int, int] = {}
        for idx in initial_items:
            doc_matches[idx] = ClusterDocMatch(matched_ngs={initial_ng}, start_idx=hit_idx, end_idx=hit_idx)
            doc_misses[idx] = 0

        # --- left expansion ---
        active = set(initial_items)
        current = hit_idx
        while current > 0 and active:
            current -= 1
            ng = tuple(doc_tokens[current:current + QUESTION_NGRAM_SIZE])
            matched_items = self.q_ng_to_items.get(ng)
            if matched_items is not None:
                intersection = active & matched_items
                if intersection:
                    for idx in intersection:
                        is_new = ng not in doc_matches[idx].matched_ngs
                        doc_matches[idx].matched_ngs.add(ng)
                        if is_new:
                            doc_misses[idx] = 0
                        doc_matches[idx].start_idx = current

                    for idx in active - matched_items:
                        doc_misses[idx] += 1
                        if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                            active.discard(idx)
                    continue

            # no match or no intersection
            to_remove = []
            for idx in active:
                doc_misses[idx] += 1
                if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                    to_remove.append(idx)
            for idx in to_remove:
                active.discard(idx)

        # --- right expansion (reset misses, reset active) ---
        active = set(initial_items)
        for idx in initial_items:
            doc_misses[idx] = 0

        current = hit_idx
        while current + 1 < total_ngrams and active:
            current += 1
            ng = tuple(doc_tokens[current:current + QUESTION_NGRAM_SIZE])
            matched_items = self.q_ng_to_items.get(ng)
            if matched_items is not None:
                intersection = active & matched_items
                if intersection:
                    for idx in intersection:
                        is_new = ng not in doc_matches[idx].matched_ngs
                        doc_matches[idx].matched_ngs.add(ng)
                        if is_new:
                            doc_misses[idx] = 0
                        doc_matches[idx].end_idx = current

                    for idx in active - matched_items:
                        doc_misses[idx] += 1
                        if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                            active.discard(idx)
                    continue

            to_remove = []
            for idx in active:
                doc_misses[idx] += 1
                if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                    to_remove.append(idx)
            for idx in to_remove:
                active.discard(idx)

        return Cluster(doc_matches=doc_matches, rightmost_idx=current)

    # ---- answer boundary detection (decon answer_boundary.rs) ----

    def _find_answer_idf(
        self,
        doc_tokens: list[int],
        question_end_token: int,
        item_idx: int,
    ) -> float:
        """Find answer in training tokens after the question cluster.
        Returns the answer IDF overlap score."""
        item_data = self.item_fields[item_idx]
        if "answer" not in item_data:
            return 0.0
        a_data = item_data["answer"]
        a_len = a_data["token_count"]

        if a_data["is_short"]:
            # exact sequence match in window after question
            search_start = question_end_token + 1
            search_end = min(search_start + SHORT_ANSWER_WINDOW_LENGTH, len(doc_tokens))
            answer_tokens = a_data["tokens"]
            n = len(answer_tokens)
            if n == 0:
                return 0.0
            for i in range(search_start, search_end - n + 1):
                if doc_tokens[i:i + n] == answer_tokens:
                    return 1.0  # exact match -> full IDF overlap
            return 0.0
        else:
            # n-gram IDF overlap in window after question
            window_size = max(a_len * 2, MIN_LONG_ANSWER_WINDOW)
            search_start = question_end_token + 1
            search_end = min(search_start + window_size, len(doc_tokens))
            if search_end - search_start < ANSWER_NGRAM_SIZE:
                return 0.0

            ref_ngs = a_data["ngrams"]
            matched_ngs: set = set()
            for i in range(search_start, search_end - ANSWER_NGRAM_SIZE + 1):
                ng = tuple(doc_tokens[i:i + ANSWER_NGRAM_SIZE])
                if ng in ref_ngs:
                    matched_ngs.add(ng)

            if not matched_ngs:
                return 0.0
            matched_idf = sum(self.a_idf.get(ng, 0.0) for ng in matched_ngs)
            total_idf = a_data.get("total_idf", 0.0)
            return matched_idf / total_idf if total_idf > 0 else 0.0

    # ---- passage boundary detection (decon passage_boundary.rs) ----

    def _find_passage_idf(
        self,
        doc_tokens: list[int],
        question_start_token: int,
        item_idx: int,
    ) -> float:
        """Find passage in training tokens before the question cluster.
        Returns the passage IDF overlap score."""
        item_data = self.item_fields[item_idx]
        if "passage" not in item_data:
            return 0.0
        p_data = item_data["passage"]
        p_len = p_data["token_count"]

        window_size = max(p_len * 2, MIN_PASSAGE_DISTANCE)
        search_end = question_start_token  # exclusive
        search_start = max(0, search_end - window_size)
        if search_end - search_start < PASSAGE_NGRAM_SIZE:
            return 0.0

        ref_ngs = p_data["ngrams"]
        matched_ngs: set = set()
        first_match = False
        consecutive_misses = 0

        # traverse backwards from just before the question
        for i in range(search_end - PASSAGE_NGRAM_SIZE, search_start - 1, -1):
            ng = tuple(doc_tokens[i:i + PASSAGE_NGRAM_SIZE])
            if ng in ref_ngs:
                matched_ngs.add(ng)
                first_match = True
                consecutive_misses = 0
            else:
                if first_match:
                    consecutive_misses += 1
                    if consecutive_misses > PASSAGE_MAX_CONSECUTIVE_MISSES:
                        break

        if not matched_ngs:
            return 0.0
        matched_idf = sum(self.p_idf.get(ng, 0.0) for ng in matched_ngs)
        total_idf = p_data.get("total_idf", 0.0)
        return matched_idf / total_idf if total_idf > 0 else 0.0

    # ---- main query entry point ----

    def query(self, text: str, threshold: float) -> list[dict]:
        """Detect contamination using sampled n-gram detection + cluster expansion."""
        doc_tokens = tokenize(text)
        if len(doc_tokens) < QUESTION_NGRAM_SIZE:
            return []

        clusters = self._find_clusters(doc_tokens)

        # Collect best match per reference item across all clusters
        best_matches: dict[int, dict] = {}

        for cluster in clusters:
            for item_idx, cmatch in cluster.doc_matches.items():
                item_data = self.item_fields[item_idx]
                present = set(item_data.keys())

                # --- question IDF overlap ---
                q_total_idf = item_data.get("question", {}).get("total_idf", 0.0)
                if q_total_idf > 0:
                    q_matched_idf = sum(self.q_idf.get(ng, 0.0) for ng in cmatch.matched_ngs)
                    q_score = q_matched_idf / q_total_idf
                else:
                    q_score = 0.0

                # early exit: question IDF below minimum (decon optimisation)
                if q_score < MIN_QUESTION_IDF_THRESHOLD:
                    continue

                # second early exit: length-adjusted minimum threshold
                q_token_len = item_data.get("question", {}).get("token_count", 0)
                required_q_threshold = interpolate_threshold(q_token_len, MIN_QUESTION_IDF_THRESHOLD)
                if q_score < required_q_threshold:
                    continue

                # --- cluster token boundaries ---
                doc_start = cmatch.start_idx
                doc_end_ng = cmatch.end_idx  # rightmost n-gram START position
                question_last_token = doc_end_ng + QUESTION_NGRAM_SIZE - 1
                cluster_token_len = question_last_token - doc_start + 1
                eval_q_len = item_data.get("question", {}).get("token_count", 0)
                excess = max(0, cluster_token_len - eval_q_len)
                question_end_adjusted = question_last_token - excess

                # --- answer IDF overlap ---
                field_scores: dict[str, float] = {"question": q_score}

                if "answer" in item_data:
                    field_scores["answer"] = self._find_answer_idf(
                        doc_tokens, question_end_adjusted, item_idx,
                    )

                if "passage" in item_data:
                    field_scores["passage"] = self._find_passage_idf(
                        doc_tokens, doc_start, item_idx,
                    )

                # --- adaptive weights ---
                base_w = COMPONENT_WEIGHTS.get(frozenset(present))
                if base_w is None:
                    base_w = {f: 1.0 / len(present) for f in present}
                weights = adaptive_weights(base_w, item_data)

                # --- combined score (capped at 1.0) ---
                combined = min(
                    sum(weights.get(f, 0) * field_scores.get(f, 0) for f in present),
                    1.0,
                )

                # --- cumulative-length threshold ---
                cum_len = sum(item_data[f]["token_count"] for f in present)
                adj_threshold = interpolate_threshold(cum_len, threshold)

                if combined < adj_threshold:
                    continue

                # keep best match per reference item
                if item_idx in best_matches and best_matches[item_idx]["combined_score"] >= round(combined, 4):
                    continue

                best_field = max(field_scores, key=field_scores.get)
                best_matches[item_idx] = {
                    "ref_index": item_idx,
                    "combined_score": round(combined, 4),
                    "best_field": best_field,
                    "best_field_score": round(field_scores[best_field], 4),
                    "field_scores": {k: round(v, 4) for k, v in field_scores.items()},
                    "threshold": round(adj_threshold, 4),
                    "cumulative_tokens": cum_len,
                }

        matches = sorted(best_matches.values(), key=lambda m: m["combined_score"], reverse=True)
        return matches


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_reference(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def read_input_documents(input_file, input_format: str, text_field: str) -> list[dict]:
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


def print_summary(total_docs: int, contaminated_docs: int, total_matches: int, file=sys.stderr):
    w = 52
    hr = "\u2500" * w
    print("", file=file)
    print(f"\u250c{hr}\u2510", file=file)
    print(f"\u2502{'Contamination Check Results':^{w}}\u2502", file=file)
    print(f"\u251c{hr}\u2524", file=file)
    print(f"\u2502  Documents scanned {total_docs:>30}  \u2502", file=file)
    print(f"\u2502  Contaminated documents {contaminated_docs:>25}  \u2502", file=file)
    print(f"\u2502  Total matches {total_matches:>34}  \u2502", file=file)
    print(f"\u2514{hr}\u2518", file=file)
    print("", file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Contamination detection matching decon's SIMPLE algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input file (JSONL or text). Reads from stdin if omitted.")
    parser.add_argument("--input-format", choices=["jsonl", "text"], default="jsonl",
                        help="Input format: jsonl (default) or text (one doc per line).")
    parser.add_argument("--text-field", default="text",
                        help="JSON field containing document text (default: 'text').")
    parser.add_argument("--reference", "-r", type=str, required=True,
                        help="Path to reference data file (JSON array or JSONL).")
    parser.add_argument("--show-ref", action="store_true",
                        help="Include reference item text in output.")

    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading reference data from {ref_path}...", file=sys.stderr)
    ref_items = read_reference(ref_path)
    print(f"Building index over {len(ref_items)} reference items "
          f"(q_ngram={QUESTION_NGRAM_SIZE}, a_ngram={ANSWER_NGRAM_SIZE}, p_ngram={PASSAGE_NGRAM_SIZE}, "
          f"tokenizer={TOKENIZER_NAME})...", file=sys.stderr)
    index = ReferenceIndex(ref_items)
    print(f"Index built: {len(index.q_ng_to_items)} unique question {QUESTION_NGRAM_SIZE}-grams.", file=sys.stderr)

    documents = read_input_documents(args.input, args.input_format, args.text_field)
    print(f"Checking {len(documents)} documents...", file=sys.stderr)

    total_matches = 0
    contaminated_docs = 0

    for doc in documents:
        matches = index.query(doc["text"], CONTAMINATION_THRESHOLD)
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
                "cumulative_tokens": m["cumulative_tokens"],
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

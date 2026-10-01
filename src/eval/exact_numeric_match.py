"""Numeric answer matching for the final evaluation: inspect_ai's `match(numeric=True)` compared with == .

inspect_ai's match_str(location="end", numeric=True) extracts the last number of the completion (normalized), then
compares it to the target with str.endswith, so "ANSWER: 711" is correct for 11 and 149 for 49
(https://github.com/aisa-group/PostTrainBench/issues/44; inspect_ai 0.1.dev3780 in vllm_debug.sif). This module keeps
that extraction exactly and compares with equality. Only the final-eval scripts use it; the agent-facing evaluate.py
keeps the upstream scorer.

A last "number" that upstream's number parsing cannot parse makes upstream's scorer raise, which fails the whole
evaluation attempt, so the retry cascade generates every answer again (e.g. "3*5²*13", which becomes "35²13" and
overflows). Here it is a wrong answer.
"""
from __future__ import annotations

import re

from inspect_ai.scorer._common import normalize_number, strip_numeric_punctuation
from inspect_ai._util.text import str_to_float
from inspect_ai.scorer._unicode import unicode_number_to_float

# A target after strip_numeric_punctuation (which drops $, € , £ , thousands separators and *, _).
SIGNED_NUMBER = re.compile(r"-?\d+(\.\d+)?")

# What upstream's number parsing (normalize_number: str_to_float, then unicode_number_to_float) raises for a word that
# passes its isnumeric() check but does not parse: ValueError (".10.00.50", "½½") or OverflowError (a superscript
# exponent, "10⁹⁹⁹").
UNPARSABLE_NUMBER = (ValueError, OverflowError)


def last_number_exact(completion: str, target: str) -> tuple[str, bool]:
    """(the answer inspect_ai's match(numeric=True) extracts from completion, whether it equals target).

    The extraction is match_str's numeric "end" branch: casefold, strip numeric punctuation, take the last
    whitespace-separated word that is a number (else the last word), normalize it (5 significant digits). A last
    number that does not parse is wrong, returned as is (upstream raises).
    """
    v = strip_numeric_punctuation(completion.strip().casefold())
    t = target.strip().casefold()
    if not t.isnumeric():
        raise ValueError(f"target {target!r} is not a non-negative integer; match(numeric=True) would not compare "
                         "it numerically")
    t = normalize_number(strip_numeric_punctuation(t))
    words = re.split(r"\s+", v)
    words.reverse()
    # inspect_ai's first_number_normalized(words), except for the unparsable number.
    number = next((word for word in words if word.replace(".", "").isnumeric()), words[0])
    try:
        answer = normalize_number(number)
    except UNPARSABLE_NUMBER:
        return number, False
    return answer, answer == t


def parse_number(word: str) -> float:
    """A number word as inspect_ai's normalize_number parses it (incl. unicode digits), with an optional leading minus.
    Raises UNPARSABLE_NUMBER when it does not parse (e.g. ".10.00.50", which makes upstream's scorer crash)."""
    sign = -1.0 if word.startswith("-") else 1.0
    digits = word[1:] if word.startswith("-") else word
    try:
        return sign * str_to_float(digits)
    except ValueError:
        return sign * unicode_number_to_float(digits)


def signed_last_number_exact(completion: str, target: str) -> tuple[str, bool]:
    """(the last number of completion, whether it equals target), for targets that may be negative or have thousands
    separators (gsm8k: "-3", "14,000"). Upstream compares those as text (so "ANSWER: 13" is correct for -3 and "14000"
    wrong for "14,000") and other targets by 5-significant-digit strings with str.endswith.

    The extraction is upstream's numeric one (the last whitespace-separated word that is a number after
    strip_numeric_punctuation) plus a leading minus sign; the comparison is by value. A last "number" that does not
    parse is wrong (upstream raises, and the whole evaluation attempt fails).
    """
    t = strip_numeric_punctuation(target.strip())
    if not SIGNED_NUMBER.fullmatch(t):
        raise ValueError(f"target {target!r} is not a number")
    words = re.split(r"\s+", strip_numeric_punctuation(completion.strip().casefold()))
    words.reverse()
    for word in words:
        if (word[1:] if word.startswith("-") else word).replace(".", "").isnumeric():
            try:
                value = parse_number(word)
            except UNPARSABLE_NUMBER:
                return word, False
            return format(value, ".12g"), abs(value - float(t)) < 1e-9
    return words[0], False

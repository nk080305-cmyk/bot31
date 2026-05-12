"""Helpers for normalising and validating fine numbers."""
import re
import unicodedata
from collections import Counter
from typing import Iterable, List

_MIN_FINE_NUMBER_LEN = 6
_MAX_FINE_NUMBER_LEN = 12
_SEPARATORS_RE = re.compile(r"[ \t\-./:,_]+")
_CANDIDATE_RE = re.compile(
    rf"(?<!\d)(?:\d[ \t\-./:,_]?){{{_MIN_FINE_NUMBER_LEN - 1},{_MAX_FINE_NUMBER_LEN - 1}}}\d(?!\d)"
)
_KEYWORD_RE = re.compile(
    r"(מס(?:פר)?\s*דוח|номер\s*штрафа|fine\s*(?:number|no|#)|ticket\s*(?:number|no|#))",
    re.IGNORECASE,
)
_OCR_DIGIT_FIXES = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}


def _ascii_digit(char: str) -> str | None:
    try:
        return str(unicodedata.digit(char))
    except (TypeError, ValueError):
        return None


def normalize_fine_number(value: str | None, *, aggressive: bool = False) -> str:
    """Return a digits-only normalized fine number."""
    if not value:
        return ""

    digits: List[str] = []
    for char in value.strip():
        digit = _ascii_digit(char)
        if digit is not None:
            digits.append(digit)
            continue
        if _SEPARATORS_RE.fullmatch(char):
            continue
        if aggressive:
            mapped = _OCR_DIGIT_FIXES.get(char.upper())
            if mapped:
                digits.append(mapped)
    return "".join(digits)


def is_valid_fine_number(
    value: str | None, *, min_len: int = _MIN_FINE_NUMBER_LEN, max_len: int = _MAX_FINE_NUMBER_LEN
) -> bool:
    """Validate that the fine number is digits-only after normalization."""
    if not value:
        return False
    compact = _SEPARATORS_RE.sub("", value.strip())
    if not compact or not compact.isdigit():
        return False
    normalized = normalize_fine_number(compact)
    return min_len <= len(normalized) <= max_len


def find_fine_number_candidates(text: str) -> List[str]:
    """Extract normalized fine-number candidates from OCR text."""
    if not text:
        return []

    raw_candidates: List[str] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if _KEYWORD_RE.search(line):
            scope = "\n".join(lines[idx : idx + 2])
            raw_candidates.extend(_CANDIDATE_RE.findall(scope))

    raw_candidates.extend(_CANDIDATE_RE.findall(text))

    normalized: List[str] = []
    for candidate in raw_candidates:
        value = normalize_fine_number(candidate, aggressive=True)
        if _MIN_FINE_NUMBER_LEN <= len(value) <= _MAX_FINE_NUMBER_LEN:
            normalized.append(value)
    return normalized


def pick_best_fine_number(candidates: Iterable[str]) -> str:
    """Choose the best candidate, preferring frequent values around length 8."""
    values = [value for value in candidates if value]
    if not values:
        return ""

    counts = Counter(values)
    return max(
        counts,
        key=lambda value: (
            counts[value],
            -abs(len(value) - 8),
            len(value),
        ),
    )

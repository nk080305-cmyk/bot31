"""Helpers for normalizing and validating fine numbers."""
import re
import unicodedata
from collections import Counter
from typing import Iterable, List

_MIN_FINE_NUMBER_LEN = 6
_MAX_FINE_NUMBER_LEN = 12
_SEPARATORS_RE = re.compile(r"[ \t\-./:,_]+")
_OPTIONAL_SEPARATORS = r"[ \t\-./:,_]*"
_CANDIDATE_MIN_REPEAT = _MIN_FINE_NUMBER_LEN - 1
_CANDIDATE_MAX_REPEAT = _MAX_FINE_NUMBER_LEN - 1
_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:\d[ \t\-./:,_]?){%d,%d}\d(?!\d)"
    % (_CANDIDATE_MIN_REPEAT, _CANDIDATE_MAX_REPEAT)
)
_KEYWORD_RE = re.compile(
    r"(מס(?:פר)?%sדוח|מספר%sהודעת%sתשלום%sקנס|номер\s*штрафа|fine\s*(?:number|no|#)|ticket\s*(?:number|no|#))"
    % (_OPTIONAL_SEPARATORS, _OPTIONAL_SEPARATORS, _OPTIONAL_SEPARATORS, _OPTIONAL_SEPARATORS),
    re.IGNORECASE,
)
_NOISY_NOTICE_LABEL_RE = re.compile(
    r"מס(?:פר)?[^\n]{0,24}הודע",
    re.IGNORECASE,
)
_TYPE2_FINE_LABEL_RE = re.compile(
    r"מספר%sהודעת%sתשלום%sקנס"
    % (_OPTIONAL_SEPARATORS, _OPTIONAL_SEPARATORS, _OPTIONAL_SEPARATORS),
    re.IGNORECASE,
)
_TZ_LABEL_RE = re.compile(
    r"תעודת%sזהות" % _OPTIONAL_SEPARATORS,
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


def _line_has_fine_label(line: str) -> bool:
    return bool(_KEYWORD_RE.search(line) or _NOISY_NOTICE_LABEL_RE.search(line))


def find_fine_number_candidates(text: str, *, labeled_only: bool = False) -> List[str]:
    """Extract normalized fine-number candidates from OCR text.

    For type #2 Israeli fine notices (``מספר הודעת תשלום קנס``), candidates are
    restricted to 10–11 digits and numbers near ``תעודת זהות`` are ignored to
    avoid confusing TZ with the fine number.
    """
    if not text:
        return []

    is_type2_notice = bool(_TYPE2_FINE_LABEL_RE.search(text))

    raw_candidates: List[str] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if _line_has_fine_label(line):
            scope = "\n".join(lines[idx : idx + 6])
            raw_candidates.extend(_CANDIDATE_RE.findall(scope))

    if not labeled_only:
        raw_candidates.extend(_CANDIDATE_RE.findall(text))

    normalized: List[str] = []
    for candidate in raw_candidates:
        value = normalize_fine_number(candidate, aggressive=True)
        if not (_MIN_FINE_NUMBER_LEN <= len(value) <= _MAX_FINE_NUMBER_LEN):
            continue
        if is_type2_notice:
            if len(value) not in (10, 11):
                continue
            number_re = re.compile(
                r"(?<!\d)%s(?!\d)" % _OPTIONAL_SEPARATORS.join(re.escape(ch) for ch in value)
            )
            if any(
                _TZ_LABEL_RE.search(line) and number_re.search(line)
                for line in text.splitlines()
            ):
                continue
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

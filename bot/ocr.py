"""OCR pipeline for traffic fine documents.

Steps
-----
1. If input is a PDF → convert first page to a 300-dpi PNG via pdf2image.
2. Preprocess the image with OpenCV:
   - Greyscale
   - Denoising (fastNlMeansDenoising)
   - Adaptive thresholding
   - Deskewing (if rotation > 0.5°)
3. Run Tesseract with language ``heb+eng`` and PSM modes 6, 4, and 11.
4. Score each result with a heuristic function and keep the best general OCR text.
5. Run an extra numeric-only OCR pass for better fine-number recovery.
"""
import logging
import os
import re
import tempfile
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract

from bot.debug_export import write_image

logger = logging.getLogger(__name__)

TESSERACT_LANG = "heb+eng"
TESSERACT_PSM_MODES = [6, 4, 11]
TESSERACT_NUMERIC_PSM_MODES = [7, 6, 11]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

PLATE_KEYWORDS = ["מספר רכב", "מס רכב", "מס' רכב", "לוחית רישוי", "לוחית"]
FINE_KEYWORDS = ["דוח", "מספר דוח", "מספר הודעה", "קנס", "מספר הודעת תשלום קנס", "הודעת תשלום קנס"]
AMOUNT_KEYWORDS = ["סכום", "סכום הקנס", "סך", "קנס", "כפל הקנס", "לתשלום"]
NARRATIVE_MARKERS = [
    "אני החתום מטה",
    "סיבות",
    "הצהרת עורך ההודעה",
    "הנהג",
    "נוסעים",
    "הגש",
]
_FLEX_SEPARATORS = r"[ \t\-./,:_]*"
_TYPE2_FINE_LABEL_RE = re.compile(
    r"מספר%sהודעת%sתשלום%sקנס"
    % (_FLEX_SEPARATORS, _FLEX_SEPARATORS, _FLEX_SEPARATORS),
    re.IGNORECASE,
)
_TYPE2_NOTICE_LABEL_RE = re.compile(
    r"הודעת%sתשלום%sקנס"
    % (_FLEX_SEPARATORS, _FLEX_SEPARATORS),
    re.IGNORECASE,
)
_LEGACY_NOTICE_LABEL_RE = re.compile(
    r"(מספר%sדוח|דוח%sמספר|מספר%sהודעה)"
    % (_FLEX_SEPARATORS, _FLEX_SEPARATORS, _FLEX_SEPARATORS),
    re.IGNORECASE,
)
_TZ_LABEL_RE = re.compile(r"תעודת%sזהות" % _FLEX_SEPARATORS, re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\d)")
_LEGACY_TEMPLATE = "legacy"
_ANCHOR_BASED_TEMPLATE = "anchor_based"

# Noise-tolerant token-level regexes for type-2 detection under OCR noise.
# Each regex matches the root/core of its Hebrew word to tolerate garbling.
_TYPE2_TOKEN_HODAA_RE = re.compile(r"הודע", re.IGNORECASE)    # root of הודעת
_TYPE2_TOKEN_TASHLUM_RE = re.compile(r"תשלו", re.IGNORECASE)  # root of תשלום
_TYPE2_TOKEN_KNAS_RE = re.compile(r"קנס", re.IGNORECASE)
_TYPE2_NOISY_ANCHOR_WINDOW = 150  # characters around a token to search for peers
_LEGACY_FINE_LABEL_KEYWORDS = [
    "מספר הודעת תשלום קנס",
    "הודעת תשלום קנס",
    "מספר הודעה",
    "מספר דוח",
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_multi_preprocess_enabled() -> bool:
    if os.getenv("OCR_MULTI_VARIANTS") is not None:
        return _env_flag("OCR_MULTI_VARIANTS", default=False)
    return _env_flag("OCR_MULTI_PREPROCESS", default=False)


def mask_secret(value: str | None, prefix_len: int = 7, suffix_len: int = 4) -> str:
    """Return a masked version of a secret string safe for logging.

    Shows the first *prefix_len* and last *suffix_len* characters of *value*
    so the output is recognisable without leaking the full secret.  Strings
    that are too short to mask safely are replaced with ``[REDACTED]``.

    Examples
    --------
    >>> mask_secret("sk-proj-ABCDEF1234WXYZ")
    'sk-proj...WXYZ'
    >>> mask_secret("short")
    '[REDACTED]'
    """
    if not value:
        return "[REDACTED]"
    if len(value) <= prefix_len + suffix_len:
        return "[REDACTED]"
    return f"{value[:prefix_len]}...{value[-suffix_len:]}"


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _deskew(image: np.ndarray) -> np.ndarray:
    """Rotate the image to correct small skew angles."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Return a preprocessed greyscale image suitable for Tesseract."""
    # Convert to greyscale if colour
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image.copy()

    # Denoise
    denoised = cv2.fastNlMeansDenoising(grey, h=10)

    # Adaptive threshold → clean binary image
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    # Deskew
    deskewed = _deskew(binary)

    return deskewed


def preprocess_variants(image: np.ndarray) -> List[np.ndarray]:
    """Return OCR preprocess variants: equalized gray + 2 binary variants."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh_global = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)[1]
    thresh_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return [gray, thresh_global, thresh_adaptive]


def preprocess_numeric_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for numeric OCR (fine number recovery)."""
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image.copy()

    upscaled = cv2.resize(grey, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 6
    )
    return _deskew(binary)


# ---------------------------------------------------------------------------
# Scoring heuristic
# ---------------------------------------------------------------------------

_ALLOWED_PUNCTUATION = set(' \n\t.,:-/()₪"\'' + "—–")


def score_ocr_result(text: str) -> float:
    """Return a quality score for an OCR result (higher is better)."""
    stripped = text.strip()
    if len(stripped) < 10:
        return 0.0

    score = 0.0

    # Reward longer meaningful output
    score += min(len(stripped) / 80.0, 15.0)

    # Reward Hebrew characters (common in Israeli traffic fines)
    score += sum(0.4 for c in stripped if "\u05d0" <= c <= "\u05ea")

    # Reward digits (amounts, dates, fine numbers)
    score += sum(0.2 for c in stripped if c.isdigit())

    # Penalise excessive non-alphanumeric, non-Hebrew, non-allowed characters
    noise = sum(
        1
        for c in stripped
        if not c.isalnum() and c not in _ALLOWED_PUNCTUATION and not ("\u05d0" <= c <= "\u05ea")
    )
    score -= noise * 0.15

    return max(score, 0.0)


# ---------------------------------------------------------------------------
# Tesseract runner
# ---------------------------------------------------------------------------

def _run_tesseract_on_file(image_path: str) -> str:
    """Run Tesseract with multiple PSM modes and return the best result."""
    results: List[Tuple[float, str]] = []
    for psm in TESSERACT_PSM_MODES:
        config = f"--psm {psm} --oem 3"
        try:
            text = pytesseract.image_to_string(image_path, lang=TESSERACT_LANG, config=config)
            sc = score_ocr_result(text)
            results.append((sc, text))
            logger.debug("PSM %d → score=%.2f, chars=%d", psm, sc, len(text))
        except Exception as exc:
            logger.warning("Tesseract PSM %d failed: %s", psm, exc)

    if not results:
        return ""

    best_score, best_text = max(results, key=lambda x: x[0])
    logger.info("OCR best score=%.2f from %d PSM configs", best_score, len(results))
    return best_text.strip()


def _run_tesseract_on_variants(images: List[np.ndarray]) -> str:
    """Run scored OCR over image variants * PSM modes and return best text."""
    results: List[Tuple[float, str]] = []
    for idx, image in enumerate(images):
        for psm in TESSERACT_PSM_MODES:
            config = f"--psm {psm} --oem 3"
            try:
                text = pytesseract.image_to_string(image, lang=TESSERACT_LANG, config=config)
                sc = score_ocr_result(text)
                results.append((sc, text))
                logger.debug(
                    "Variant %d PSM %d → score=%.2f, chars=%d", idx, psm, sc, len(text)
                )
            except Exception as exc:
                logger.warning("Tesseract variant %d PSM %d failed: %s", idx, psm, exc)
    if not results:
        return ""
    best_score, best_text = max(results, key=lambda x: x[0])
    logger.info("OCR best score=%.2f from %d variant*PSM configs", best_score, len(results))
    return best_text.strip()


def _run_tesseract_numeric_on_file(image_path: str) -> str:
    """Run numeric-focused Tesseract pass (digits and separators only)."""
    results: List[Tuple[int, str]] = []
    config_suffix = "-c tessedit_char_whitelist=0123456789/.-: "
    for psm in TESSERACT_NUMERIC_PSM_MODES:
        config = f"--psm {psm} --oem 3 {config_suffix}"
        try:
            text = pytesseract.image_to_string(image_path, lang="eng", config=config)
            digit_count = sum(1 for ch in text if ch.isdigit())
            results.append((digit_count, text))
            logger.debug("Numeric OCR PSM %d → digits=%d, chars=%d", psm, digit_count, len(text))
        except Exception as exc:
            logger.warning("Numeric OCR PSM %d failed: %s", psm, exc)

    if not results:
        return ""
    _, best_text = max(results, key=lambda x: x[0])
    return best_text.strip()


def _run_tesseract_numeric_on_variants(images: List[np.ndarray]) -> str:
    """Run numeric OCR over image variants * numeric PSM modes and return best text."""
    results: List[Tuple[int, str]] = []
    config_suffix = "-c tessedit_char_whitelist=0123456789"
    for idx, image in enumerate(images):
        for psm in TESSERACT_NUMERIC_PSM_MODES:
            config = f"--psm {psm} --oem 3 {config_suffix}"
            try:
                text = pytesseract.image_to_string(image, lang="eng", config=config)
                digit_count = sum(1 for ch in text if ch.isdigit())
                results.append((digit_count, text))
                logger.debug(
                    "Numeric variant %d PSM %d → digits=%d, chars=%d",
                    idx,
                    psm,
                    digit_count,
                    len(text),
                )
            except Exception as exc:
                logger.warning("Numeric OCR variant %d PSM %d failed: %s", idx, psm, exc)
    if not results:
        return ""
    _, best_text = max(results, key=lambda x: x[0])
    return best_text.strip()


def run_ocr_multi(image: np.ndarray) -> Tuple[str, str]:
    """Run fixed PSM6 OCR over preprocess variants and join outputs."""
    texts: List[str] = []
    nums: List[str] = []
    for variant in preprocess_variants(image):
        texts.append(
            pytesseract.image_to_string(variant, lang="heb+eng", config="--oem 3 --psm 6")
        )
        nums.append(
            pytesseract.image_to_string(
                variant,
                lang="eng",
                config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
            )
        )
    return "\n".join(t.strip() for t in texts if t.strip()), "\n".join(
        n.strip() for n in nums if n.strip()
    )


def _context_score(text: str, number: str, keywords: List[str], window: int = 150) -> int:
    """Score keyword proximity; same-line occurrences are weighted highest.

    Scoring per keyword occurrence:
    - Number appears on the **same line** as the keyword: ``+8``
    - Number appears within ±*window* characters (any line): ``+2``

    The same-line bonus ensures that a number printed directly beside a
    keyword (e.g. "מספר דוח: 51903219") scores far higher than a number
    that merely happens to be within the character window, even if the
    latter appears more frequently in the document.
    """
    if not number:
        return 0
    score = 0
    number_re = re.compile(
        r"(?<!\d)%s(?!\d)" % _FLEX_SEPARATORS.join(re.escape(ch) for ch in number)
    )

    # Same-line bonus (highest priority)
    for kw in keywords:
        kw_re = re.compile(
            _FLEX_SEPARATORS.join(re.escape(part) for part in kw.split()),
            re.IGNORECASE,
        )
        for line in text.splitlines():
            if kw_re.search(line) and number_re.search(line):
                score += 8

    # Window proximity (lower priority, catches multi-line layouts)
    for kw in keywords:
        kw_re = re.compile(
            _FLEX_SEPARATORS.join(re.escape(part) for part in kw.split()),
            re.IGNORECASE,
        )
        for match in kw_re.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            if number_re.search(text[start:end]):
                score += 2

    return score


def _candidate_score(
    candidate: str, counts: Counter, kind: str, text: str
) -> Tuple[int, int, bool, int]:
    """Return a comparison tuple for *candidate* (higher is better).

    The tuple is ordered so that **context proximity wins first**, breaking
    ties by frequency, then by preferred length, then by length penalty.
    This ensures that a number appearing next to a relevant keyword beats
    an unrelated number that happens to occur more often.
    """
    if kind == "plate":
        ctx = _context_score(text, candidate, PLATE_KEYWORDS)
        return (
            ctx,
            counts[candidate],
            len(candidate) in (7, 8),
            -abs(len(candidate) - 8),
        )
    ctx = _context_score(text, candidate, FINE_KEYWORDS)
    return (
        ctx,
        counts[candidate],
        len(candidate) in (8, 9),
        -abs(len(candidate) - 8),
    )


def _digits_pattern(number: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\d)%s(?!\d)" % _FLEX_SEPARATORS.join(re.escape(ch) for ch in number))


def _line_has_keyword(line: str, keyword: str) -> bool:
    kw_re = re.compile(
        _FLEX_SEPARATORS.join(re.escape(part) for part in keyword.split()),
        re.IGNORECASE,
    )
    return bool(kw_re.search(line))


def _plate_candidate_profile(text: str, candidate: str) -> Dict[str, int | bool]:
    number_re = _digits_pattern(candidate)
    lines = [line for line in text.splitlines() if number_re.search(line)]
    anchor_lines = sum(1 for line in lines if any(_line_has_keyword(line, kw) for kw in PLATE_KEYWORDS))
    narrative_lines = sum(
        1 for line in lines if any(marker in line for marker in NARRATIVE_MARKERS)
    )
    return {
        "occurrences": len(lines),
        "anchor_lines": anchor_lines,
        "narrative_lines": narrative_lines,
        "narrative_only": bool(lines and anchor_lines == 0 and narrative_lines > 0),
    }


def _amount_candidate_score(
    candidate: str, counts: Counter, ocr_text: str, date_like_values: set[str]
) -> Tuple[bool, int, bool, int, int]:
    value = int(candidate)
    amount_ctx = _context_score(ocr_text, candidate, AMOUNT_KEYWORDS, window=120)
    plausible = 20 <= value <= 20000
    return (
        candidate not in date_like_values,
        amount_ctx,
        plausible,
        counts[candidate],
        -abs(len(candidate) - 3),
    )


def _detect_type2_token_proximity(text: str) -> str | None:
    """Return a token fragment if type-2 key tokens appear near each other.

    Checks whether the root fragments of הודעת, תשלום, and קנס all appear
    within *_TYPE2_NOISY_ANCHOR_WINDOW* characters of each other, tolerating
    OCR garbling of the full label phrase.  Returns a short description of the
    matching fragment for use in log messages, or ``None`` if not found.
    """
    for match in _TYPE2_TOKEN_HODAA_RE.finditer(text):
        center = match.start()
        chunk = text[max(0, center - _TYPE2_NOISY_ANCHOR_WINDOW): center + _TYPE2_NOISY_ANCHOR_WINDOW]
        if _TYPE2_TOKEN_TASHLUM_RE.search(chunk) and _TYPE2_TOKEN_KNAS_RE.search(chunk):
            fragment = text[center: center + min(8, len(text) - center)].strip()
            return fragment[:8] if fragment else "הודע+תשלו+קנס"
    return None


def _find_first_marker(text: str, markers: list[str]) -> str | None:
    for marker in markers:
        if marker and marker in text:
            return marker
    return None


def _detect_fine_template_with_reason(ocr_text: str) -> Tuple[str, str]:
    text = ocr_text or ""
    legacy_label_present = bool(_LEGACY_NOTICE_LABEL_RE.search(text))
    if _TYPE2_FINE_LABEL_RE.search(text):
        return _ANCHOR_BASED_TEMPLATE, "explicit_type2_label"
    if _TYPE2_NOTICE_LABEL_RE.search(text):
        narrative_marker = _find_first_marker(text, NARRATIVE_MARKERS)
        if narrative_marker:
            return (
                _ANCHOR_BASED_TEMPLATE,
                f"type2_notice_with_narrative_marker:{narrative_marker}",
            )
        if not legacy_label_present:
            return _ANCHOR_BASED_TEMPLATE, "type2_notice_without_legacy_notice_label"
        return (
            _LEGACY_TEMPLATE,
            "fallback_legacy_notice:type2_notice_with_legacy_label_no_narrative_marker",
        )
    # Noise-tolerant fallback: key type-2 tokens appear near each other even
    # though the full label phrase was garbled by OCR noise.  Only triggers when
    # no legacy-specific label (מספר דוח / מספר הודעה) is present.
    noisy_fragment = _detect_type2_token_proximity(text)
    if noisy_fragment:
        if not legacy_label_present:
            return _ANCHOR_BASED_TEMPLATE, f"type2_token_proximity:{noisy_fragment}"
        return (
            _LEGACY_TEMPLATE,
            f"fallback_legacy_notice:legacy_notice_label_blocks_type2_token_proximity:{noisy_fragment}",
        )
    if legacy_label_present:
        return _LEGACY_TEMPLATE, "fallback_legacy_notice:legacy_notice_label"
    return _LEGACY_TEMPLATE, "fallback_legacy_notice:no_type2_markers"


def _detect_fine_template(ocr_text: str) -> str:
    """Choose extraction template from stable OCR notice labels.

    ``מספר הודעת תשלום קנס`` is a stable anchor for the problematic notice
    variant; all other notices keep the legacy extraction behavior.
    """
    return _detect_fine_template_with_reason(ocr_text)[0]


def extract_plate_and_fine_candidates(ocr_text: str, numeric_text: str) -> Dict[str, Any]:
    """Extract plate/fine heuristics from OCR text blocks.

    Selection rules
    ---------------
    * Plate candidates must be exactly 7 or 8 digits and are prioritized when
      they appear near explicit vehicle-number anchors (for example ``מס רכב``).
    * Fine candidates are normally 7–10 digits. For type #2 notices where the
      label ``מספר הודעת תשלום קנס`` is present, fine candidates are restricted
      to 10–11 digits and numbers close to ``תעודת זהות`` are ignored.
    * Amount candidates are short numeric values near amount anchors (for
      example ``סכום`` / ``סך`` / ``קנס``); date-shaped values are rejected.
    * **Context is required**: a candidate is only accepted when it appears
      within ±150 characters of at least one keyword from the relevant set
      (``PLATE_KEYWORDS`` / ``FINE_KEYWORDS``).  If no candidate satisfies
      this constraint the field is returned as ``None`` so that the LLM
      fallback can run.
    * **No duplicates**: plate and fine must differ.  If they somehow end up
      equal, the field with the higher context score keeps its value; the
      other is set to ``None`` to trigger LLM fallback.
    * Context score is the **primary** sort key; frequency (Counter) and
      length preference are tie-breakers.

    Debug logging
    -------------
    Set the environment variable ``OCR_DEBUG=1`` to emit detailed candidate
    scores at ``DEBUG`` level.  Secrets (e.g. ``OPENAI_API_KEY``) are never
    included in these logs; use :func:`mask_secret` for any sensitive value
    you need to reference.
    """
    full_text = f"{ocr_text}\n{numeric_text}"
    raw = re.findall(r"\d[\d \t\-./,:]{4,16}\d", full_text)
    cleaned = ["".join(ch for ch in token if ch.isdigit()) for token in raw]
    cleaned = [n for n in cleaned if 6 <= len(n) <= 12 and len(set(n)) > 2]

    debug = _env_flag("OCR_DEBUG", default=False)

    if not cleaned:
        if debug:
            logger.debug("OCR heuristic: no numeric candidates found")
        return {
            "plate": None,
            "fine": None,
            "amount": None,
            "plate_confident": False,
            "fine_confident": False,
            "amount_confident": False,
        }

    counts: Counter = Counter(cleaned)
    template, template_reason = _detect_fine_template_with_reason(ocr_text)
    logger.info("Fine OCR routing template_detected=%s reason=%s", template, template_reason)

    if debug:
        top_n = counts.most_common(10)
        logger.debug("OCR heuristic candidates (top %d): %s", len(top_n), top_n)
        plate_anchor_hits: list[tuple[str, str]] = []
        for line in ocr_text.splitlines():
            for kw in PLATE_KEYWORDS:
                if _line_has_keyword(line, kw):
                    plate_anchor_hits.append((kw, line.strip()[:120]))
                    break
        if plate_anchor_hits:
            logger.debug("  plate anchor hits: %s", plate_anchor_hits[:8])
        else:
            logger.debug("  plate anchor hits: none")

    # --- Plate selection ---------------------------------------------------
    plate_candidates = [n for n in cleaned if len(n) in (7, 8)]
    plate_profiles = (
        {n: _plate_candidate_profile(ocr_text, n) for n in set(plate_candidates)}
        if template == _ANCHOR_BASED_TEMPLATE
        else {}
    )

    def _plate_rank(n: str) -> Tuple[int, ...]:
        base_score = _candidate_score(n, counts, "plate", ocr_text)
        if template != _ANCHOR_BASED_TEMPLATE:
            return base_score
        profile = plate_profiles.get(n, {})
        return (
            base_score[0],
            not bool(profile.get("narrative_only", False)),
            base_score[1],
            base_score[2],
            base_score[3],
        )

    if debug:
        for n in sorted(set(plate_candidates), key=_plate_rank, reverse=True)[:8]:
            profile = plate_profiles.get(n, {})
            if template == _ANCHOR_BASED_TEMPLATE:
                logger.debug(
                    "  plate candidate=%s rank=%s occurrences=%s anchor_lines=%s narrative_lines=%s narrative_only=%s",
                    n,
                    _plate_rank(n),
                    profile.get("occurrences", 0),
                    profile.get("anchor_lines", 0),
                    profile.get("narrative_lines", 0),
                    profile.get("narrative_only", False),
                )
            else:
                logger.debug("  plate candidate=%s rank=%s", n, _plate_rank(n))

    best_plate_pre: str | None = (
        max(plate_candidates, key=_plate_rank)
        if plate_candidates
        else None
    )

    # Context requirement for plate
    plate_ctx = _context_score(ocr_text, best_plate_pre or "", PLATE_KEYWORDS) if best_plate_pre else 0
    if best_plate_pre and plate_ctx == 0:
        if debug:
            profile = plate_profiles.get(best_plate_pre, {})
            if template == _ANCHOR_BASED_TEMPLATE and profile.get("narrative_only"):
                logger.debug(
                    "  plate %s rejected: narrative/body candidate without labeled anchor",
                    best_plate_pre,
                )
            else:
                logger.debug("  plate %s rejected: no context proximity", best_plate_pre)
        best_plate: str | None = None
        plate_ctx = 0
    else:
        best_plate = best_plate_pre

    date_like_values = {
        "".join(ch for ch in m.group() if ch.isdigit())
        for m in _DATE_TOKEN_RE.finditer(full_text)
    }

    # --- Fine selection ----------------------------------------------------
    # Exclude best_plate (the accepted plate, after context check) to prevent
    # the same number occupying both slots.  If best_plate was rejected (None),
    # no number is excluded so all 7-10 digit candidates can compete.
    if template == _ANCHOR_BASED_TEMPLATE:
        fine_candidates = []
        for n in cleaned:
            if n == best_plate or len(n) not in (10, 11):
                continue
            number_re = re.compile(
                r"(?<!\d)%s(?!\d)" % _FLEX_SEPARATORS.join(re.escape(ch) for ch in n)
            )
            near_tz = any(
                _TZ_LABEL_RE.search(line) and number_re.search(line)
                for line in ocr_text.splitlines()
            )
            if not near_tz:
                fine_candidates.append(n)
    else:
        fine_candidates = [n for n in cleaned if 7 <= len(n) <= 10 and n != best_plate]
        labeled_notice_candidates: list[str] = []
        for idx, line in enumerate(ocr_text.splitlines()):
            if not any(_line_has_keyword(line, kw) for kw in _LEGACY_FINE_LABEL_KEYWORDS):
                continue
            scope_start = max(0, idx - 1)
            scope_end = min(len(ocr_text.splitlines()), idx + 2)
            scope = "\n".join(ocr_text.splitlines()[scope_start:scope_end])
            for token in re.findall(r"\d[\d \t\-./,:]{4,16}\d", scope):
                candidate = "".join(ch for ch in token if ch.isdigit())
                if len(candidate) not in (7, 8, 9, 10) or candidate == best_plate:
                    continue
                number_re = _digits_pattern(candidate)
                near_tz = any(
                    _TZ_LABEL_RE.search(scope_line) and number_re.search(scope_line)
                    for scope_line in scope.splitlines()
                )
                if near_tz or candidate in date_like_values:
                    continue
                labeled_notice_candidates.append(candidate)
        if labeled_notice_candidates:
            fine_candidates = labeled_notice_candidates + fine_candidates

    if debug:
        for n in sorted(set(fine_candidates), key=lambda x: _candidate_score(x, counts, "fine", ocr_text), reverse=True)[:5]:
            sc = _candidate_score(n, counts, "fine", ocr_text)
            logger.debug("  fine candidate %s score=%s", n, sc)

    best_fine_pre: str | None = (
        max(
            fine_candidates,
            key=lambda n: (
                n not in date_like_values,
                _candidate_score(n, counts, "fine", ocr_text),
            ),
        )
        if fine_candidates
        else None
    )

    # Context requirement for fine
    fine_ctx = _context_score(ocr_text, best_fine_pre or "", FINE_KEYWORDS) if best_fine_pre else 0
    if best_fine_pre and fine_ctx == 0:
        if debug:
            logger.debug("  fine %s rejected: no context proximity", best_fine_pre)
        best_fine: str | None = None
        fine_ctx = 0
    else:
        best_fine = best_fine_pre

    # --- Fine amount selection ----------------------------------------------
    amount_raw = re.findall(r"(?<!\d)(?:\d[ \t\-./,:]?){0,5}\d(?!\d)", full_text)
    amount_cleaned = ["".join(ch for ch in token if ch.isdigit()) for token in amount_raw]
    amount_cleaned = [n for n in amount_cleaned if 1 <= len(n) <= 6]
    amount_counts: Counter = Counter(amount_cleaned)
    amount_candidates = [n for n in amount_cleaned if n != best_plate and n != best_fine]
    amount_candidates = [n for n in amount_candidates if int(n) > 0]

    if debug:
        for n in sorted(
            set(amount_candidates),
            key=lambda x: _amount_candidate_score(x, amount_counts, ocr_text, date_like_values),
            reverse=True,
        )[:8]:
            logger.debug(
                "  amount candidate=%s rank=%s rejected_date=%s",
                n,
                _amount_candidate_score(n, amount_counts, ocr_text, date_like_values),
                n in date_like_values,
            )

    best_amount_pre: str | None = (
        max(
            amount_candidates,
            key=lambda n: _amount_candidate_score(n, amount_counts, ocr_text, date_like_values),
        )
        if amount_candidates
        else None
    )
    amount_ctx = _context_score(ocr_text, best_amount_pre or "", AMOUNT_KEYWORDS, window=120) if best_amount_pre else 0
    if best_amount_pre and best_amount_pre in date_like_values:
        if debug:
            logger.debug("  amount %s rejected: date-like token", best_amount_pre)
        best_amount = None
        amount_ctx = 0
    elif best_amount_pre and amount_ctx == 0:
        if debug:
            logger.debug("  amount %s selected with weak anchor proximity", best_amount_pre)
        best_amount = best_amount_pre
    else:
        best_amount = best_amount_pre

    # --- Tie-break if plate == fine ----------------------------------------
    if best_plate and best_fine and best_plate == best_fine:
        if debug:
            logger.debug(
                "  plate and fine equal (%s): breaking tie by context score"
                " (plate_ctx=%d, fine_ctx=%d)",
                best_plate,
                plate_ctx,
                fine_ctx,
            )
        if plate_ctx >= fine_ctx:
            if debug:
                logger.debug("  plate wins tie-break → fine set to None")
            best_fine = None
            fine_ctx = 0
        else:
            if debug:
                logger.debug("  fine wins tie-break → plate set to None")
            best_plate = None
            plate_ctx = 0

    plate_confident = bool(best_plate and counts[best_plate] >= 2 and plate_ctx >= 2)
    fine_confident = bool(best_fine and counts[best_fine] >= 2 and fine_ctx >= 2)
    amount_confident = bool(best_amount and (amount_ctx >= 2 or 20 <= int(best_amount) <= 20000))

    if debug:
        logger.debug(
            "OCR heuristic winners: plate=%s (ctx=%d, confident=%s),"
            " fine=%s (ctx=%d, confident=%s), amount=%s (ctx=%d, confident=%s)",
            best_plate,
            plate_ctx,
            plate_confident,
            best_fine,
            fine_ctx,
            fine_confident,
            best_amount,
            amount_ctx,
            amount_confident,
        )

    return {
        "plate": best_plate,
        "fine": best_fine,
        "amount": best_amount,
        "plate_confident": plate_confident,
        "fine_confident": fine_confident,
        "amount_confident": amount_confident,
        "plate_ctx": plate_ctx,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr_image(image_path: str) -> str:
    """OCR a raster image file (JPG / PNG)."""
    text, _ = ocr_image_with_numeric(image_path)
    return text


def ocr_image_with_numeric(image_path: str) -> Tuple[str, str]:
    """OCR a raster image and return ``(general_text, numeric_text)``.

    ``general_text`` is produced from the standard heb+eng OCR pipeline.
    ``numeric_text`` is produced from a numeric-focused OCR pass for better
    fine-number recovery.

    Returns
    -------
    Tuple[str, str]
        ``(general_text, numeric_text)``. Any failed OCR pass returns an empty
        string for that element.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    if _is_multi_preprocess_enabled():
        variants = preprocess_variants(img)
        for idx, variant in enumerate(variants):
            write_image(f"preprocessed_variant_{idx}.png", variant)
        best_general = _run_tesseract_on_variants(variants)
        best_numeric = _run_tesseract_numeric_on_variants(variants)
        multi_general, multi_numeric = run_ocr_multi(img)
        general_text = "\n".join(part for part in (best_general, multi_general) if part).strip()
        numeric_text = "\n".join(part for part in (best_numeric, multi_numeric) if part).strip()
        return general_text, numeric_text

    preprocessed = preprocess_image(img)
    numeric_preprocessed = preprocess_numeric_image(img)
    write_image("preprocessed_general.png", preprocessed)
    write_image("preprocessed_numeric.png", numeric_preprocessed)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_general:
        general_path = tmp_general.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_numeric:
        numeric_path = tmp_numeric.name
    try:
        cv2.imwrite(general_path, preprocessed)
        cv2.imwrite(numeric_path, numeric_preprocessed)
        return _run_tesseract_on_file(general_path), _run_tesseract_numeric_on_file(numeric_path)
    finally:
        for path in (general_path, numeric_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def ocr_pdf(pdf_path: str) -> str:
    """Convert the first page of a PDF to PNG and OCR it."""
    text, _ = ocr_pdf_with_numeric(pdf_path)
    return text


def ocr_pdf_with_numeric(pdf_path: str) -> Tuple[str, str]:
    """OCR the first PDF page and return ``(general_text, numeric_text)``.

    The PDF page is converted to PNG and then processed by both the standard
    OCR pipeline and a numeric-focused OCR pipeline.

    Returns
    -------
    Tuple[str, str]
        ``(general_text, numeric_text)``.
    """
    from pdf2image import convert_from_path  # imported here to keep startup light

    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
    if not pages:
        raise ValueError("PDF produced no pages")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pages[0].save(tmp_path, "PNG")
        return ocr_image_with_numeric(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_text(file_path: str) -> str:
    """Dispatch to the correct OCR function based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return ocr_pdf(file_path)
    if ext in {".jpg", ".jpeg", ".png"}:
        return ocr_image(file_path)
    raise ValueError(f"Unsupported file extension for OCR: {ext}")


def extract_text_with_numeric(file_path: str) -> Tuple[str, str]:
    """Dispatch OCR and return (general_text, numeric_text)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return ocr_pdf_with_numeric(file_path)
    if ext in {".jpg", ".jpeg", ".png"}:
        return ocr_image_with_numeric(file_path)
    raise ValueError(f"Unsupported file extension for OCR: {ext}")

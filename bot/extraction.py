"""GPT-first structured extraction pipeline for Israeli traffic fine fields.

Architecture
------------
1. OCR produces text; typed candidates (plate / fine) are collected from it.
2. Primary GPT call performs structured JSON extraction (plate, fine,
   per-field confidence, source).
3. Local validation checks extracted fields.
4. If a field is missing or invalid, a targeted second GPT call is made for
   that field only.
5. Final local validation.
5b. OCR heuristic fallback: if a field is still None but the caller supplied
    an OCR heuristic value that passes validation, that value is used instead
    of discarding it.
6. Typed OCR cross-check before accepting the final value: plate candidates
   are cross-checked against plate-length (7–8 digit) OCR sequences only;
   fine candidates are cross-checked against fine-length OCR sequences only
   (8 digits for legacy, 10–11 digits for type-2 notices).  Vision-sourced
   fields bypass the cross-check.

Merge priority
--------------
vision GPT > primary GPT (valid) > retry GPT (valid) > OCR heuristic fallback

This module is the single entry point for the extraction pipeline.
Everything else (heuristics, per-stage overrides) is deliberately kept
outside this flow to maintain predictable, traceable behavior.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Digit utilities
# ---------------------------------------------------------------------------

_STRIP_NON_DIGITS = re.compile(r"\D")
# Matches sequences of digits optionally separated by spaces/hyphens/dots
_OCR_DIGIT_SEQUENCE = re.compile(
    r"(?<!\d)(?:\d[ \t\-./]?){5,11}\d(?!\d)"
)
# Type-2 notice marker – explicit label form
_TYPE2_RE = re.compile(
    r"מספר[ \t\-./,:_]*הודעת[ \t\-./,:_]*תשלום[ \t\-./,:_]*קנס",
    re.IGNORECASE,
)
# Type-2 detection via decision-notice anchor markers (used by OCR routing)
_DECISION_NOTICE_ANCHOR_RE = re.compile(
    r"הודעה\s+על\s+ה?החלטה|תעודת\s+עובד\s+הציבור|תאו[ר]?\s+העובדות\s+המהוות",
    re.IGNORECASE,
)


def _digits(value: str) -> str:
    """Return digits-only string."""
    return _STRIP_NON_DIGITS.sub("", value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_plate(value: Optional[str]) -> bool:
    """Israeli vehicle plate: 7 or 8 digits (older/newer format)."""
    if not value:
        return False
    d = _digits(value)
    return len(d) in (7, 8)


def is_valid_fine(value: Optional[str], *, is_type2: bool = False) -> bool:
    """Fine-number validation.

    - Legacy notice (municipal / parking): exactly 8 digits.
    - Type-2 notice (מספר הודעת תשלום קנס): 10 or 11 digits.
    """
    if not value:
        return False
    d = _digits(value)
    if is_type2:
        return len(d) in (10, 11)
    return len(d) == 8


def detect_type2(ocr_text: str) -> bool:
    """Return True when OCR text contains a type-2 / decision-notice indicator.

    Two complementary signals are checked so that garbled OCR still yields the
    correct template even when the full label ``מספר הודעת תשלום קנס`` is not
    readable:

    1. The explicit type-2 label ``מספר הודעת תשלום קנס`` (any flex spacing).
    2. Decision-notice anchor markers used by the OCR routing stage:
       - ``הודעה על החלטה``
       - ``תעודת עובד הציבור``
       - ``תאור העובדות המהוות``

    Either signal is sufficient.
    """
    text = ocr_text or ""
    if _TYPE2_RE.search(text):
        return True
    return bool(_DECISION_NOTICE_ANCHOR_RE.search(text))


# ---------------------------------------------------------------------------
# OCR candidate collection
# ---------------------------------------------------------------------------


def extract_ocr_candidates(ocr_text: str, numeric_text: str = "") -> List[str]:
    """Collect all 6–12 digit sequences from OCR texts.

    Handles sequences split by spaces, hyphens, dots, and similar OCR
    artefacts that are common in Israeli fine documents.
    """
    combined = f"{ocr_text}\n{numeric_text}"
    seen: Dict[str, None] = {}
    for m in _OCR_DIGIT_SEQUENCE.finditer(combined):
        val = _digits(m.group())
        if 6 <= len(val) <= 12:
            seen[val] = None
    return list(seen)


def extract_plate_ocr_candidates(ocr_text: str, numeric_text: str = "") -> List[str]:
    """Collect 7–8 digit sequences (plate-length) from OCR texts.

    Only plate-length sequences are returned so that a GPT plate candidate
    cannot be validated against a longer ID or phone number even when the
    shorter number is a prefix of the longer one.
    """
    combined = f"{ocr_text}\n{numeric_text}"
    seen: Dict[str, None] = {}
    for m in _OCR_DIGIT_SEQUENCE.finditer(combined):
        val = _digits(m.group())
        if len(val) in (7, 8):
            seen[val] = None
    return list(seen)


def extract_fine_ocr_candidates(
    ocr_text: str, numeric_text: str = "", *, is_type2: bool = False
) -> List[str]:
    """Collect fine-length digit sequences from OCR texts.

    - Legacy template (is_type2=False): exactly 8-digit sequences.
    - Type-2 template (is_type2=True): 10 or 11-digit sequences.

    Using typed candidates prevents a fine number from matching against a
    plate-length or ID-length sequence of a different category.
    """
    combined = f"{ocr_text}\n{numeric_text}"
    seen: Dict[str, None] = {}
    for m in _OCR_DIGIT_SEQUENCE.finditer(combined):
        val = _digits(m.group())
        if is_type2:
            if len(val) in (10, 11):
                seen[val] = None
        else:
            if len(val) == 8:
                seen[val] = None
    return list(seen)


def normalized_candidate_match(value: Optional[str], candidates: List[str]) -> bool:
    """Check whether *value* can be grounded in at least one OCR candidate.

    Uses digit-only normalisation so OCR splits like ``51 903 219`` match the
    clean form ``51903219``.  Also accepts containment (either direction) for
    cases where OCR merges adjacent numbers or reads only part of one.
    """
    if not value:
        return False
    val_d = _digits(value)
    if not val_d or len(val_d) < 6:
        return False
    for c in candidates:
        c_d = _digits(c)
        if not c_d:
            continue
        if val_d == c_d:
            return True
        # Containment in either direction (both at least 7 digits to avoid
        # false positives from very short common substrings)
        if len(val_d) >= 7 and len(c_d) >= 7:
            if val_d in c_d or c_d in val_d:
                return True
    return False


# ---------------------------------------------------------------------------
# Vision-sourced value markers
# ---------------------------------------------------------------------------

_VISION_SOURCES = frozenset({"gpt_vision", "vision"})


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def extract_all(
    ocr_text: str,
    numeric_ocr_text: str = "",
    *,
    gpt_extract_fn=None,
    gpt_retry_plate_fn=None,
    gpt_retry_fine_fn=None,
    is_type2: Optional[bool] = None,
    ocr_plate: Optional[str] = None,
    ocr_fine: Optional[str] = None,
) -> Dict[str, Any]:
    """GPT-first extraction pipeline for vehicle plate and fine number.

    Parameters
    ----------
    ocr_text:
        General OCR text from the document.
    numeric_ocr_text:
        Numeric-focused OCR text (optional).
    gpt_extract_fn:
        Primary GPT extraction callable ``async (ocr_text) -> dict``.
        Defaults to :func:`bot.openai_client.gpt_extract_structured`.
    gpt_retry_plate_fn:
        Targeted plate retry callable ``async (ocr_text) -> dict``.
        Defaults to :func:`bot.openai_client.gpt_retry_plate`.
    gpt_retry_fine_fn:
        Targeted fine retry callable ``async (ocr_text) -> dict``.
        Defaults to :func:`bot.openai_client.gpt_retry_fine`.
    is_type2:
        Override the type-2 detection.  When *None* (default) the template
        is auto-detected from *ocr_text* by :func:`detect_type2`.  Pass
        ``True`` or ``False`` to force a specific mode (e.g. when the OCR
        routing stage has already determined the template more reliably).
    ocr_plate:
        OCR heuristic plate candidate from the pre-pipeline routing stage.
        Used as a fallback when the GPT pipeline cannot produce a valid plate.
    ocr_fine:
        OCR heuristic fine candidate from the pre-pipeline routing stage.
        Used as a fallback when the GPT pipeline cannot produce a valid fine.

    Returns
    -------
    dict
        ``{"plate": str|None, "fine": str|None, "confidence": dict, "source": dict}``

    Merge priority
    --------------
    vision GPT > primary GPT (valid) > retry GPT (valid) > OCR heuristic fallback
    """
    # Lazy imports to avoid circular dependency at module load time
    from bot.openai_client import (  # noqa: PLC0415
        gpt_extract_structured as _gpt_extract,
        gpt_retry_fine as _gpt_retry_fine,
        gpt_retry_plate as _gpt_retry_plate,
    )

    if gpt_extract_fn is None:
        gpt_extract_fn = _gpt_extract
    if gpt_retry_plate_fn is None:
        gpt_retry_plate_fn = _gpt_retry_plate
    if gpt_retry_fine_fn is None:
        gpt_retry_fine_fn = _gpt_retry_fine

    # Resolve is_type2: use caller-supplied value or auto-detect
    if is_type2 is None:
        is_type2 = detect_type2(ocr_text)

    # Normalise caller-supplied OCR heuristic candidates
    ocr_plate_d: Optional[str] = _digits(str(ocr_plate or "")) or None
    ocr_fine_d: Optional[str] = _digits(str(ocr_fine or "")) or None

    # ── Step 1: Collect typed OCR candidates for cross-checking ────────────
    plate_candidates = extract_plate_ocr_candidates(ocr_text, numeric_ocr_text)
    fine_candidates = extract_fine_ocr_candidates(
        ocr_text, numeric_ocr_text, is_type2=is_type2
    )
    logger.info(
        "Extraction pipeline start: is_type2=%s plate_candidates=%d fine_candidates=%d"
        " ocr_plate=%r ocr_fine=%r",
        is_type2,
        len(plate_candidates),
        len(fine_candidates),
        ocr_plate_d,
        ocr_fine_d,
    )

    # ── Step 2: Primary GPT structured extraction ───────────────────────────
    gpt_result = await gpt_extract_fn(ocr_text)
    plate: Optional[str] = _digits(str(gpt_result.get("plate") or "")) or None
    fine: Optional[str] = _digits(str(gpt_result.get("fine") or "")) or None
    confidence: Dict[str, Any] = dict(gpt_result.get("confidence") or {})
    source: Dict[str, str] = dict(gpt_result.get("source") or {})
    source.setdefault("plate", "gpt_primary")
    source.setdefault("fine", "gpt_primary")

    logger.info(
        "Primary GPT result: plate=%r fine=%r confidence=%s",
        plate,
        fine,
        confidence,
    )

    # ── Step 3: Local validation ────────────────────────────────────────────
    if not is_valid_plate(plate):
        logger.info("Primary plate %r invalid; will retry", plate)
        plate = None
    if not is_valid_fine(fine, is_type2=is_type2):
        logger.info("Primary fine %r invalid (is_type2=%s); will retry", fine, is_type2)
        fine = None

    # Prevent same value in both slots
    if plate and fine and plate == fine:
        logger.info("Plate and fine identical (%r); clearing fine", fine)
        fine = None

    # ── Step 4: Targeted retry for missing fields ───────────────────────────
    if not plate:
        logger.info("Running targeted GPT retry for plate")
        plate_retry = await gpt_retry_plate_fn(ocr_text)
        if isinstance(plate_retry, dict):
            plate = _digits(str(plate_retry.get("plate") or "")) or None
            retry_conf = plate_retry.get("confidence")
            if isinstance(retry_conf, dict):
                confidence["plate"] = float(retry_conf.get("plate", 0.6))
            elif isinstance(retry_conf, (int, float)):
                confidence["plate"] = float(retry_conf)
            else:
                confidence["plate"] = 0.6
        else:
            plate = _digits(str(plate_retry or "")) or None
        if plate:
            source["plate"] = "gpt_retry"
        logger.info("Plate retry result: plate=%r", plate)

    if not fine:
        logger.info("Running targeted GPT retry for fine")
        fine_retry = await gpt_retry_fine_fn(ocr_text)
        if isinstance(fine_retry, dict):
            fine = _digits(str(fine_retry.get("fine") or "")) or None
            retry_conf = fine_retry.get("confidence")
            if isinstance(retry_conf, dict):
                confidence["fine"] = float(retry_conf.get("fine", 0.6))
            elif isinstance(retry_conf, (int, float)):
                confidence["fine"] = float(retry_conf)
            else:
                confidence["fine"] = 0.6
        else:
            fine = _digits(str(fine_retry or "")) or None
        if fine:
            source["fine"] = "gpt_retry"
        logger.info("Fine retry result: fine=%r", fine)

    # ── Step 5: Final validation after retry ───────────────────────────────
    if not is_valid_plate(plate):
        logger.info("Plate %r still invalid after retry; clearing", plate)
        plate = None
    if not is_valid_fine(fine, is_type2=is_type2):
        logger.info("Fine %r still invalid after retry; clearing", fine)
        fine = None

    # Prevent same value after retry
    if plate and fine and plate == fine:
        logger.info("Plate and fine identical after retry (%r); clearing fine", fine)
        fine = None

    # ── Step 5b: OCR heuristic fallback ────────────────────────────────────
    # When GPT (primary + retry) could not produce a valid field, use the
    # OCR heuristic candidate supplied by the routing stage – provided it
    # passes the same validation rules.  This prevents the pipeline from
    # discarding a correct OCR result just because GPT returned garbage.
    if not plate and ocr_plate_d:
        if is_valid_plate(ocr_plate_d):
            logger.info(
                "Plate: GPT pipeline returned None; using OCR heuristic fallback"
                " plate=%r (source=ocr_heuristic)",
                ocr_plate_d,
            )
            plate = ocr_plate_d
            source["plate"] = "ocr_heuristic"
            confidence.setdefault("plate", 0.7)
        else:
            logger.info(
                "Plate: OCR heuristic fallback %r failed validation; keeping None",
                ocr_plate_d,
            )

    if not fine and ocr_fine_d:
        if is_valid_fine(ocr_fine_d, is_type2=is_type2):
            logger.info(
                "Fine: GPT pipeline returned None; using OCR heuristic fallback"
                " fine=%r (source=ocr_heuristic, is_type2=%s)",
                ocr_fine_d,
                is_type2,
            )
            fine = ocr_fine_d
            source["fine"] = "ocr_heuristic"
            confidence.setdefault("fine", 0.7)
        else:
            logger.info(
                "Fine: OCR heuristic fallback %r failed validation (is_type2=%s);"
                " keeping None",
                ocr_fine_d,
                is_type2,
            )

    # ── Step 6: Typed OCR cross-check ──────────────────────────────────────
    # Vision-sourced values are image-grounded and bypass the OCR text check.
    # OCR heuristic fallback values came from the OCR text already; they are
    # also exempt from the text cross-check.
    if plate and source.get("plate") not in _VISION_SOURCES and source.get("plate") != "ocr_heuristic":
        if not normalized_candidate_match(plate, plate_candidates):
            logger.info(
                "Plate %r rejected by typed OCR cross-check"
                " (plate_candidates: %s)",
                plate,
                plate_candidates[:10],
            )
            # Try OCR heuristic as a safe replacement before giving up
            if ocr_plate_d and is_valid_plate(ocr_plate_d):
                logger.info(
                    "Plate: cross-check rejected GPT plate %r;"
                    " falling back to OCR heuristic plate=%r",
                    plate,
                    ocr_plate_d,
                )
                plate = ocr_plate_d
                source["plate"] = "ocr_heuristic"
                confidence["plate"] = 0.7
            else:
                plate = None

    if fine and source.get("fine") not in _VISION_SOURCES and source.get("fine") != "ocr_heuristic":
        if not normalized_candidate_match(fine, fine_candidates):
            logger.info(
                "Fine %r rejected by typed OCR cross-check"
                " (fine_candidates: %s, is_type2=%s)",
                fine,
                fine_candidates[:10],
                is_type2,
            )
            # Try OCR heuristic as a safe replacement before giving up
            if ocr_fine_d and is_valid_fine(ocr_fine_d, is_type2=is_type2):
                logger.info(
                    "Fine: cross-check rejected GPT fine %r;"
                    " falling back to OCR heuristic fine=%r",
                    fine,
                    ocr_fine_d,
                )
                fine = ocr_fine_d
                source["fine"] = "ocr_heuristic"
                confidence["fine"] = 0.7
            else:
                fine = None

    logger.info(
        "Extraction pipeline result: plate=%r fine=%r source=%s confidence=%s",
        plate,
        fine,
        source,
        confidence,
    )
    return {
        "plate": plate,
        "fine": fine,
        "confidence": confidence,
        "source": source,
    }

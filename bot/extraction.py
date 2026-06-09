"""GPT-first structured extraction pipeline for Israeli traffic fine fields.

Architecture
------------
1. OCR produces text; raw digit candidates are collected from it.
2. Primary GPT call performs structured JSON extraction (plate, fine,
   per-field confidence, source).
3. Local validation checks extracted fields.
4. If a field is missing or invalid, a targeted second GPT call is made for
   that field only.
5. Final local validation.
6. OCR cross-check before accepting the final value (skipped for
   vision-sourced fields which are already image-grounded).

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
# Type-2 notice marker
_TYPE2_RE = re.compile(
    r"מספר[ \t\-./,:_]*הודעת[ \t\-./,:_]*תשלום[ \t\-./,:_]*קנס",
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
    """Return True when OCR text contains the type-2 notice marker."""
    return bool(_TYPE2_RE.search(ocr_text or ""))


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

    Returns
    -------
    dict
        ``{"plate": str|None, "fine": str|None, "confidence": dict, "source": dict}``
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

    is_type2 = detect_type2(ocr_text)

    # ── Step 1: Collect OCR candidates for cross-checking ──────────────────
    candidates = extract_ocr_candidates(ocr_text, numeric_ocr_text)
    logger.info(
        "Extraction pipeline start: is_type2=%s ocr_candidates=%d",
        is_type2,
        len(candidates),
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

    # ── Step 6: OCR cross-check ─────────────────────────────────────────────
    # Vision-sourced values are image-grounded and bypass the OCR text check.
    if plate and source.get("plate") not in _VISION_SOURCES:
        if not normalized_candidate_match(plate, candidates):
            logger.info(
                "Plate %r rejected by OCR cross-check (candidates: %s)",
                plate,
                candidates[:10],
            )
            plate = None

    if fine and source.get("fine") not in _VISION_SOURCES:
        if not normalized_candidate_match(fine, candidates):
            logger.info(
                "Fine %r rejected by OCR cross-check (candidates: %s)",
                fine,
                candidates[:10],
            )
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

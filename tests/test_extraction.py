"""Regression tests for the GPT-first extraction pipeline (bot.extraction).

These tests use realistic OCR samples from production logs and mock GPT
functions to isolate pipeline logic from live API calls.

Template 1 – Legacy municipal fine notice (חניה / parking):
  Correct plate: 6486471 (7 digits)
  Correct fine:  51903219 (8 digits)
  OCR issue: the fine number appears garbled in OCR ("19052 19") so it
  must be recovered via vision or a targeted retry.

Template 2 – Anchor-based decision notice (תעודת עובד הציבור):
  Correct plate: 05911509 (8 digits, from "05911 5-09" in OCR)
  Correct fine:  30850005064 (11 digits, type-2 format)
  OCR issue: multiple ID-like numbers (7345742623, 2895338) must NOT
  be chosen as the vehicle plate.
"""
import asyncio
from typing import Any, Dict, List, Optional

import pytest

from bot.extraction import (
    detect_type2,
    extract_all,
    extract_fine_ocr_candidates,
    extract_ocr_candidates,
    extract_plate_ocr_candidates,
    is_valid_fine,
    is_valid_plate,
    normalized_candidate_match,
)

# ---------------------------------------------------------------------------
# Realistic OCR samples from production logs
# ---------------------------------------------------------------------------

# Template 1 – Legacy municipal notice.  The fine number "51903219" does NOT
# appear verbatim in this OCR; it is obtained via vision or targeted retry.
TEMPLATE1_OCR = (
    "wre\n"
    "op ל\n"
    "areas\n"
    "wrt\n"
    "הבר\n"
    "ו\n"
    "שר\n"
    "ד\n"
    "=n\n"
    "ב\n"
    "19052 19\n"
    "וה\n"
    "3/4/2023\n"
    "אקום הטביוה: חול\n"
    "חניוך בזק אטוק 3\n"
    "מטפר רבב\n"
    "6 רכב\n"
    "6486471\n"
    "ig\n"
    "a1 צבע\n"
    "לבר\n"
    "יצרן רכב\n"
    "mia\n"
    "עבירן\n"
    "- 133\n"
    "וומנת/הטחדת רכבך\n"
    "בחקום wean בתאלום\n"
    ",113723 להודאוה\n"
    "~f\n"
    "spon\n"
    'גובה הקנם בט"ח: 100\n'
    "אופן חסירה:\n"
    "הצמדוה\n"
    "הערות ההפקח\n"
    "7ר-\n"
    "2\n"
    "קיים דוח התראוג מונה\n"
    "חנייה\n"
    "ow מפקו\n"
    "aan דוד\n"
    "ווטלום הקנם לא יאוחר\n"
    "חיום: 2/7/2023\n"
)

# Template 2 – Anchor-based (type-2) decision notice.
TEMPLATE2_OCR = (
    '——— הודעה על החלטה להטיל קנס/תעבזרה - "תעודת עובד הציבור"\n'
    "מספר הודעוו\n"
    "30850005064\n"
    "7345742623\n"
    "2895338 (nw) ron wo|\n"
    "Aaleall [Sc rll 05911 5-09 mona. nx ב 3 תאורהעובדות המהוות\n"
    "תאריך 16/00/2028\n"
    "16:53 nwa\n"
    "בכביש 46\n"
    "מספר הודעת תשלום קנס: 30850005064\n"
)

# ---------------------------------------------------------------------------
# Helper: no-op GPT functions that return empty results
# ---------------------------------------------------------------------------


async def _noop_retry_plate(text: str) -> Dict[str, Any]:
    return {"plate": None, "confidence": {"plate": 0.0}}


async def _noop_retry_fine(text: str) -> Dict[str, Any]:
    return {"fine": None, "confidence": {"fine": 0.0}}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_is_valid_plate_7_digits():
    assert is_valid_plate("6486471") is True


def test_is_valid_plate_8_digits():
    assert is_valid_plate("05911509") is True


def test_is_valid_plate_rejects_9_digits():
    assert is_valid_plate("7345742623") is False


def test_is_valid_plate_rejects_none():
    assert is_valid_plate(None) is False


def test_is_valid_plate_rejects_6_digits():
    assert is_valid_plate("123456") is False


def test_is_valid_fine_8_digits_legacy():
    assert is_valid_fine("51903219") is True


def test_is_valid_fine_rejects_7_digits_legacy():
    assert is_valid_fine("1905219") is False


def test_is_valid_fine_type2_10_digits():
    assert is_valid_fine("1234567890", is_type2=True) is True


def test_is_valid_fine_type2_11_digits():
    assert is_valid_fine("30850005064", is_type2=True) is True


def test_is_valid_fine_type2_rejects_8_digits():
    assert is_valid_fine("51903219", is_type2=True) is False


def test_is_valid_fine_rejects_none():
    assert is_valid_fine(None) is False


def test_detect_type2_returns_true_for_type2_marker():
    assert detect_type2("מספר הודעת תשלום קנס: 12345678") is True


def test_detect_type2_returns_false_for_legacy():
    assert detect_type2(TEMPLATE1_OCR) is False


def test_detect_type2_returns_true_for_template2():
    assert detect_type2(TEMPLATE2_OCR) is True


# ---------------------------------------------------------------------------
# OCR candidate extraction
# ---------------------------------------------------------------------------


def test_extract_ocr_candidates_finds_plate_and_fine_when_present():
    ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    candidates = extract_ocr_candidates(ocr)
    assert "51903219" in candidates
    assert "6486471" in candidates


def test_extract_ocr_candidates_joins_separated_digits():
    ocr = "05911 5-09\n"
    candidates = extract_ocr_candidates(ocr)
    assert "05911509" in candidates


def test_extract_ocr_candidates_filters_very_short():
    ocr = "123 45678\n"  # 3 + 5 digits separated, joined = 8, but "123" alone is 3 (<6)
    candidates = extract_ocr_candidates(ocr)
    assert "123" not in candidates


def test_extract_ocr_candidates_includes_numeric_text():
    candidates = extract_ocr_candidates("some text", "30850005064")
    assert "30850005064" in candidates


# ---------------------------------------------------------------------------
# Normalized candidate match
# ---------------------------------------------------------------------------


def test_normalized_match_exact():
    assert normalized_candidate_match("51903219", ["51903219"]) is True


def test_normalized_match_with_spaces_in_candidate():
    # Candidate "51 903 219" → digits "51903219" → matches value "51903219"
    assert normalized_candidate_match("51903219", ["51 903 219"]) is True


def test_normalized_match_value_subset_of_candidate():
    # "05911509" is contained in the longer "305911509X"=not useful,
    # but value "6486471" contained in "64864710"
    assert normalized_candidate_match("6486471", ["64864710"]) is True


def test_normalized_match_rejects_unrelated_number():
    assert normalized_candidate_match("99999999", ["51903219", "6486471"]) is False


def test_normalized_match_empty_candidates():
    assert normalized_candidate_match("51903219", []) is False


def test_normalized_match_none_value():
    assert normalized_candidate_match(None, ["51903219"]) is False


# ---------------------------------------------------------------------------
# Template 1 regressions – fine number 51903219
# ---------------------------------------------------------------------------


def test_template1_fine_via_vision_source_bypasses_crosscheck():
    """With vision-sourced fine, OCR cross-check is skipped.

    Template 1 OCR does not contain 51903219 verbatim; vision extraction
    reads the image directly and returns the correct value.  The pipeline
    must accept vision-sourced values without cross-checking against OCR text.
    """

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "6486471",
            "fine": "51903219",
            "confidence": {"plate": 0.9, "fine": 0.92},
            "source": {"plate": "gpt_vision", "fine": "gpt_vision"},
        }

    result = asyncio.run(
        extract_all(
            TEMPLATE1_OCR,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["fine"] == "51903219", (
        f"Expected fine=51903219 (vision-sourced), got {result['fine']!r}"
    )
    assert result["plate"] == "6486471"


def test_template1_fine_from_ocr_with_keyword():
    """When OCR text contains the fine number near a keyword, it is accepted.

    Simulates a cleaner OCR scan where 51903219 is present with context.
    """
    clean_ocr = (
        "מספר דוח: 51903219\n"
        "מספר רכב: 6486471\n"
        "גובה הקנס: 100 ₪\n"
    )

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "6486471",
            "fine": "51903219",
            "confidence": {"plate": 0.9, "fine": 0.92},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    result = asyncio.run(
        extract_all(
            clean_ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["fine"] == "51903219"
    assert result["plate"] == "6486471"


def test_template1_invalid_primary_fine_triggers_retry():
    """When primary GPT returns an invalid fine, the retry is called.

    Template 1 scenario: primary GPT returns 7-digit 1905219 (OCR noise),
    which fails the 8-digit legacy validation and triggers a targeted retry.
    """
    retry_called = []

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "6486471",
            "fine": "1905219",  # 7 digits – invalid for legacy
            "confidence": {"plate": 0.9, "fine": 0.7},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    async def mock_retry_fine(text: str) -> Dict[str, Any]:
        retry_called.append(True)
        return {"fine": "51903219", "confidence": {"fine": 0.88}}

    clean_ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    result = asyncio.run(
        extract_all(
            clean_ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=mock_retry_fine,
        )
    )
    assert retry_called, "Fine retry must be triggered when primary fine is invalid"
    assert result["fine"] == "51903219"


# ---------------------------------------------------------------------------
# Template 2 regressions – plate not confused with ID
# ---------------------------------------------------------------------------


def test_template2_plate_and_fine_correctly_identified():
    """Template 2: plate 05911509, fine 30850005064 (type-2), not ID 7345742623."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "05911509",
            "fine": "30850005064",
            "confidence": {"plate": 0.85, "fine": 0.92},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    result = asyncio.run(
        extract_all(
            TEMPLATE2_OCR,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["plate"] == "05911509", (
        f"Expected plate=05911509, got {result['plate']!r}"
    )
    assert result["fine"] == "30850005064", (
        f"Expected fine=30850005064, got {result['fine']!r}"
    )
    # ID number must not be plate
    assert result["plate"] != "7345742623"
    assert result["plate"] != "2895338"


def test_template2_9digit_id_as_plate_rejected_by_validation():
    """10-digit phone/ID number cannot pass plate validation."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "7345742623",  # 10 digits → invalid plate
            "fine": "30850005064",
            "confidence": {"plate": 0.7, "fine": 0.9},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    async def mock_retry_plate(text: str) -> Dict[str, Any]:
        return {"plate": "05911509", "confidence": {"plate": 0.85}}

    result = asyncio.run(
        extract_all(
            TEMPLATE2_OCR,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=mock_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["plate"] == "05911509", (
        "10-digit ID must be rejected and retry should return valid plate"
    )


# ---------------------------------------------------------------------------
# Validation failure and targeted retry
# ---------------------------------------------------------------------------


def test_invalid_plate_triggers_retry():
    """When primary plate is too short/long, retry is called."""
    retry_called = []

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "12345",  # 5 digits – invalid
            "fine": "51903219",
            "confidence": {"plate": 0.5, "fine": 0.9},
            "source": {},
        }

    async def mock_retry(text: str) -> Dict[str, Any]:
        retry_called.append(True)
        return {"plate": "1234567", "confidence": {"plate": 0.8}}

    ocr = "מספר רכב: 1234567\nמספר דוח: 51903219\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=mock_retry,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert retry_called, "Plate retry must be triggered for invalid plate"
    assert result["plate"] == "1234567"


def test_retry_with_still_invalid_result_gives_none():
    """If retry also returns invalid value, field becomes None."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {"plate": "ABC", "fine": "51903219", "confidence": {}, "source": {}}

    async def bad_retry(text: str) -> Dict[str, Any]:
        return {"plate": "XYZ", "confidence": {"plate": 0.3}}

    ocr = "מספר דוח: 51903219\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=bad_retry,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["plate"] is None


# ---------------------------------------------------------------------------
# OCR cross-check rejects hallucinated values
# ---------------------------------------------------------------------------


def test_ocr_crosscheck_rejects_hallucinated_fine():
    """Fine number not supported by any OCR candidate is rejected."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "6486471",
            "fine": "99999999",  # not in OCR text
            "confidence": {"plate": 0.9, "fine": 0.88},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["fine"] is None, (
        "Hallucinated fine (not in OCR) must be rejected by OCR cross-check"
    )


def test_ocr_crosscheck_rejects_hallucinated_plate():
    """Plate number not supported by any OCR candidate is rejected."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "9999999",  # not in OCR text
            "fine": "51903219",
            "confidence": {"plate": 0.8, "fine": 0.9},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["plate"] is None, (
        "Hallucinated plate (not in OCR) must be rejected by OCR cross-check"
    )


def test_ocr_crosscheck_accepts_value_in_candidates():
    """Value that matches an OCR candidate passes cross-check."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "6486471",
            "fine": "51903219",
            "confidence": {"plate": 0.9, "fine": 0.92},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["fine"] == "51903219"
    assert result["plate"] == "6486471"


def test_vision_source_bypasses_ocr_crosscheck():
    """Vision-sourced values are not subject to OCR cross-check."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        # Return values that are NOT in OCR candidates but are vision-sourced
        return {
            "plate": "6486471",
            "fine": "51903219",
            "confidence": {"plate": 0.95, "fine": 0.97},
            "source": {"plate": "gpt_vision", "fine": "gpt_vision"},
        }

    # OCR text that does NOT contain 51903219 or 6486471
    sparse_ocr = "19052 19\n6 רכב\n100\n"
    result = asyncio.run(
        extract_all(
            sparse_ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["fine"] == "51903219", (
        "Vision-sourced fine must bypass OCR cross-check"
    )
    assert result["plate"] == "6486471", (
        "Vision-sourced plate must bypass OCR cross-check"
    )


# ---------------------------------------------------------------------------
# Same-value prevention
# ---------------------------------------------------------------------------


def test_same_value_not_allowed_in_both_fields():
    """The same digits must not occupy both plate and fine slots."""

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "12345678",
            "fine": "12345678",  # same as plate
            "confidence": {"plate": 0.9, "fine": 0.9},
            "source": {},
        }

    ocr = "מספר רכב: 12345678\nמספר דוח: 12345678\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    # They cannot both be the same value
    assert not (
        result["plate"] is not None
        and result["fine"] is not None
        and result["plate"] == result["fine"]
    ), "Plate and fine must not share the same value"


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------


def test_source_tracked_for_primary_extraction():
    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "6486471",
            "fine": "51903219",
            "confidence": {"plate": 0.9, "fine": 0.9},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["source"].get("plate") == "gpt_primary"
    assert result["source"].get("fine") == "gpt_primary"


def test_source_updated_to_gpt_retry_after_retry():
    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {"plate": None, "fine": "51903219", "confidence": {}, "source": {}}

    async def mock_retry_plate(text: str) -> Dict[str, Any]:
        return {"plate": "6486471", "confidence": {"plate": 0.8}}

    ocr = "מספר דוח: 51903219\nמספר רכב: 6486471\n"
    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=mock_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["plate"] == "6486471"
    assert result["source"].get("plate") == "gpt_retry"


# ---------------------------------------------------------------------------
# Typed OCR candidate extraction
# ---------------------------------------------------------------------------


def test_extract_plate_ocr_candidates_returns_only_7_8_digit():
    ocr = "מספר הודעת: 30850005064\nמספר רכב: 6486471\n9999\n05911509\n"
    candidates = extract_plate_ocr_candidates(ocr)
    assert "6486471" in candidates
    assert "05911509" in candidates
    # 11-digit fine number must NOT appear in plate candidates
    assert "30850005064" not in candidates


def test_extract_fine_ocr_candidates_legacy_8_digits():
    ocr = "מספר דוח: 09224227\nרכב: 6486471\n30850005064\n"
    candidates = extract_fine_ocr_candidates(ocr, is_type2=False)
    assert "09224227" in candidates
    # plate-length numbers must NOT appear in legacy fine candidates
    assert "6486471" not in candidates
    # type-2-length numbers must NOT appear in legacy fine candidates
    assert "30850005064" not in candidates


def test_extract_fine_ocr_candidates_type2_10_11_digits():
    ocr = "מספר: 30850005064\nרכב: 6486471\n09224227\n"
    candidates = extract_fine_ocr_candidates(ocr, is_type2=True)
    assert "30850005064" in candidates
    # 8-digit legacy fine must NOT appear in type-2 fine candidates
    assert "09224227" not in candidates
    # plate must NOT appear in type-2 fine candidates
    assert "6486471" not in candidates


# ---------------------------------------------------------------------------
# Regression 1 – Legacy template: OCR heuristic fine preserved when GPT fails
# ---------------------------------------------------------------------------


def test_regression_legacy_ocr_fine_preserved_when_gpt_returns_invalid():
    """Case 1 regression: GPT returns invalid fine (113723, 6 digits); OCR
    heuristic has valid fine 09224227 (8 digits).  Pipeline must preserve
    the OCR heuristic value instead of returning None.
    """

    async def mock_gpt(text: str) -> Dict[str, Any]:
        # Simulates production: GPT reads "113723" from garbled OCR
        return {
            "plate": "6486471",
            "fine": "113723",  # 6 digits – invalid for legacy
            "confidence": {"plate": 0.85, "fine": 0.6},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    async def mock_retry_fine(text: str) -> Dict[str, Any]:
        # Retry also returns the same invalid value
        return {"fine": "113723", "confidence": {"fine": 0.55}}

    # OCR heuristic has found the correct fine candidate in context
    result = asyncio.run(
        extract_all(
            TEMPLATE1_OCR,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=mock_retry_fine,
            ocr_plate="6486471",
            ocr_fine="09224227",  # OCR heuristic candidate – 8 digits, valid
        )
    )
    assert result["fine"] == "09224227", (
        f"OCR heuristic fine must be preserved when GPT pipeline fails; "
        f"got {result['fine']!r}"
    )
    assert result["plate"] == "6486471"
    assert result["source"].get("fine") == "ocr_heuristic"


# ---------------------------------------------------------------------------
# Regression 2 – Decision-notice: is_type2 detected via anchor markers
# ---------------------------------------------------------------------------


# OCR text that uses decision-notice markers but NOT the explicit type-2
# label (to simulate garbled OCR that loses the fine-label line).
_DECISION_MARKERS_ONLY_OCR = (
    '——— הודעה על החלטה להטיל קנס - "תעודת עובד הציבור"\n'
    "7345742623\n"
    "2895338\n"
    "05911 5-09\n"
    "30850005064\n"
    "תאור העובדות המהוות\n"
)


def test_regression_detect_type2_via_decision_notice_markers():
    """is_type2 must be True when OCR contains decision-notice anchors even
    if the explicit 'מספר הודעת תשלום קנס' label is absent (garbled OCR).
    """
    assert detect_type2(_DECISION_MARKERS_ONLY_OCR) is True


def test_regression_type2_fine_validation_uses_correct_mode():
    """When is_type2=True (from decision markers), an 11-digit fine must pass
    validation; it would be wrongly rejected under is_type2=False (legacy).
    """
    # 11-digit fine is valid only when type-2 rules are applied
    assert is_valid_fine("30850005064", is_type2=True) is True
    assert is_valid_fine("30850005064", is_type2=False) is False


def test_regression_type2_pipeline_validates_fine_with_correct_mode():
    """When pipeline auto-detects type-2 from decision markers, the 11-digit
    fine from GPT must be accepted (not rejected as 'invalid legacy fine').
    """

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "05911509",
            "fine": "30850005064",
            "confidence": {"plate": 0.85, "fine": 0.92},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    result = asyncio.run(
        extract_all(
            _DECISION_MARKERS_ONLY_OCR,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
        )
    )
    assert result["fine"] == "30850005064", (
        f"Type-2 fine must be accepted when type-2 is auto-detected via "
        f"decision markers; got {result['fine']!r}"
    )


# ---------------------------------------------------------------------------
# Regression 3 – Typed OCR cross-check: GPT plate rejected when it only
# matches a longer generic OCR token, not a plate-length candidate
# ---------------------------------------------------------------------------


def test_regression_typed_crosscheck_rejects_plate_from_long_ocr_token():
    """GPT plate 73457426 (8 digits) must be rejected when it only appears as
    a prefix of the 10-digit ID token 7345742623 in the OCR text, and does
    NOT exist as a standalone 7-8 digit sequence.  A generic cross-check
    would accept it via containment; the typed check must reject it.
    """

    # OCR contains 7345742623 (10 digits) but NOT 73457426 as a standalone token
    ocr = (
        "7345742623\n"          # 10-digit ID – not a valid plate
        "30850005064\n"         # 11-digit fine
        "05911 5-09\n"          # 8-digit plate (05911509)
        "תאור העובדות המהוות\n"  # decision-notice marker
    )

    async def mock_gpt(text: str) -> Dict[str, Any]:
        # GPT incorrectly extracts the first 8 digits of the ID as the plate
        return {
            "plate": "73457426",  # first 8 digits of 7345742623 – NOT a plate
            "fine": "30850005064",
            "confidence": {"plate": 0.7, "fine": 0.9},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    result = asyncio.run(
        extract_all(
            ocr,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=_noop_retry_fine,
            ocr_plate="05911509",   # OCR heuristic correctly identified the plate
            ocr_fine="30850005064",
        )
    )
    # GPT plate 73457426 is a substring of 7345742623 but NOT a standalone
    # plate-length OCR token → typed cross-check must reject it and fall
    # back to the OCR heuristic plate.
    assert result["plate"] != "73457426", (
        "GPT plate derived from a longer ID token must be rejected by typed cross-check"
    )
    assert result["plate"] == "05911509", (
        f"OCR heuristic plate must be used after typed cross-check rejects GPT plate; "
        f"got {result['plate']!r}"
    )


# ---------------------------------------------------------------------------
# Regression 4 – Merge preference: stronger OCR candidate wins over
# invalid/weaker GPT candidate (end-to-end with is_type2 parameter)
# ---------------------------------------------------------------------------


def test_regression_merge_prefers_ocr_over_invalid_gpt_with_explicit_is_type2():
    """When is_type2 is passed explicitly (True), GPT primary and retry both
    return an invalid legacy-format fine; OCR heuristic fine (11 digits, valid
    for type-2) must be preserved in the final result.
    """

    async def mock_gpt(text: str) -> Dict[str, Any]:
        return {
            "plate": "05911509",
            "fine": "09224227",  # 8-digit legacy fine – INVALID for type-2
            "confidence": {"plate": 0.85, "fine": 0.7},
            "source": {"plate": "gpt_primary", "fine": "gpt_primary"},
        }

    async def mock_retry_fine(text: str) -> Dict[str, Any]:
        return {"fine": "09224227", "confidence": {"fine": 0.6}}

    result = asyncio.run(
        extract_all(
            TEMPLATE2_OCR,
            gpt_extract_fn=mock_gpt,
            gpt_retry_plate_fn=_noop_retry_plate,
            gpt_retry_fine_fn=mock_retry_fine,
            is_type2=True,          # forced by OCR routing stage
            ocr_plate="05911509",
            ocr_fine="30850005064",  # OCR heuristic: 11 digits, valid for type-2
        )
    )
    assert result["fine"] == "30850005064", (
        f"OCR heuristic type-2 fine must beat an invalid GPT legacy fine; "
        f"got {result['fine']!r}"
    )
    assert result["source"].get("fine") == "ocr_heuristic"
    assert result["plate"] == "05911509"

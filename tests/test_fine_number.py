import asyncio

from bot.fine_number import (
    find_fine_number_candidates,
    is_valid_fine_number,
    normalize_fine_number,
    pick_best_fine_number,
)
from bot.handlers.upload import _apply_heuristic_candidates
from bot.handlers.upload import _apply_vision_candidates
from bot.handlers.upload import _choose_final_plate_candidate
from bot.handlers.upload import _ensure_fine_number


def test_normalize_fine_number_basic():
    assert normalize_fine_number(" 51-9032 19 ") == "51903219"


def test_normalize_fine_number_aggressive_ocr():
    assert normalize_fine_number("5I9O3Z19", aggressive=True) == "51903219"


def test_is_valid_fine_number():
    assert is_valid_fine_number("51 9032 19")
    assert not is_valid_fine_number("AB1234")
    assert not is_valid_fine_number("123")


def test_find_fine_number_candidates_prefers_keyword_scope():
    text = "מספר דוח: 5190-3219\nסכום: 250"
    candidates = find_fine_number_candidates(text)
    assert "51903219" in candidates


def test_find_fine_number_candidates_labeled_only_ignores_unlabeled_numbers():
    text = (
        "19052 19\n"
        "מספר רכב 6486471\n"
        "גובה הקנס 100\n"
    )
    candidates = find_fine_number_candidates(text, labeled_only=True)
    assert candidates == []


def test_find_fine_number_candidates_type2_avoids_tz():
    text = (
        "מספר הודעת תשלום קנס: 12345-67890\n"
        "תעודת זהות: 123456789\n"
    )
    candidates = find_fine_number_candidates(text)
    assert "1234567890" in candidates
    assert "123456789" not in candidates


def test_find_fine_number_candidates_type2_merged_label_punctuation():
    text = (
        "מספר-הודעת/תשלום,קנס 10.234.567.890\n"
        "תעודת-זהות 234567890\n"
    )
    candidates = find_fine_number_candidates(text)
    assert "10234567890" in candidates
    assert "234567890" not in candidates


def test_pick_best_fine_number():
    assert pick_best_fine_number(["12345678", "12345678", "999999"]) == "12345678"


def test_ensure_fine_number_replaces_invalid_with_focused_result():
    async def fake_extractor(_ocr_text: str, _numeric_ocr_text: str):
        return {"fine_number": "51-9032-19", "confidence": 0.91}

    details = {"fine_number": {"value": "ABCD", "confidence": 0.95}}
    updated = asyncio.run(
        _ensure_fine_number(
            details,
            "ticket number: ???",
            "",
            focused_extractor=fake_extractor,
        )
    )
    assert updated["fine_number"]["value"] == "51903219"
    assert updated["fine_number"]["confidence"] >= 0.9


def test_ensure_fine_number_uses_heuristic_candidate():
    async def fake_extractor(_ocr_text: str, _numeric_ocr_text: str):
        return {"fine_number": None, "confidence": 0.0}

    details = {"fine_number": {"value": None, "confidence": 0.1}}
    updated = asyncio.run(
        _ensure_fine_number(
            details,
            "מספר דוח 8123-4567",
            "8123 4567",
            focused_extractor=fake_extractor,
        )
    )
    assert updated["fine_number"]["value"] == "81234567"
    assert updated["fine_number"]["confidence"] >= 0.55


def test_apply_heuristic_candidates_sets_weak_fields():
    details = {
        "vehicle_plate": {"value": "", "confidence": 0.2},
        "fine_number": {"value": "", "confidence": 0.2},
        "fine_amount": {"value": "", "confidence": 0.2},
    }
    candidates = {
        "plate": "12345678",
        "fine": "51903219",
        "amount": "250",
        "plate_confident": True,
        "fine_confident": True,
        "amount_confident": True,
    }
    updated = _apply_heuristic_candidates(details, candidates)
    assert updated["vehicle_plate"]["value"] == "12345678"
    assert updated["fine_number"]["value"] == "51903219"
    assert updated["fine_amount"]["value"] == "250"


def test_apply_vision_candidates_overrides_only_valid_vision_fields():
    details = {
        "vehicle_plate": {"value": "7654321", "confidence": 0.6},
        "fine_number": {"value": "51903219", "confidence": 0.7},
    }

    updated = _apply_vision_candidates(details, {"license_plate": "12345678"})

    assert updated["vehicle_plate"]["value"] == "12345678"
    assert updated["vehicle_plate"]["confidence"] == 0.98
    assert updated["fine_number"]["value"] == "51903219"


def test_choose_final_plate_candidate_prefers_ocr_8_over_vision_7():
    selected, source, reason = _choose_final_plate_candidate("1234567", "12345678")
    assert selected == "12345678"
    assert source == "ocr"
    assert reason == "ocr_format_preferred"


def test_choose_final_plate_candidate_prefers_vision_8_over_ocr_7():
    selected, source, reason = _choose_final_plate_candidate("12345678", "1234567")
    assert selected == "12345678"
    assert source == "vision"
    assert reason == "vision_format_preferred"


def test_choose_final_plate_candidate_same_class_keeps_deterministic_vision_preference():
    selected, source, reason = _choose_final_plate_candidate("11112222", "33334444")
    assert selected == "11112222"
    assert source == "vision"
    assert reason == "same_class_prefer_vision"


def test_choose_final_plate_candidate_picks_valid_over_invalid():
    selected, source, reason = _choose_final_plate_candidate("12345", "1234567")
    assert selected == "1234567"
    assert source == "ocr"
    assert reason == "ocr_format_preferred"


# ---------------------------------------------------------------------------
# Regression tests for the two real production fine notice types
# ---------------------------------------------------------------------------

def test_ensure_fine_number_calls_focused_for_moderate_confidence():
    """Focused extractor is called when AI confidence is moderate (< 0.75).

    Regression for legacy notice type: AI returns a plausible but wrong fine
    number with confidence below the high-confidence threshold.  The focused
    extractor (with a targeted prompt) should be given a chance to correct it.
    """
    called = []

    async def fake_focused(_ocr, _num):
        called.append(True)
        return {"fine_number": "51903219", "confidence": 0.92}

    # AI returned a "valid" 7-digit fine number but with only moderate confidence
    details = {"fine_number": {"value": "4030573", "confidence": 0.65}}
    updated = asyncio.run(
        _ensure_fine_number(
            details,
            "הודעת תשלום קנס\nסכום: 2076",
            "",
            focused_extractor=fake_focused,
        )
    )
    assert called, "focused extractor must be called when confidence < 0.75"
    assert updated["fine_number"]["value"] == "51903219"
    assert updated["fine_number"]["confidence"] >= 0.9


def test_ensure_fine_number_skips_focused_for_high_confidence():
    """Focused extractor is NOT called when AI confidence is already >= 0.75."""
    called = []

    async def fake_focused(_ocr, _num):
        called.append(True)
        return {"fine_number": "00000000", "confidence": 0.99}

    details = {"fine_number": {"value": "51903219", "confidence": 0.85}}
    updated = asyncio.run(
        _ensure_fine_number(
            details,
            "מספר דוח: 51903219",
            "",
            focused_extractor=fake_focused,
        )
    )
    assert not called, "focused extractor must NOT be called when confidence >= 0.75"
    assert updated["fine_number"]["value"] == "51903219"


def test_ensure_fine_number_drops_unsupported_high_confidence_legacy_value():
    """Unsupported municipal top-of-page numbers should not survive by confidence alone."""
    called = []

    async def fake_focused(_ocr, _num):
        called.append(True)
        return {"fine_number": None, "confidence": 0.0}

    ocr_text = (
        "19052 19\n"
        "מטפר רבב\n"
        "6 רכב\n"
        "6486471\n"
        "גובה הקנם בט\"ח: 100\n"
    )
    updated = asyncio.run(
        _ensure_fine_number(
            {"fine_number": {"value": "1905219", "confidence": 0.85}},
            ocr_text,
            "1905219 6486471 100",
            focused_extractor=fake_focused,
        )
    )
    assert called, "focused extractor should re-check unsupported municipal values"
    assert updated["fine_number"]["value"] is None


def test_reconcile_uses_plate_with_strong_anchor_ctx_even_if_not_confident():
    """Plate with strong anchor context is used even without plate_confident flag.

    Regression for second-type notice: plate 2266111 appears only once (not
    confident by count) but sits directly next to a plate keyword.  The
    reconciliation should use it via plate_ctx >= 4.
    """
    from bot.handlers.upload import _reconcile_vehicle_plate

    details = {}
    heuristic_candidates = {
        "plate": "2266111",
        "fine": None,
        "amount": "2076",
        "plate_confident": False,   # count < 2, so not confident
        "fine_confident": False,
        "amount_confident": True,
        "plate_ctx": 8,             # same-line label match → strong anchor
    }
    vision_fields: dict = {}
    result = _reconcile_vehicle_plate(
        details,
        heuristic_candidates,
        vision_fields,
        user_id=1,
        case_id="test-case",
    )
    assert result.get("vehicle_plate", {}).get("value") == "2266111", (
        "Plate with plate_ctx>=4 should be selected even when plate_confident=False"
    )


def test_reconcile_ignores_plate_with_no_anchor_ctx_and_not_confident():
    """Plate with zero anchor context and not confident stays ignored."""
    from bot.handlers.upload import _reconcile_vehicle_plate

    details = {}
    heuristic_candidates = {
        "plate": "2266111",
        "fine": None,
        "amount": None,
        "plate_confident": False,
        "fine_confident": False,
        "amount_confident": False,
        "plate_ctx": 0,  # no anchor context
    }
    vision_fields: dict = {}
    result = _reconcile_vehicle_plate(
        details,
        heuristic_candidates,
        vision_fields,
        user_id=1,
        case_id="test-case",
    )
    # Without context or confidence, plate should not be applied
    assert "vehicle_plate" not in result or result.get("vehicle_plate", {}).get("value") is None

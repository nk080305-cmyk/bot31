import asyncio

from bot.fine_number import (
    find_fine_number_candidates,
    is_valid_fine_number,
    normalize_fine_number,
    pick_best_fine_number,
)
from bot.handlers.upload import _apply_heuristic_candidates
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
    }
    candidates = {
        "plate": "12345678",
        "fine": "51903219",
        "plate_confident": True,
        "fine_confident": True,
    }
    updated = _apply_heuristic_candidates(details, candidates)
    assert updated["vehicle_plate"]["value"] == "12345678"
    assert updated["fine_number"]["value"] == "51903219"

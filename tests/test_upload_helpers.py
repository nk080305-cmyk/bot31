import asyncio

from bot.handlers.upload import _ensure_fine_number


def test_ensure_fine_number_skips_focused_llm_when_heuristic_found():
    calls = {"count": 0}

    async def focused_extractor(_ocr_text: str, _numeric_text: str):
        calls["count"] += 1
        return {"fine_number": "11111111", "confidence": 0.9}

    details = {"fine_number": {"value": "", "confidence": 0.2}}
    result = asyncio.run(
        _ensure_fine_number(
            details,
            "מספר דוח: 51903219",
            "51903219 2266111",
            focused_extractor=focused_extractor,
        )
    )
    assert result["fine_number"]["value"] == "51903219"
    assert calls["count"] == 0


def test_ensure_fine_number_calls_focused_llm_when_no_valid_candidates():
    calls = {"count": 0}

    async def focused_extractor(_ocr_text: str, _numeric_text: str):
        calls["count"] += 1
        return {"fine_number": "51903219", "confidence": 0.85}

    details = {"fine_number": {"value": "", "confidence": 0.0}}
    result = asyncio.run(
        _ensure_fine_number(
            details,
            "ללא מספר דוח",
            "12 34",
            focused_extractor=focused_extractor,
        )
    )
    assert result["fine_number"]["value"] == "51903219"
    assert calls["count"] == 1

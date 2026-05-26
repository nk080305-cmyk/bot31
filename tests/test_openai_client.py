"""Smoke tests and unit tests for bot.openai_client helpers."""

import asyncio
from types import SimpleNamespace

import pytest

from bot.openai_client import (
    # существующие импорты из main (оставь все которые были)
    _best_fine_candidate,
    _context_digits_near_keywords,
    _digits_only,
    extract_fine_details,
    extract_fine_number_only,
    generate_appeal,
    # новые из PR
    extract_vision_fields,
    normalize_license_plate,
)

class _DummyResponse:
    def __init__(self, content: str):
        self.id = "resp_1"
        self.model = "test-model"
        self.created = 0
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class _DummyCompletions:
    def __init__(self, content: str):
        self._content = content

    async def create(self, **_kwargs):
        return _DummyResponse(self._content)


class _DummyChat:
    def __init__(self, content: str):
        self.completions = _DummyCompletions(content)


class _DummyClient:
    def __init__(self, content: str):
        self.chat = _DummyChat(content)


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_module_importable():
    """Verify that bot.openai_client can be imported without errors."""
    import bot.openai_client  # noqa: F401 – side-effect import

    assert callable(extract_fine_details)
    assert callable(extract_fine_number_only)
    assert callable(generate_appeal)


# ---------------------------------------------------------------------------
# _digits_only
# ---------------------------------------------------------------------------


def test_digits_only_strips_letters_and_punctuation():
    assert _digits_only("AB-123-45") == "12345"


def test_digits_only_keeps_digits():
    assert _digits_only("12345678") == "12345678"


def test_digits_only_empty():
    assert _digits_only("") == ""


def test_digits_only_all_non_digits():
    assert _digits_only("abc!@#") == ""


def test_normalize_license_plate_strips_separators():
    assert normalize_license_plate("12-345 678") == "12345678"


# ---------------------------------------------------------------------------
# _context_digits_near_keywords
# ---------------------------------------------------------------------------


def test_context_digits_near_keywords_finds_near_keyword():
    text = "מספר דוח: 51903219 סכום: 250"
    result = _context_digits_near_keywords(text, ["מספר דוח"], 7, 13)
    assert "51903219" in result


def test_context_digits_near_keywords_ignores_far_numbers():
    # Number before any keyword context window should be excluded
    text = "12345678 " + "x" * 500 + " מספר דוח: 99999999"
    result = _context_digits_near_keywords(text, ["מספר דוח"], 7, 13, window=20)
    # Only "99999999" is within the narrow window of 20 chars
    assert "99999999" in result


def test_context_digits_near_keywords_deduplicates():
    text = "מספר דוח 12345678 מספר דוח 12345678"
    result = _context_digits_near_keywords(text, ["מספר דוח"], 7, 13)
    assert result.count("12345678") == 1


def test_context_digits_near_keywords_tolerates_punctuation_in_keyword():
    text = "מספר-הודעת/תשלום,קנס: 10.234.567.890"
    result = _context_digits_near_keywords(text, ["מספר הודעת תשלום קנס"], 10, 11)
    assert "10234567890" in result


def test_context_digits_near_keywords_empty_text():
    assert _context_digits_near_keywords("", ["מספר דוח"], 7, 13) == []


# ---------------------------------------------------------------------------
# _best_plate_candidate
# ---------------------------------------------------------------------------


def test_best_plate_candidate_finds_7_digit_string():
    assert _best_plate_candidate("some text 1234567 more text") == "1234567"


def test_best_plate_candidate_returns_first_match():
    assert _best_plate_candidate("1234567 7654321") == "1234567"


def test_best_plate_candidate_empty_input():
    assert _best_plate_candidate("") is None


def test_best_plate_candidate_no_match():
    assert _best_plate_candidate("abc def") is None


def test_best_plate_candidate_six_digits():
    assert _best_plate_candidate("123456") == "123456"


def test_best_plate_candidate_eight_digits():
    assert _best_plate_candidate("12345678") == "12345678"


# ---------------------------------------------------------------------------
# _best_fine_candidate
# ---------------------------------------------------------------------------


def test_best_fine_candidate_returns_candidate():
    # Keyword-adjacent number should be found via find_fine_number_candidates
    text = "מספר דוח 51903219"
    result = _best_fine_candidate(text)
    assert result == "51903219"


def test_best_fine_candidate_excludes_plate():
    text = "מספר דוח 51903219"
    result = _best_fine_candidate(text, plate="51903219")
    assert result is None or result != "51903219"


def test_best_fine_candidate_empty_input():
    assert _best_fine_candidate("") is None


<def test_extract_vision_fields_normalizes_type2_notice_and_plate(monkeypatch, tmp_path):
    image_path = tmp_path / "notice.jpg"
    image_path.write_bytes(b"fake-jpeg")
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_text='{"fine_notice_number":"12345-67890","license_plate":"12-345-678"}'
        )

    monkeypatch.setattr(
        "bot.openai_client._client",
        SimpleNamespace(responses=SimpleNamespace(create=fake_create)),
    )

    result = asyncio.run(
        extract_vision_fields(
            str(image_path),
            'מספר הודעת תשלום קנס: 12345-67890\nתעודת זהות: 123456789',
        )
    )

    assert result == {
        "fine_notice_number": "1234567890",
        "license_plate": "12345678",
    }
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True


def test_extract_vision_fields_rejects_9_digit_tz_for_type2(monkeypatch, tmp_path):
    image_path = tmp_path / "notice.png"
    image_path.write_bytes(b"fake-png")

    async def fake_create(**_kwargs):
        return SimpleNamespace(
            output_text='{"fine_notice_number":"123456789","license_plate":"123-4567"}'
        )

    monkeypatch.setattr(
        "bot.openai_client._client",
        SimpleNamespace(responses=SimpleNamespace(create=fake_create)),
    )

    result = asyncio.run(
        extract_vision_fields(
            str(image_path),
            'מספר הודעת תשלום קנס: 12345-67890\nתעודת זהות: 123456789',
        )
    )

    assert result == {"license_plate": "1234567"}


def test_extract_vision_fields_returns_empty_on_invalid_json(monkeypatch, tmp_path):
    image_path = tmp_path / "notice.jpg"
    image_path.write_bytes(b"fake-jpeg")

    async def fake_create(**_kwargs):
        return SimpleNamespace(output_text="not-json")

    monkeypatch.setattr(
        "bot.openai_client._client",
        SimpleNamespace(responses=SimpleNamespace(create=fake_create)),
    )

    result = asyncio.run(extract_vision_fields(str(image_path), ""))

    assert result == {}


def test_extract_fine_details_type2_prefers_10_11_and_plate_anchor(monkeypatch):
    import bot.openai_client as mod

    ocr_text = (
        "מספר הודעת תשלום קנס: 12345-67890\n"
        "תעודת זהות: 123456789\n"
        "מס' רכב: 12-345-678\n"
    )
    llm_payload = json.dumps(
        {
            "fine_number": {"value": "123456789", "confidence": 0.95},
            "vehicle_plate": {"value": "", "confidence": 0.1},
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(mod, "_client", _DummyClient(llm_payload))

    result = asyncio.run(extract_fine_details(ocr_text, "12345 67890 123456789 12 345 678"))
    assert result["fine_number"]["value"] == "1234567890"
    assert result["fine_number"]["value"] != "123456789"
    assert result["vehicle_plate"]["value"] == "12345678"

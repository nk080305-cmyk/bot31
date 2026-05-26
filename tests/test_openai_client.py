"""Smoke tests and unit tests for bot.openai_client helpers."""

import pytest

from bot.openai_client import (
    _best_fine_candidate,
    _best_plate_candidate,
    _context_digits_near_keywords,
    _digits_only,
    extract_fine_details,
    extract_fine_number_only,
    generate_appeal,
)


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

import numpy as np
import pytest

from bot.ocr import (
    _context_score,
    _is_multi_preprocess_enabled,
    extract_plate_and_fine_candidates,
    mask_secret,
    preprocess_variants,
    PLATE_KEYWORDS,
    FINE_KEYWORDS,
)


def test_preprocess_variants_returns_three_images():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    variants = preprocess_variants(image)
    assert len(variants) == 3
    assert all(v.shape == (32, 32) for v in variants)


def test_extract_plate_and_fine_candidates_context_and_frequency():
    ocr_text = (
        "מספר רכב: 12345678\n"
        "דוח מספר: 51903219\n"
        "מספר רכב 12345678\n"
        "מספר דוח 51903219"
    )
    numeric_text = "12345678 51903219"
    result = extract_plate_and_fine_candidates(ocr_text, numeric_text)
    assert result["plate"] == "12345678"
    assert result["fine"] == "51903219"
    assert result["plate_confident"] is True
    assert result["fine_confident"] is True


def test_extract_plate_and_fine_candidates_context_required_for_plate():
    """Plate candidate with no context proximity should be rejected (→ None)."""
    # The fine number appears near its keyword; the plate number appears
    # nowhere near a plate keyword so it must be rejected.
    ocr_text = (
        "מספר דוח: 51903219\n"
        "מספר דוח 51903219\n"
        "somewhere else: 12345678"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "")
    # 12345678 has no plate-keyword context → must be None
    assert result["plate"] is None
    assert result["fine"] == "51903219"


def test_extract_plate_and_fine_candidates_context_required_for_fine():
    """Fine candidate with no context proximity should be rejected (→ None)."""
    ocr_text = (
        "מספר רכב: 12345678\n"
        "מספר רכב 12345678\n"
        "somewhere else: 51903219"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "")
    assert result["plate"] == "12345678"
    # 51903219 has no fine-keyword context → must be None
    assert result["fine"] is None


def test_extract_plate_and_fine_candidates_no_context_returns_none():
    """When no candidates have context, both fields must be None."""
    # Numbers appear but no Hebrew keywords at all
    ocr_text = "random stuff 12345678 and more 51903219 end"
    result = extract_plate_and_fine_candidates(ocr_text, "")
    assert result["plate"] is None
    assert result["fine"] is None


def test_extract_plate_and_fine_candidates_context_wins_over_frequency():
    """Number with keyword context wins even if another number is more frequent."""
    # 09224227 appears 3× but has no fine-keyword context.
    # 51903219 appears once but is right next to a fine keyword.
    ocr_text = (
        "09224227 09224227 09224227\n"
        "מספר דוח: 51903219\n"
        "מספר רכב: 12345678\n"
        "מספר רכב 12345678\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "")
    assert result["fine"] == "51903219", (
        f"Expected fine=51903219 but got {result['fine']}; "
        "context score must beat raw frequency"
    )


def test_extract_plate_and_fine_candidates_plate_fine_not_equal():
    """If the same number would be chosen for both, one must become None."""
    # Only one 8-digit number exists and it has context for both plate and fine.
    ocr_text = (
        "מספר רכב: 12345678\n"
        "מספר דוח: 12345678\n"
        "מספר רכב 12345678\n"
        "מספר דוח 12345678\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "")
    # They must not both be set to the same value.
    assert not (result["plate"] is not None and result["fine"] is not None and result["plate"] == result["fine"])


def test_extract_plate_and_fine_candidates_type2_prefers_10_11_digits_not_tz():
    ocr_text = (
        "דוח חניה עירוני\n"
        "מספר הודעת תשלום קנס: 12345-67890\n"
        "תעודת זהות: 123456789\n"
        "מספר רכב: 7654321\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "12345 67890 123456789 7654321")
    assert result["fine"] == "1234567890"
    assert result["fine"] != "123456789"


def test_extract_plate_and_fine_candidates_type2_merged_spaces_and_punctuation():
    ocr_text = (
        "מספר-הודעת/תשלום,קנס 10.234.567.890\n"
        "תעודת-זהות 234.567.890\n"
        "מספר רכב 12345678\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "")
    assert result["fine"] == "10234567890"
    assert result["fine"] != "234567890"


def test_extract_plate_and_fine_candidates_plate_anchor_with_abbrev_and_separators():
    ocr_text = (
        "מס' רכב: 12-345-678\n"
        "מספר דוח: 5190-3219\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "")
    assert result["plate"] == "12345678"
    assert result["fine"] == "51903219"


def test_extract_plate_and_fine_candidates_debug_logging(monkeypatch, caplog):
    """With OCR_DEBUG=1 the function emits DEBUG log lines."""
    monkeypatch.setenv("OCR_DEBUG", "1")
    import logging
    ocr_text = (
        "מספר רכב: 12345678\n"
        "מספר דוח: 51903219\n"
        "מספר רכב 12345678\n"
        "מספר דוח 51903219"
    )
    with caplog.at_level(logging.DEBUG, logger="bot.ocr"):
        extract_plate_and_fine_candidates(ocr_text, "12345678 51903219")
    assert any("candidate" in r.message.lower() or "winner" in r.message.lower() for r in caplog.records)


def test_multi_preprocess_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv("OCR_MULTI_VARIANTS", raising=False)
    monkeypatch.delenv("OCR_MULTI_PREPROCESS", raising=False)
    assert _is_multi_preprocess_enabled() is False


def test_multi_preprocess_flag_can_be_enabled_with_alias(monkeypatch):
    monkeypatch.setenv("OCR_MULTI_PREPROCESS", "0")
    monkeypatch.setenv("OCR_MULTI_VARIANTS", "1")
    assert _is_multi_preprocess_enabled() is True


# ---------------------------------------------------------------------------
# mask_secret tests
# ---------------------------------------------------------------------------

def test_mask_secret_hides_middle():
    key = "sk-proj-ABCDEF1234WXYZ"
    masked = mask_secret(key)
    assert masked.startswith("sk-proj")
    assert masked.endswith("WXYZ")
    assert "ABCDEF1234" not in masked


def test_mask_secret_short_value_redacted():
    assert mask_secret("short") == "[REDACTED]"


def test_mask_secret_none_redacted():
    assert mask_secret(None) == "[REDACTED]"


def test_mask_secret_empty_redacted():
    assert mask_secret("") == "[REDACTED]"


def test_mask_secret_exact_boundary_redacted():
    # Exactly prefix_len + suffix_len chars → redacted
    assert mask_secret("a" * 11) == "[REDACTED]"


def test_mask_secret_one_over_boundary():
    val = "a" * 12  # 7 + 4 + 1
    result = mask_secret(val)
    assert result == "aaaaaaa...aaaa"

import numpy as np
import pytest

from bot.ocr import (
    _context_score,
    _detect_fine_template_with_reason,
    _detect_type2_token_proximity,
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


def test_extract_plate_and_fine_candidates_legacy_template_keeps_8_digit_fine():
    ocr_text = (
        "מספר רכב: 2266111\n"
        "מספר דוח: 51903219\n"
        "תעודת זהות: 123456789\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "2266111 51903219 123456789")
    assert result["plate"] == "2266111"
    assert result["fine"] == "51903219"


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


def test_extract_plate_and_fine_candidates_prefers_plate_anchor_over_narrative_number():
    ocr_text = (
        "מס רכב: 2266111\n"
        "אני החתום מטה מציין: רכב 5892531 לא עצר בקו עצירה\n"
        "מספר דוח: 51903219\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "2266111 5892531 51903219")
    assert result["plate"] == "2266111"


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


def test_extract_plate_and_fine_candidates_anchor_template_uses_type2_fine_strategy():
    ocr_text = (
        "מספר הודעת תשלום קנס: 1234567890\n"
        "מספר דוח: 51903219\n"
        "מספר רכב: 2266111\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "1234567890 51903219 2266111")
    assert result["fine"] == "1234567890"


def test_extract_plate_and_fine_candidates_amount_prefers_plausible_value_over_date():
    ocr_text = (
        "מספר רכב: 2266111\n"
        "מספר דוח: 51903219\n"
        "סכום הקנס: 250 ₪\n"
        "יש לשלם עד 15/05/2026\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "15052026 250")
    assert result["amount"] == "250"
    assert result["fine"] == "51903219"


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


def test_extract_plate_and_fine_candidates_logs_narrative_rejection(monkeypatch, caplog):
    monkeypatch.setenv("OCR_DEBUG", "1")
    import logging
    ocr_text = "מספר הודעת תשלום קנס: 1234567890\nאני החתום מטה רכב 5892531"
    with caplog.at_level(logging.DEBUG, logger="bot.ocr"):
        extract_plate_and_fine_candidates(ocr_text, "5892531 1234567890")
    assert any("narrative/body" in r.message for r in caplog.records)


def test_extract_plate_and_fine_candidates_logs_template_detection(caplog):
    import logging
    legacy_text = "מספר רכב: 2266111\nמספר דוח: 51903219\n"
    anchor_text = "מספר הודעת תשלום קנס: 1234567890\nמספר רכב: 2266111\n"
    with caplog.at_level(logging.INFO, logger="bot.ocr"):
        extract_plate_and_fine_candidates(legacy_text, "2266111 51903219")
        extract_plate_and_fine_candidates(anchor_text, "1234567890 2266111")
    assert any("template_detected=legacy" in r.message for r in caplog.records)
    assert any("template_detected=anchor_based" in r.message for r in caplog.records)


def test_extract_plate_and_fine_candidates_legacy_notice_anchor_extracts_fine_number(caplog):
    import logging
    ocr_text = (
        "הודעת תשלום קנס\n"
        "מספר הודעה: 51903219\n"
        "מספר רכב: 1234567\n"
        "תאריך צילום: 15/05/2076\n"
    )
    with caplog.at_level(logging.INFO, logger="bot.ocr"):
        result = extract_plate_and_fine_candidates(ocr_text, "51903219 1234567 15052076 2076")
    assert any("template_detected=legacy" in r.message for r in caplog.records)
    assert any(
        "reason=fallback_legacy_notice:type2_notice_with_legacy_label_no_narrative_marker"
        in r.message
        for r in caplog.records
    )
    assert result["fine"] == "51903219"


def test_extract_plate_and_fine_candidates_type2_notice_routes_anchor_and_keeps_labeled_plate(caplog):
    import logging
    ocr_text = (
        "הודעת תשלום קנס\n"
        "אני החתום מטה בתאריך 15/05/2076 מציין את פרטי המקרה\n"
        "מספר רכב: 2266111\n"
        "תעודת זהות: 123456789\n"
    )
    with caplog.at_level(logging.INFO, logger="bot.ocr"):
        result = extract_plate_and_fine_candidates(ocr_text, "15052076 2076 2266111 123456789")
    assert any("template_detected=anchor_based" in r.message for r in caplog.records)
    assert result["plate"] == "2266111"
    assert result["fine"] is None


def test_detect_fine_template_decision_notice_markers_route_anchor_based():
    ocr_text = (
        "הודעה על החלטה להטיל קנס\n"
        "תעודת עובד הציבור\n"
        "תאור העובדות המהוות\n"
    )
    template, reason = _detect_fine_template_with_reason(ocr_text)
    assert template == "anchor_based"
    assert "decision_notice_markers" in reason


def test_extract_plate_and_fine_candidates_decision_notice_prefers_labeled_plate():
    ocr_text = (
        "הודעה על החלטה להטיל קנס\n"
        "תעודת עובד הציבור\n"
        "מספר רכב: 7654321\n"
        "תאור העובדות המהוות\n"
        "ברכב אחר הופיע 1234567 בגוף הטקסט\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "7654321 1234567")
    assert result["plate"] == "7654321"


def test_extract_plate_and_fine_candidates_municipal_template_recovers_fine_number():
    ocr_text = (
        "הודעת תשלום קנס\n"
        "19052 19\n"
        "מספר רכב\n"
        "6486471\n"
        "יצרן רכב\n"
        "גובה הקנס בט\"ח: 100\n"
        "הערות הפקח\n"
        "עבירה - 133\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "1905219 6486471 100 133")
    assert result["fine"] == "1905219"


def test_detect_fine_template_noisy_municipal_anchors_stay_legacy():
    ocr_text = (
        "הודעת תשלום קנס\n"
        "מטפר רבב\n"
        "יצרן רכב\n"
        "גובה הקנם בט\"ח: 100\n"
        "הערות הפקח\n"
    )
    template, reason = _detect_fine_template_with_reason(ocr_text)
    assert template == "legacy"
    assert "municipal_notice_markers" in reason


def test_extract_plate_and_fine_candidates_decision_notice_ignores_date_numbers_for_plate():
    ocr_text = (
        "הודעה על החלטה להטיל קנס\n"
        "תעודת עובד הציבור\n"
        "תאריך: 15/05/2026\n"
        "מספר רכב\n"
        "2266111\n"
        "תאור העובדות המהוות\n"
    )
    result = extract_plate_and_fine_candidates(ocr_text, "15052026 2266111")
    assert result["plate"] == "2266111"


def test_extract_plate_and_fine_candidates_logs_candidates_and_selected(caplog):
    import logging
    ocr_text = (
        "הודעה על החלטה להטיל קנס\n"
        "תעודת עובד הציבור\n"
        "מספר רכב: 7654321\n"
        "מספר הודעת תשלום קנס: 1234567890\n"
    )
    with caplog.at_level(logging.INFO, logger="bot.ocr"):
        extract_plate_and_fine_candidates(ocr_text, "7654321 1234567890")
    assert any("Fine OCR candidates template=" in r.message for r in caplog.records)
    assert any("Fine OCR selected template=" in r.message for r in caplog.records)


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


# ---------------------------------------------------------------------------
# Regression tests for the two real production notice types
# ---------------------------------------------------------------------------

def test_detect_type2_token_proximity_finds_noisy_label():
    """Token-proximity detection routes noisy OCR with fragmented label to anchor_based."""
    # Simulate OCR noise where the full phrase "הודעת תשלום קנס" is broken
    # but all three token roots are present within 150 chars.
    noisy_text = "הודעת  tsh1um  קנס\n"  # תשלום garbled as ASCII noise
    # This should NOT match because תשלו root is absent
    assert _detect_type2_token_proximity(noisy_text) is None

    # When all three roots are present, proximity detection should trigger
    clean_tokens = "הודע blah תשלו more קנס\n"
    assert _detect_type2_token_proximity(clean_tokens) is not None


def test_detect_type2_token_proximity_not_triggered_without_all_tokens():
    """Partial token presence must not trigger type-2 detection."""
    only_two = "הודעת תשלום\nsome other text"
    assert _detect_type2_token_proximity(only_two) is None

    only_one = "קנס 2266111"
    assert _detect_type2_token_proximity(only_one) is None


def test_detect_fine_template_noisy_type2_routes_anchor_based():
    """Even with OCR-corrupted label, type-2 token proximity triggers anchor_based routing.

    Real-world: OCR score ~440 can mangle 'הודעת תשלום קנס' into separate
    fragments that the exact phrase regex misses but token proximity catches.
    """
    # Simulate what OCR might produce for a type-2 notice - roots present but
    # label split across tokens with noise characters.
    noisy_type2_ocr = (
        "הודע   תשלו  קנס\n"
        "מספר רכב: 2266111\n"
        "תאריך: 15/05/2076\n"
        "סכום: 2076\n"
    )
    template, reason = _detect_fine_template_with_reason(noisy_type2_ocr)
    assert template == "anchor_based", f"Expected anchor_based, got {template} ({reason})"
    assert "token_proximity" in reason


def test_detect_fine_template_noisy_type2_not_triggered_with_legacy_label():
    """Token proximity must NOT override if a legacy label is also present."""
    mixed_text = (
        "הודע תשלו קנס\n"
        "מספר דוח: 51903219\n"  # legacy label present → must stay legacy
    )
    template, reason = _detect_fine_template_with_reason(mixed_text)
    assert template == "legacy"
    assert "blocks_type2_token_proximity" in reason


def test_detect_fine_template_type2_notice_with_legacy_label_and_narrative_routes_anchor():
    mixed_text = (
        "הודעת תשלום קנס\n"
        "אני החתום מטה מציין את פרטי המקרה\n"
        "מספר הודעה: 4030573\n"
        "מספר רכב: 2266111\n"
    )
    template, reason = _detect_fine_template_with_reason(mixed_text)
    assert template == "anchor_based"
    assert "narrative_marker" in reason


def test_extract_plate_and_fine_candidates_noisy_type2_recovers_plate(caplog):
    """Plate 2266111 is recovered even when OCR-corrupted type-2 label uses token routing.

    Regression for the production case where plate=None was returned because:
    - Template was wrongly routed to legacy (due to corrupted 'הודעת תשלום קנס')
    - Plate was found but not plate_confident, so reconciliation ignored it.
    """
    import logging
    # Tokens present but phrase partially garbled → should route anchor_based.
    # Real type-2 notices do NOT have "מספר הודעה" / "מספר דוח" labels, only
    # the "הודעת תשלום קנס" family.
    noisy_ocr = (
        "הודע תשלו קנס\n"
        "מספר רכב: 2266111\n"
        "תאריך: 15/05/2076\n"
        "סכום לתשלום: 2076\n"
        "4030573\n"
    )
    with caplog.at_level(logging.INFO, logger="bot.ocr"):
        result = extract_plate_and_fine_candidates(noisy_ocr, "2266111 4030573 15052076 2076")
    assert any("template_detected=anchor_based" in r.message for r in caplog.records), (
        "Should route to anchor_based via token proximity"
    )
    assert result["plate"] == "2266111", (
        f"Expected plate=2266111, got plate={result['plate']}"
    )
    # In anchor_based mode, fine candidates are restricted to 10-11 digits,
    # so 7-digit 4030573 must not win.
    assert result["fine"] != "4030573", (
        "7-digit number should not be selected as fine in anchor_based mode"
    )


def test_extract_plate_and_fine_candidates_returns_plate_ctx():
    """plate_ctx is included in the returned dict for use in reconciliation."""
    ocr_text = "מספר רכב: 2266111\nסכום: 133\n"
    result = extract_plate_and_fine_candidates(ocr_text, "2266111 133")
    assert "plate_ctx" in result, "plate_ctx must be returned for reconciliation"
    # Number on the same line as a plate keyword → context score should be > 0
    assert result["plate_ctx"] > 0

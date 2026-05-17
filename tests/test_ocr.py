import numpy as np

from bot.ocr import (
    _is_multi_preprocess_enabled,
    extract_plate_and_fine_candidates,
    preprocess_variants,
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


def test_multi_preprocess_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv("OCR_MULTI_VARIANTS", raising=False)
    monkeypatch.delenv("OCR_MULTI_PREPROCESS", raising=False)
    assert _is_multi_preprocess_enabled() is False


def test_multi_preprocess_flag_can_be_enabled_with_alias(monkeypatch):
    monkeypatch.setenv("OCR_MULTI_PREPROCESS", "0")
    monkeypatch.setenv("OCR_MULTI_VARIANTS", "1")
    assert _is_multi_preprocess_enabled() is True

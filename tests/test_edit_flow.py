"""Unit tests for the edit-flow helpers (validation, formatters)."""
import pytest

from bot.formatters import format_fine_details
from bot.handlers.edit import _validate_field


# ---------------------------------------------------------------------------
# _validate_field
# ---------------------------------------------------------------------------


class TestValidateField:
    def test_empty_value_rejected(self):
        assert _validate_field("fine_number", "", "en") is not None

    def test_whitespace_only_rejected(self):
        assert _validate_field("fine_number", "   ", "en") is not None

    # fine_number accepts any non-empty string (can contain letters/digits)
    def test_fine_number_valid(self):
        assert _validate_field("fine_number", "51903219", "en") is None

    def test_fine_number_alphanumeric(self):
        assert _validate_field("fine_number", "AB1234", "en") is None

    # date fields
    @pytest.mark.parametrize("val", ["3/4/2023", "03/04/2023", "3.4.2023", "03.04.2023"])
    def test_date_valid_formats(self, val):
        assert _validate_field("fine_date", val, "en") is None

    @pytest.mark.parametrize("val", ["2023-04-03", "3-4-2023", "notadate"])
    def test_date_invalid_formats(self, val):
        assert _validate_field("fine_date", val, "en") is not None

    @pytest.mark.parametrize("val", ["3/4/2023", "2.7.2023"])
    def test_payment_deadline_valid(self, val):
        assert _validate_field("payment_deadline", val, "en") is None

    # amount field
    @pytest.mark.parametrize("val", ["100", "250.50", "0"])
    def test_amount_valid(self, val):
        assert _validate_field("fine_amount", val, "en") is None

    @pytest.mark.parametrize("val", ["abc", "10,5", "$100"])
    def test_amount_invalid(self, val):
        assert _validate_field("fine_amount", val, "en") is not None

    # non-validated fields just need to be non-empty
    @pytest.mark.parametrize("field", ["vehicle_plate", "violation", "location"])
    def test_free_text_fields_accept_any_nonempty(self, field):
        assert _validate_field(field, "anything", "en") is None

    def test_returns_localised_message(self):
        error_ru = _validate_field("fine_date", "bad-date", "ru")
        error_en = _validate_field("fine_date", "bad-date", "en")
        assert error_ru is not None
        assert error_en is not None
        assert error_ru != error_en  # different languages → different strings


# ---------------------------------------------------------------------------
# format_fine_details
# ---------------------------------------------------------------------------


class TestFormatFineDetails:
    def _make_field(self, value, confidence=0.9, manual=False):
        d = {"value": value, "confidence": confidence}
        if manual:
            d["manual"] = True
        return d

    def test_high_confidence_uses_checkmark(self):
        details = {"fine_number": self._make_field("12345", 0.95)}
        text, has_low = format_fine_details(details, "en")
        assert "✅" in text
        assert "12345" in text
        assert not has_low

    def test_medium_confidence_uses_warning(self):
        details = {"fine_number": self._make_field("12345", 0.6)}
        text, has_low = format_fine_details(details, "en")
        assert "⚠️" in text
        assert has_low

    def test_low_confidence_uses_cross(self):
        details = {"fine_number": self._make_field("12345", 0.3)}
        text, has_low = format_fine_details(details, "en")
        assert "❌" in text
        assert has_low

    def test_manual_field_uses_pencil_and_no_confidence_label(self):
        details = {"fine_number": self._make_field("51903219", manual=True)}
        text, has_low = format_fine_details(details, "en")
        assert "✏️" in text
        assert "51903219" in text
        # no confidence bracket label
        assert "[" not in text
        assert not has_low

    def test_missing_field_is_skipped(self):
        text, _ = format_fine_details({}, "en")
        assert text == ""

    def test_non_dict_field_is_skipped(self):
        details = {"fine_number": "plain string"}
        text, _ = format_fine_details(details, "en")
        assert text == ""

    def test_empty_value_shows_dash(self):
        details = {"fine_number": {"value": None, "confidence": 0.9}}
        text, _ = format_fine_details(details, "en")
        assert "—" in text

    def test_multiple_fields_in_canonical_order(self):
        details = {
            "payment_deadline": self._make_field("30.06.2023"),
            "fine_number": self._make_field("99"),
        }
        text, _ = format_fine_details(details, "en")
        fine_num_pos = text.index("99")
        deadline_pos = text.index("30.06.2023")
        assert fine_num_pos < deadline_pos  # fine_number comes before payment_deadline

    def test_localisation_ru(self):
        details = {"fine_number": self._make_field("1", 0.9)}
        text, _ = format_fine_details(details, "ru")
        assert "высокая" in text

    def test_localisation_he(self):
        details = {"fine_number": self._make_field("1", 0.9)}
        text, _ = format_fine_details(details, "he")
        assert "גבוה" in text

"""Unit tests for the edit-flow helpers (validation, formatters)."""
import pytest

from bot.formatters import format_fine_details
from bot.handlers.edit import _validate_field
from bot.i18n import t
from bot.keyboards import appeal_reason_keyboard, confirmation_keyboard


# ---------------------------------------------------------------------------
# _validate_field
# ---------------------------------------------------------------------------


class TestValidateField:
    def test_empty_value_rejected(self):
        assert _validate_field("fine_number", "", "en") is not None

    def test_whitespace_only_rejected(self):
        assert _validate_field("fine_number", "   ", "en") is not None

    # fine_number: digits-only after normalization, 6-12 digits
    def test_fine_number_valid(self):
        assert _validate_field("fine_number", "51903219", "en") is None

    def test_fine_number_with_separators_valid(self):
        assert _validate_field("fine_number", "51-9032 19", "en") is None

    def test_fine_number_alphanumeric_invalid(self):
        assert _validate_field("fine_number", "AB1234", "en") is not None

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
        # Russian error should contain Cyrillic text
        assert any("\u0400" <= c <= "\u04FF" for c in error_ru)
        # English error should not be the same as Russian
        assert error_ru != error_en


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


# ---------------------------------------------------------------------------
# confirmation_keyboard
# ---------------------------------------------------------------------------


class TestConfirmationKeyboard:
    @pytest.mark.parametrize("lang", ["ru", "en", "he"])
    def test_has_two_buttons(self, lang):
        kb = confirmation_keyboard(lang)
        all_buttons = [btn for row in kb.inline_keyboard for btn in row]
        assert len(all_buttons) == 2

    @pytest.mark.parametrize("lang", ["ru", "en", "he"])
    def test_confirm_callback(self, lang):
        kb = confirmation_keyboard(lang)
        all_buttons = [btn for row in kb.inline_keyboard for btn in row]
        callbacks = [btn.callback_data for btn in all_buttons]
        assert "confirm_details" in callbacks

    @pytest.mark.parametrize("lang", ["ru", "en", "he"])
    def test_incorrect_callback(self, lang):
        kb = confirmation_keyboard(lang)
        all_buttons = [btn for row in kb.inline_keyboard for btn in row]
        callbacks = [btn.callback_data for btn in all_buttons]
        assert "edit_details" in callbacks

    def test_button_labels_localised(self):
        kb_ru = confirmation_keyboard("ru")
        kb_en = confirmation_keyboard("en")
        ru_texts = [btn.text for row in kb_ru.inline_keyboard for btn in row]
        en_texts = [btn.text for row in kb_en.inline_keyboard for btn in row]
        assert ru_texts != en_texts


# ---------------------------------------------------------------------------
# appeal_reason_keyboard
# ---------------------------------------------------------------------------


class TestAppealReasonKeyboard:
    @pytest.mark.parametrize("lang", ["ru", "en", "he"])
    def test_has_five_buttons(self, lang):
        kb = appeal_reason_keyboard(lang)
        all_buttons = [btn for row in kb.inline_keyboard for btn in row]
        assert len(all_buttons) == 5

    @pytest.mark.parametrize("lang", ["ru", "en", "he"])
    def test_reason_callbacks(self, lang):
        kb = appeal_reason_keyboard(lang)
        all_buttons = [btn for row in kb.inline_keyboard for btn in row]
        callbacks = {btn.callback_data for btn in all_buttons}
        expected = {"reason_1", "reason_2", "reason_3", "reason_4", "reason_other"}
        assert callbacks == expected

    def test_button_labels_localised(self):
        kb_ru = appeal_reason_keyboard("ru")
        kb_en = appeal_reason_keyboard("en")
        ru_texts = [btn.text for row in kb_ru.inline_keyboard for btn in row]
        en_texts = [btn.text for row in kb_en.inline_keyboard for btn in row]
        assert ru_texts != en_texts


# ---------------------------------------------------------------------------
# i18n – new locale keys
# ---------------------------------------------------------------------------


class TestNewLocaleKeys:
    NEW_KEYS = [
        "btn_data_correct",
        "btn_data_incorrect",
        "choose_appeal_reason",
        "reason_1",
        "reason_2",
        "reason_3",
        "reason_4",
        "reason_other",
        "enter_appeal_reason",
    ]

    @pytest.mark.parametrize("lang", ["ru", "en", "he"])
    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_key_exists_and_not_fallback(self, lang, key):
        """Each new locale key must resolve to a real translation, not the bare key."""
        value = t(key, lang)
        assert value != key, f"Missing translation: key={key!r} lang={lang!r}"
        # All translated strings should be at least a few characters long
        # (even the shortest expected text like "✅" followed by a word)
        MIN_TRANSLATION_LENGTH = 3
        assert len(value) > MIN_TRANSLATION_LENGTH

"""Handlers for the edit-extracted-data and back/return UX flow.

State machine
-------------
None             – normal state (summary displayed with 3-button keyboard)
choosing_field   – user is looking at the field-selection menu
entering_value   – user is typing a corrected value for a chosen field

Callback data contract
----------------------
edit_details          – enter edit mode (show field-selection menu)
back_to_summary       – field-selection → back to summary
back_to_start         – summary → send-file prompt ("start over")
edit_field:<key>      – user chose a specific field to edit
back_to_fields        – value-input → back to field-selection
"""
import logging
import re
from typing import Optional

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.db import add_audit_log, get_latest_case, get_or_create_user, update_case_details
from bot.formatters import FIELD_KEYS, field_label, format_fine_details
from bot.i18n import t
from bot.keyboards import confirmation_keyboard

logger = logging.getLogger(__name__)
router = Router()

# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------


class EditStates(StatesGroup):
    choosing_field = State()
    entering_value = State()


# ---------------------------------------------------------------------------
# Keyboards (field-selection and value-input; confirmation_keyboard is in bot.keyboards)
# ---------------------------------------------------------------------------


def _field_selection_keyboard(lang: str) -> InlineKeyboardMarkup:
    """One button per editable field, plus a Back button."""
    btn_key_map = {
        "fine_number": "btn_field_fine_number",
        "fine_date": "btn_field_fine_date",
        "fine_amount": "btn_field_fine_amount",
        "vehicle_plate": "btn_field_vehicle_plate",
        "violation": "btn_field_violation",
        "location": "btn_field_location",
        "payment_deadline": "btn_field_payment_deadline",
    }
    buttons = [
        [
            InlineKeyboardButton(
                text=t(btn_key_map[field], lang),
                callback_data=f"edit_field:{field}",
            )
        ]
        for field in FIELD_KEYS
    ]
    buttons.append(
        [InlineKeyboardButton(text=t("btn_back", lang), callback_data="back_to_summary")]
    )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _back_keyboard(lang: str) -> InlineKeyboardMarkup:
    """Single Back button shown under the value-input prompt."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t("btn_back", lang), callback_data="back_to_fields")]
        ]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _show_summary(target: Message, user_id: int, lang: str) -> None:
    """Fetch the latest case and display the fine-details summary."""
    case = await get_latest_case(user_id)
    if not case:
        await target.answer(t("no_case", lang))
        return
    details = case["data"].get("details", {})
    details_text, has_low = format_fine_details(details, lang)
    if details_text:
        await target.answer(t("fine_details", lang, details=details_text))
    if has_low:
        await target.answer(t("low_confidence", lang))
    await target.answer(t("confirm_details", lang), reply_markup=confirmation_keyboard(lang))


def _validate_field(field: str, value: str, lang: str) -> Optional[str]:
    """Return an error message string if the value is invalid, else None."""
    value = value.strip()
    if not value:
        return t("edit_empty_value", lang)
    if field in ("fine_date", "payment_deadline"):
        if not re.match(r"^\d{1,2}[/\.]\d{1,2}[/\.]\d{4}$", value):
            return t("edit_invalid_date", lang)
    if field == "fine_amount":
        if not re.match(r"^\d+(\.\d+)?$", value):
            return t("edit_invalid_amount", lang)
    return None


# ---------------------------------------------------------------------------
# Callbacks: enter / leave edit mode
# ---------------------------------------------------------------------------


@router.callback_query(F.data == "edit_details")
async def cb_edit_details(callback: CallbackQuery, state: FSMContext) -> None:
    """Open the field-selection menu."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    await state.set_state(EditStates.choosing_field)
    await callback.message.answer(
        t("choose_field_to_edit", lang),
        reply_markup=_field_selection_keyboard(lang),
    )


@router.callback_query(F.data == "back_to_summary")
async def cb_back_to_summary(callback: CallbackQuery, state: FSMContext) -> None:
    """Return from field-selection to the summary."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    await state.clear()
    await _show_summary(callback.message, callback.from_user.id, lang)


@router.callback_query(F.data == "back_to_start")
async def cb_back_to_start(callback: CallbackQuery, state: FSMContext) -> None:
    """Return to the 'send your fine' prompt."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    await state.clear()
    await callback.message.answer(t("send_file", lang))


# ---------------------------------------------------------------------------
# Callbacks: choose a field
# ---------------------------------------------------------------------------


@router.callback_query(F.data.startswith("edit_field:"))
async def cb_edit_field(callback: CallbackQuery, state: FSMContext) -> None:
    """Prompt the user to enter a new value for the chosen field."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()

    field = callback.data.split(":", 1)[1]
    if field not in FIELD_KEYS:
        return

    await state.set_state(EditStates.entering_value)
    await state.update_data(editing_field=field)

    label = field_label(field, lang)
    await callback.message.answer(
        t("enter_new_value", lang, field=label),
        reply_markup=_back_keyboard(lang),
    )


@router.callback_query(F.data == "back_to_fields")
async def cb_back_to_fields(callback: CallbackQuery, state: FSMContext) -> None:
    """Return from the value-input prompt to the field-selection menu."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    await state.set_state(EditStates.choosing_field)
    await state.update_data(editing_field=None)
    await callback.message.answer(
        t("choose_field_to_edit", lang),
        reply_markup=_field_selection_keyboard(lang),
    )


# ---------------------------------------------------------------------------
# Message handler: receive new field value
# ---------------------------------------------------------------------------


@router.message(EditStates.entering_value)
async def handle_field_value(message: Message, state: FSMContext) -> None:
    """Validate, persist and confirm the corrected field value."""
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")

    fsm_data = await state.get_data()
    field = fsm_data.get("editing_field")
    if not field:
        await state.clear()
        return

    value = (message.text or "").strip()

    error = _validate_field(field, value, lang)
    if error:
        await message.answer(error)
        return

    case = await get_latest_case(message.from_user.id)
    if not case:
        await message.answer(t("no_case", lang))
        await state.clear()
        return

    details = case["data"].get("details", {})
    if not isinstance(details.get(field), dict):
        details[field] = {}
    details[field]["value"] = value
    details[field]["manual"] = True
    # Remove AI confidence – this field was set by the user, not extracted
    details[field].pop("confidence", None)

    await update_case_details(case["id"], details)
    await add_audit_log(
        "field_edited",
        {"user_id": message.from_user.id, "case_id": case["id"], "field": field},
    )
    logger.info(
        "Field %s updated manually for user_id=%s case_id=%s",
        field,
        message.from_user.id,
        case["id"],
    )

    await state.clear()
    await message.answer(t("edit_field_updated", lang))
    await _show_summary(message, message.from_user.id, lang)

"""Handlers for /appeal command and the "Generate Appeal" inline button.

State machine
-------------
None              – no active state
choosing_reason   – user is looking at the appeal-reason keyboard
entering_reason_other – user is typing a custom appeal reason (option 5)
"""
import logging
from typing import Optional

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from bot.db import add_audit_log, get_latest_case, get_or_create_user
from bot.i18n import t
from bot.keyboards import appeal_reason_keyboard
from bot.openai_client import generate_appeal

logger = logging.getLogger(__name__)
router = Router()

_MAX_TG_MESSAGE = 4096

# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------


class AppealStates(StatesGroup):
    choosing_reason = State()
    entering_reason_other = State()


# ---------------------------------------------------------------------------
# Hebrew template texts for fixed appeal reasons
# ---------------------------------------------------------------------------

_REASON_TEXTS_HE: dict[str, str] = {
    "reason_1": "התמרור/הסימון לא היה גלוי/ברור לעין",
    "reason_2": "שגיאה בנתוני הזיהוי (לוחית רישוי/שעה/מקום)",
    "reason_3": "עצירה קצרה עקב הכרח/כוח עליון",
    "reason_4": "בידי אישור/היתר לחנייה/עצירה",
}

# ---------------------------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------------------------


async def _do_generate_appeal(
    target_message: Message,
    user_id: int,
    lang: str,
    appeal_reason: Optional[str] = None,
) -> None:
    """Fetch the latest case for *user_id* and send a Hebrew appeal letter."""
    case = await get_latest_case(user_id)
    if not case:
        await target_message.answer(t("no_case", lang))
        return

    await target_message.answer(t("generating_appeal", lang))

    try:
        details = case["data"].get("details", {})
        appeal_text = await generate_appeal(details, appeal_reason=appeal_reason)
    except Exception as exc:
        logger.error("generate_appeal failed for user_id=%s: %s", user_id, exc)
        await target_message.answer(t("error_processing", lang))
        return

    await add_audit_log(
        "appeal_generated",
        {"user_id": user_id, "case_id": case["id"]},
    )
    logger.info("Appeal generated: user_id=%s case_id=%s", user_id, case["id"])

    # Telegram message length limit: 4096 chars
    header = t("appeal_ready", lang, appeal="")
    if len(header) + len(appeal_text) <= _MAX_TG_MESSAGE:
        await target_message.answer(t("appeal_ready", lang, appeal=appeal_text))
    else:
        await target_message.answer(t("appeal_ready", lang, appeal=""))
        for i in range(0, len(appeal_text), _MAX_TG_MESSAGE):
            await target_message.answer(appeal_text[i : i + _MAX_TG_MESSAGE])


# ---------------------------------------------------------------------------
# /appeal command  (skips reason selection for quick generation)
# ---------------------------------------------------------------------------


@router.message(Command("appeal"))
async def cmd_appeal(message: Message) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")
    await _do_generate_appeal(message, message.from_user.id, lang)


# ---------------------------------------------------------------------------
# Confirm callback: ✅ "Данные верны" → show reason selection keyboard
# ---------------------------------------------------------------------------


@router.callback_query(F.data == "confirm_details")
async def cb_confirm_details(callback: CallbackQuery, state: FSMContext) -> None:
    """Enter appeal-reason selection after the user confirms extracted data."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    await state.set_state(AppealStates.choosing_reason)
    await callback.message.answer(
        t("choose_appeal_reason", lang),
        reply_markup=appeal_reason_keyboard(lang),
    )


# ---------------------------------------------------------------------------
# Reason callbacks: fixed reasons 1–4
# ---------------------------------------------------------------------------


@router.callback_query(
    AppealStates.choosing_reason,
    F.data.in_({"reason_1", "reason_2", "reason_3", "reason_4"}),
)
async def cb_reason_fixed(callback: CallbackQuery, state: FSMContext) -> None:
    """Map a fixed reason to its Hebrew template and generate the appeal."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    reason_he = _REASON_TEXTS_HE[callback.data]
    await state.clear()
    await _do_generate_appeal(
        callback.message, callback.from_user.id, lang, appeal_reason=reason_he
    )


# ---------------------------------------------------------------------------
# Reason callback: "Other" – prompt for free-text reason
# ---------------------------------------------------------------------------


@router.callback_query(AppealStates.choosing_reason, F.data == "reason_other")
async def cb_reason_other(callback: CallbackQuery, state: FSMContext) -> None:
    """Ask the user to type a custom reason."""
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()
    await state.set_state(AppealStates.entering_reason_other)
    await callback.message.answer(t("enter_appeal_reason", lang))


# ---------------------------------------------------------------------------
# Message handler: receive free-text reason
# ---------------------------------------------------------------------------


@router.message(AppealStates.entering_reason_other)
async def handle_reason_other(message: Message, state: FSMContext) -> None:
    """Validate, save the custom reason and generate the appeal."""
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")
    reason_text = (message.text or "").strip()
    if not reason_text:
        await message.answer(t("edit_empty_value", lang))
        return
    await state.clear()
    await _do_generate_appeal(
        message, message.from_user.id, lang, appeal_reason=reason_text
    )

"""Handlers for /appeal command and the "Generate Appeal" inline button."""
import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.db import add_audit_log, get_latest_case, get_or_create_user
from bot.i18n import t
from bot.openai_client import generate_appeal

logger = logging.getLogger(__name__)
router = Router()

_MAX_TG_MESSAGE = 4096


async def _do_generate_appeal(target_message: Message, user_id: int, lang: str) -> None:
    """Fetch the latest case for *user_id* and send a Hebrew appeal letter."""
    case = await get_latest_case(user_id)
    if not case:
        await target_message.answer(t("no_case", lang))
        return

    await target_message.answer(t("generating_appeal", lang))

    try:
        details = case["data"].get("details", {})
        appeal_text = await generate_appeal(details)
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
# /appeal command
# ---------------------------------------------------------------------------

@router.message(Command("appeal"))
async def cmd_appeal(message: Message) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")
    await _do_generate_appeal(message, message.from_user.id, lang)


# ---------------------------------------------------------------------------
# Inline button callback
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "generate_appeal")
async def cb_generate_appeal(callback: CallbackQuery) -> None:
    user = await get_or_create_user(callback.from_user.id)
    lang = user.get("language", "ru")
    await callback.answer()  # acknowledge the press immediately
    await _do_generate_appeal(callback.message, callback.from_user.id, lang)

"""Handlers for /start, /language, /help, /delete commands."""
import logging
import os

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.db import add_audit_log, delete_user_data, get_or_create_user, update_user_language
from bot.i18n import t

logger = logging.getLogger(__name__)
router = Router()


def _language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang_ru"),
                InlineKeyboardButton(text="🇮🇱 עברית", callback_data="lang_he"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
            ]
        ]
    )


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")
    logger.info("cmd_start: user_id=%s", message.from_user.id)
    await add_audit_log(
        "start",
        {"user_id": message.from_user.id, "username": message.from_user.username},
    )
    await message.answer(t("welcome", lang), reply_markup=_language_keyboard())


# ---------------------------------------------------------------------------
# /language + inline callback
# ---------------------------------------------------------------------------

@router.message(Command("language"))
async def cmd_language(message: Message) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")
    await message.answer(t("choose_language", lang), reply_markup=_language_keyboard())


@router.callback_query(F.data.startswith("lang_"))
async def cb_set_language(callback: CallbackQuery) -> None:
    lang = callback.data.split("_", 1)[1]  # "lang_ru" → "ru"
    await update_user_language(callback.from_user.id, lang)
    logger.info("Language set: user_id=%s lang=%s", callback.from_user.id, lang)
    await callback.answer(t("language_set", lang))
    await callback.message.edit_text(t("language_set", lang))


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")
    await message.answer(t("help", lang))


# ---------------------------------------------------------------------------
# /delete
# ---------------------------------------------------------------------------

@router.message(Command("delete"))
async def cmd_delete(message: Message) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")

    deleted_count, file_paths = await delete_user_data(message.from_user.id)

    # Purge encrypted files from disk
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.info("Deleted encrypted file: %s", path)
        except OSError as exc:
            logger.error("Could not delete file %s: %s", path, exc)

    logger.info("cmd_delete: user_id=%s cases_deleted=%d", message.from_user.id, deleted_count)
    await add_audit_log(
        "data_deleted",
        {"user_id": message.from_user.id, "cases_deleted": deleted_count},
    )
    await message.answer(t("data_deleted", lang))

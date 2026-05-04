"""Reusable inline keyboard builders.

Kept in a separate module so that both the upload and edit handlers can
import them without creating circular dependencies.
"""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.i18n import t


def confirmation_keyboard(lang: str) -> InlineKeyboardMarkup:
    """3-button keyboard shown under the fine-details summary.

    Buttons
    -------
    ✅ Confirm  – generate the appeal (callback: generate_appeal)
    ✏️ Edit     – open field-selection menu (callback: edit_details)
    ↩️ Back     – return to the send-file prompt (callback: back_to_start)
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t("btn_confirm", lang), callback_data="generate_appeal"
                ),
                InlineKeyboardButton(
                    text=t("btn_edit", lang), callback_data="edit_details"
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t("btn_back", lang), callback_data="back_to_start"
                ),
            ],
        ]
    )

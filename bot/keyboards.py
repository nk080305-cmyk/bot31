"""Reusable inline keyboard builders.

Kept in a separate module so that both the upload and edit handlers can
import them without creating circular dependencies.
"""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.i18n import t


def confirmation_keyboard(lang: str) -> InlineKeyboardMarkup:
    """2-button keyboard shown under the fine-details summary.

    Buttons
    -------
    ✅ Data correct    – proceed to appeal-reason selection (callback: confirm_details)
    ❌ Data incorrect  – open field-selection menu (callback: edit_details)
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=t("btn_data_correct", lang), callback_data="confirm_details"
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t("btn_data_incorrect", lang), callback_data="edit_details"
                ),
            ],
        ]
    )


def appeal_reason_keyboard(lang: str) -> InlineKeyboardMarkup:
    """5-button keyboard for selecting the appeal reason.

    Four fixed reasons map to Hebrew template text.
    The fifth prompts for free-text input.
    """
    reason_keys = [
        "reason_1",
        "reason_2",
        "reason_3",
        "reason_4",
        "reason_other",
    ]
    buttons = [
        [InlineKeyboardButton(text=t(key, lang), callback_data=key)]
        for key in reason_keys
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

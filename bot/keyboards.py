"""
Inline keyboard factories.
"""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def confirm_keyboard() -> InlineKeyboardMarkup:
    """Shown after AI analysis so the user can confirm, edit, or cancel."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✅ Подтвердить и отправить", callback_data="confirm")],
            [InlineKeyboardButton("✏️ Редактировать текст жалобы", callback_data="edit")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )


def retry_keyboard() -> InlineKeyboardMarkup:
    """Shown when submission fails so the user can retry or cancel."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Повторить попытку", callback_data="retry")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )


def cancel_keyboard() -> InlineKeyboardMarkup:
    """Single cancel button."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("❌ Отмена", callback_data="cancel")]]
    )

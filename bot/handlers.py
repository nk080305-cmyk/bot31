"""
FSM conversation handlers for the Telegram bot.

State flow:
  START → WAITING_FILE → PROCESSING → CONFIRM → SUBMITTING → done
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from telegram import Update, Message
from telegram.constants import ParseMode
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from bot.keyboards import confirm_keyboard, retry_keyboard, cancel_keyboard
from config.settings import TEMP_DIR

logger = logging.getLogger(__name__)

# ── FSM states ────────────────────────────────────────────────────────────────
WAITING_FILE = 1
PROCESSING   = 2
CONFIRM      = 3
SUBMITTING   = 4

# ── Context data keys ─────────────────────────────────────────────────────────
KEY_VIOLATION    = "violation"
KEY_APPEAL       = "appeal_text"
KEY_ATTACHMENT   = "attachment_path"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_violation(v: dict) -> str:
    lines = [
        f"📋 *Данные нарушения*",
        f"  Номер дела: `{v.get('case_number') or '—'}`",
        f"  Дата/время: `{v.get('violation_date') or '—'} {v.get('violation_time') or ''}`.strip()",
        f"  Место: `{v.get('location') or '—'}`",
        f"  Тип нарушения: `{v.get('violation_type') or '—'}`",
        f"  Сумма штрафа: `{v.get('fine_amount') or '—'}`",
        f"  Орган: `{v.get('issuing_authority') or '—'}`",
        f"  Владелец: `{v.get('owner_name') or '—'}` (ID: `{v.get('owner_id') or '—'}`)",
        f"  Номер ТС: `{v.get('vehicle_number') or '—'}`",
    ]
    return "\n".join(lines)


async def _download_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Download the received photo or document to TEMP_DIR and return the path."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    message: Message = update.message  # type: ignore

    if message.photo:
        file_obj = await message.photo[-1].get_file()
        dest = Path(TEMP_DIR) / f"{file_obj.file_id}.jpg"
    elif message.document:
        file_obj = await message.document.get_file()
        ext = Path(message.document.file_name or "doc.pdf").suffix or ".pdf"
        dest = Path(TEMP_DIR) / f"{file_obj.file_id}{ext}"
    else:
        raise ValueError("No downloadable file in message")

    await file_obj.download_to_drive(str(dest))
    return str(dest)


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (
        "👋 *Привет\\!*\n\n"
        "Я помогу вам подать обжалование на штраф ПДД в Израиле\\.\n\n"
        "📄 *Как это работает:*\n"
        "1\\. Отправьте мне *фото* или *PDF* письма о штрафе\\.\n"
        "2\\. Я извлеку данные и сформирую текст обжалования на иврите\\.\n"
        "3\\. Вы проверите данные и подтвердите отправку\\.\n"
        "4\\. Я автоматически подам обжалование на официальный сайт\\.\n\n"
        "⚠️ *Важно:* Бот помогает сформировать обжалование, но "
        "ответственность за его содержание несёт пользователь\\.\n"
        "Личные данные хранятся только в рамках вашей сессии\\.\n\n"
        "📎 Пожалуйста, отправьте файл письма\\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    return WAITING_FILE


async def receive_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "⏳ Получил файл. Извлекаю текст…",
        reply_markup=cancel_keyboard(),
    )

    # Download
    try:
        file_path = await _download_file(update, context)
    except Exception as exc:
        logger.error("File download error: %s", exc)
        await update.message.reply_text("❌ Не удалось загрузить файл. Попробуйте ещё раз.")
        return WAITING_FILE

    context.user_data[KEY_ATTACHMENT] = file_path

    # OCR (run in thread pool to keep event loop free)
    try:
        from ocr.extractor import extract_text
        ocr_text = await asyncio.get_event_loop().run_in_executor(None, extract_text, file_path)
    except Exception as exc:
        logger.error("OCR error: %s", exc)
        await update.message.reply_text(
            "❌ Не удалось извлечь текст из файла. "
            "Убедитесь, что изображение чёткое, и попробуйте ещё раз."
        )
        return WAITING_FILE

    if not ocr_text.strip():
        await update.message.reply_text(
            "❌ Текст не найден. Пожалуйста, пришлите более чёткое изображение."
        )
        return WAITING_FILE

    await update.message.reply_text("🔍 Анализирую данные с помощью AI…")

    # AI analysis
    try:
        from ai.analyzer import analyze_violation, generate_appeal
        violation = await asyncio.get_event_loop().run_in_executor(
            None, analyze_violation, ocr_text
        )
        appeal = await asyncio.get_event_loop().run_in_executor(
            None, generate_appeal, violation
        )
    except Exception as exc:
        logger.error("AI analysis error: %s", exc)
        await update.message.reply_text("❌ Ошибка AI-анализа. Попробуйте позже.")
        return WAITING_FILE

    context.user_data[KEY_VIOLATION] = violation
    context.user_data[KEY_APPEAL]    = appeal

    # Show results for confirmation
    summary = _format_violation(violation)
    await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
    await update.message.reply_text(
        f"📝 *Текст обжалования (иврит):*\n\n{appeal}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=confirm_keyboard(),
    )
    return CONFIRM


async def callback_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "cancel":
        await query.edit_message_text("❌ Операция отменена. Для начала используйте /start.")
        context.user_data.clear()
        return ConversationHandler.END

    if query.data == "edit":
        await query.edit_message_text(
            "✏️ Введите исправленный текст обжалования:"
        )
        return CONFIRM  # will receive the edited text in next text message handler

    # data == "confirm"
    await query.edit_message_text("🚀 Отправляю обжалование…")
    return await _do_submit(update, context)


async def receive_edited_appeal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User sent a corrected appeal text."""
    context.user_data[KEY_APPEAL] = update.message.text
    await update.message.reply_text(
        "✅ Текст обновлён. Подтвердите отправку:",
        reply_markup=confirm_keyboard(),
    )
    return CONFIRM


async def callback_retry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "cancel":
        await query.edit_message_text("❌ Операция отменена. Для начала используйте /start.")
        context.user_data.clear()
        return ConversationHandler.END
    await query.edit_message_text("🔄 Повторяю попытку…")
    return await _do_submit(update, context)


async def _do_submit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Common submission logic called from confirm / retry callbacks."""
    violation  = context.user_data.get(KEY_VIOLATION, {})
    appeal     = context.user_data.get(KEY_APPEAL, "")
    attachment = context.user_data.get(KEY_ATTACHMENT)

    chat_id = (
        update.effective_chat.id  # type: ignore
    )

    try:
        from scraper.submitter import submit_appeal
        success, screenshot_path = await submit_appeal(violation, appeal, attachment)
    except Exception as exc:
        logger.error("Submission exception: %s", exc)
        success, screenshot_path = False, None

    bot = context.bot
    if success:
        await bot.send_message(
            chat_id,
            "✅ Обжалование успешно подано! Скриншот подтверждения:",
        )
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as f:
                await bot.send_photo(chat_id, photo=f)
        context.user_data.clear()
        return ConversationHandler.END
    else:
        msg = "❌ Не удалось подать обжалование. Сайт может быть недоступен."
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as f:
                await bot.send_photo(chat_id, photo=f, caption=msg)
        else:
            await bot.send_message(chat_id, msg)
        await bot.send_message(
            chat_id,
            "Попробовать ещё раз?",
            reply_markup=retry_keyboard(),
        )
        return SUBMITTING


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("❌ Операция отменена. Для начала используйте /start.")
    return ConversationHandler.END


# ─────────────────────────────────────────────────────────────────────────────
# ConversationHandler factory
# ─────────────────────────────────────────────────────────────────────────────

def build_conversation_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            WAITING_FILE: [
                MessageHandler(
                    filters.PHOTO | filters.Document.ALL,
                    receive_file,
                ),
            ],
            CONFIRM: [
                CallbackQueryHandler(callback_confirm, pattern="^(confirm|edit|cancel)$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_edited_appeal),
            ],
            SUBMITTING: [
                CallbackQueryHandler(callback_retry, pattern="^(retry|cancel)$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
    )

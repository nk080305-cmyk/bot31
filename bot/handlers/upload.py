"""Handlers for photo and document uploads.

Pipeline
--------
1. Validate file size (≤ 15 MB) and extension.
2. Download to a temporary file.
3. OCR with OpenCV preprocessing + Tesseract (heb+eng, PSM 6/4/11).
4. Extract structured fine data via OpenAI.
5. Encrypt the raw file and persist to /data/cases/.
6. Store an encrypted case record in SQLite.
7. Present the extracted data to the user with Confirm / Edit / Back buttons.
"""
import logging
import os
import tempfile
import uuid

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.config import CASES_DIR, MAX_FILE_SIZE
from bot.db import add_audit_log, get_or_create_user, save_case
from bot.encryption import encrypt_bytes
from bot.formatters import format_fine_details
from bot.i18n import t
from bot.ocr import extract_text
from bot.openai_client import extract_fine_details as ai_extract_fine_details

logger = logging.getLogger(__name__)
router = Router()

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

async def _process_file(
    message: Message, state: FSMContext, file_id: str, file_name: str, file_size: int
) -> None:
    user = await get_or_create_user(message.from_user.id)
    lang = user.get("language", "ru")

    # --- size check ---
    if file_size > MAX_FILE_SIZE:
        await message.answer(t("file_too_large", lang))
        return

    ext = os.path.splitext(file_name)[1].lower() if file_name else ".jpg"
    if ext not in _ALLOWED_EXTENSIONS:
        await message.answer(t("unsupported_format", lang))
        return

    await message.answer(t("processing", lang))

    tmp_path: str | None = None
    try:
        # --- download ---
        file_meta = await message.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
        await message.bot.download_file(file_meta.file_path, destination=tmp_path)

        # --- OCR ---
        try:
            ocr_text = extract_text(tmp_path)
        except Exception as exc:
            logger.error("OCR failed for user_id=%s: %s", message.from_user.id, exc)
            await message.answer(t("ocr_failed", lang))
            return

        if not ocr_text or len(ocr_text.strip()) < 10:
            await message.answer(t("ocr_failed", lang))
            return

        await message.answer(t("ocr_done", lang))

        # --- OpenAI extraction ---
        try:
            details = await ai_extract_fine_details(ocr_text)
        except Exception as exc:
            logger.error("Extraction failed for user_id=%s: %s", message.from_user.id, exc)
            await message.answer(t("extraction_failed", lang))
            return

        await message.answer(t("extraction_done", lang))

        # --- Encrypt and persist file ---
        os.makedirs(CASES_DIR, exist_ok=True)
        enc_filename = f"{uuid.uuid4().hex}.enc"
        enc_path = os.path.join(CASES_DIR, enc_filename)
        with open(tmp_path, "rb") as fh:
            raw_bytes = fh.read()
        with open(enc_path, "wb") as fh:
            fh.write(encrypt_bytes(raw_bytes))

        # --- Encrypt and save case to DB ---
        case_data = {
            "ocr_text": ocr_text[:2000],
            "details": details,
            "original_file_name": file_name,
        }
        case_id = await save_case(message.from_user.id, case_data, enc_path)

        # --- Audit log (PII-containing payload, stored encrypted) ---
        await add_audit_log(
            "file_processed",
            {
                "user_id": message.from_user.id,
                "case_id": case_id,
                "file_name": file_name,
            },
        )
        logger.info("Case %s created for user_id=%s", case_id, message.from_user.id)

        # --- Present results ---
        from bot.handlers.edit import confirmation_keyboard  # local import avoids circular

        details_text, has_low = format_fine_details(details, lang)
        if details_text:
            await message.answer(t("fine_details", lang, details=details_text))

        if has_low:
            await message.answer(t("low_confidence", lang))

        await message.answer(t("confirm_details", lang), reply_markup=confirmation_keyboard(lang))

    except Exception as exc:
        logger.error("Unexpected error for user_id=%s: %s", message.from_user.id, exc)
        await message.answer(t("error_processing", lang))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]  # largest available resolution
    file_name = f"{photo.file_unique_id}.jpg"
    file_size = photo.file_size or 0
    logger.info("Photo received: user_id=%s size=%d", message.from_user.id, file_size)
    await state.clear()  # reset any ongoing edit session
    await _process_file(message, state, photo.file_id, file_name, file_size)


@router.message(F.document)
async def handle_document(message: Message, state: FSMContext) -> None:
    doc = message.document
    file_name = doc.file_name or "document"
    file_size = doc.file_size or 0
    logger.info(
        "Document received: user_id=%s name=%s size=%d",
        message.from_user.id,
        file_name,
        file_size,
    )
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        user = await get_or_create_user(message.from_user.id)
        lang = user.get("language", "ru")
        await message.answer(t("unsupported_format", lang))
        return
    await state.clear()  # reset any ongoing edit session
    await _process_file(message, state, doc.file_id, file_name, file_size)

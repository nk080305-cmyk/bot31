"""Bot entry point.

Start with:
    python -m bot.main

or inside Docker:
    CMD ["python", "-m", "bot.main"]
"""
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import CASES_DIR, DATA_DIR, DB_PATH, LOG_LEVEL, TELEGRAM_BOT_TOKEN
from bot.db import cleanup_expired, init_db
from bot.handlers.appeal import router as appeal_router
from bot.handlers.common import router as common_router
from bot.handlers.upload import router as upload_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------

async def _periodic_cleanup(interval_seconds: int = 3600) -> None:
    """Delete expired cases (and their encrypted files) once per hour."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            file_paths = await cleanup_expired()
            for path in file_paths:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError as exc:
                    logger.error("Could not delete expired file %s: %s", path, exc)
            if file_paths:
                logger.info("Periodic cleanup removed %d expired file(s)", len(file_paths))
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover
            logger.error("Cleanup task error: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # Ensure runtime directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CASES_DIR, exist_ok=True)

    # Initialise database schema
    await init_db()

    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(common_router)
    dp.include_router(upload_router)
    dp.include_router(appeal_router)

    logger.info("Bot starting (DB=%s)", DB_PATH)

    cleanup_task = asyncio.create_task(_periodic_cleanup())
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
        await bot.session.close()
        logger.info("Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())

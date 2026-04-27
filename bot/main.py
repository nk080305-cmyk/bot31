"""
Entry point — initialises the bot and starts polling.
"""
from __future__ import annotations

import logging
import os

from telegram.ext import Application, ApplicationBuilder

from config.settings import TELEGRAM_BOT_TOKEN, REDIS_URL, LOG_LEVEL
from bot.handlers import build_conversation_handler


def _configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    )


def _build_app() -> Application:
    """Build and configure the Telegram Application."""
    try:
        from telegram.ext import RedisStorage  # type: ignore
        storage = RedisStorage.from_url(REDIS_URL)
        builder = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).persistence(storage)
        logging.getLogger(__name__).info("Using Redis persistence: %s", REDIS_URL)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Redis unavailable (%s); falling back to in-memory storage.", exc
        )
        builder = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN)

    app = builder.build()
    app.add_handler(build_conversation_handler())
    return app


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Israeli Traffic Violation Appeal Bot…")
    app = _build_app()
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()

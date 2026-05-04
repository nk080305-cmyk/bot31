"""Internationalization helper.

Locale files (JSON dictionaries) live in ``bot/locales/<lang>.json``.
Supported language codes: ``ru``, ``he``, ``en``.

Usage::

    from bot.i18n import t
    msg = t("welcome", lang="ru")
    msg = t("fine_details", lang="he", details="...")
"""
import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

LOCALES_DIR = os.path.join(os.path.dirname(__file__), "locales")
SUPPORTED_LANGUAGES = ("ru", "he", "en")
DEFAULT_LANGUAGE = "ru"

_translations: Dict[str, Dict[str, str]] = {}


def _load_translations() -> None:
    for lang in SUPPORTED_LANGUAGES:
        path = os.path.join(LOCALES_DIR, f"{lang}.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                _translations[lang] = json.load(fh)
        except FileNotFoundError:
            logger.error("Locale file not found: %s", path)
            _translations[lang] = {}


def t(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs: str) -> str:
    """Return the localised string for *key* in *lang*.

    Falls back to English, then Russian, then the bare *key* itself.
    Named placeholders in the locale string (e.g. ``{details}``) are filled
    using simple ``str.replace`` so that user-supplied values that contain
    curly braces cannot break formatting.
    """
    text: str = (
        _translations.get(lang, {}).get(key)
        or _translations.get("en", {}).get(key)
        or _translations.get(DEFAULT_LANGUAGE, {}).get(key)
        or key
    )
    for placeholder, value in kwargs.items():
        text = text.replace(f"{{{placeholder}}}", str(value))
    return text


_load_translations()

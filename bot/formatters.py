"""Shared formatters for fine details display.

Used by both the upload handler (initial display) and the edit handler
(re-display after corrections).
"""
from typing import Tuple

from bot.i18n import t

# Canonical order of editable fields
FIELD_KEYS = [
    "fine_number",
    "fine_date",
    "fine_amount",
    "vehicle_plate",
    "violation",
    "location",
    "payment_deadline",
]

# Map each field key to its locale key
_FIELD_LOCALE_KEY = {
    "fine_number": "fine_number",
    "fine_date": "fine_date",
    "fine_amount": "fine_amount",
    "violation": "fine_violation",
    "vehicle_plate": "fine_vehicle",
    "location": "fine_location",
    "payment_deadline": "fine_deadline",
}


def field_label(field: str, lang: str) -> str:
    """Return the localised label for a fine field."""
    return t(_FIELD_LOCALE_KEY.get(field, field), lang)


def format_fine_details(details: dict, lang: str) -> Tuple[str, bool]:
    """Render extracted fine fields as a human-readable string.

    Fields edited by the user are shown with a ✏️ emoji and no confidence
    label.  Other fields follow the normal confidence-based display.

    Returns
    -------
    (formatted_text, has_low_confidence)
    """
    lines = []
    has_low = False

    for field in FIELD_KEYS:
        data = details.get(field)
        if not isinstance(data, dict):
            continue
        value = data.get("value") or "—"
        label = field_label(field, lang)

        if data.get("manual"):
            lines.append(f"✏️ {label}: {value}")
        else:
            confidence: float = data.get("confidence", 0.0)

            if confidence >= 0.8:
                emoji = "✅"
            elif confidence >= 0.5:
                emoji = "⚠️"
                has_low = True
            else:
                emoji = "❌"
                has_low = True

            conf_label = (
                t("confidence_high", lang)
                if confidence >= 0.8
                else t("confidence_medium", lang)
                if confidence >= 0.5
                else t("confidence_low", lang)
            )
            lines.append(f"{emoji} {label}: {value}  [{conf_label}]")

    return "\n".join(lines), has_low

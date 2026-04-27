"""
AI analysis module — powered by OpenAI GPT-4o.

Two public functions:
  * analyze_violation(ocr_text) → ViolationData (structured dict)
  * generate_appeal(violation) → str (appeal text in Hebrew)
"""
from __future__ import annotations

import json
import logging
from typing import TypedDict

from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)
_client = OpenAI(api_key=OPENAI_API_KEY)


class ViolationData(TypedDict, total=False):
    case_number: str       # מספר תיק / דוח
    violation_date: str    # תאריך העבירה
    violation_time: str    # שעת העבירה
    location: str          # מיקום העבירה
    violation_type: str    # סוג העבירה
    fine_amount: str       # סכום הקנס
    issuing_authority: str # הרשות המוציאה
    owner_name: str        # שם בעל הרכב
    owner_id: str          # מספר זהות
    vehicle_number: str    # מספר רכב


_EXTRACT_SYSTEM = """
You are an expert assistant that extracts structured information from Israeli
traffic violation letters.  The text may be in Hebrew, English, or a mix.

Return ONLY a valid JSON object with these keys (use empty string when missing):
{
  "case_number": "",
  "violation_date": "",
  "violation_time": "",
  "location": "",
  "violation_type": "",
  "fine_amount": "",
  "issuing_authority": "",
  "owner_name": "",
  "owner_id": "",
  "vehicle_number": ""
}
""".strip()

_APPEAL_SYSTEM = """
You are a legal assistant specialising in Israeli traffic law.
Write a formal appeal letter in Hebrew addressed to the relevant authority.
The letter must:
- Be polite and professional
- Reference the case number, date, and type of violation
- Present standard legal grounds for the appeal (e.g. technical malfunction,
  lack of clarity in road signs, first offence, clean driving record)
- Request the cancellation or reduction of the fine
- Be signed as "מגיש הערר" (the appellant)

Return ONLY the letter text, no preamble.
""".strip()


def analyze_violation(ocr_text: str) -> ViolationData:
    """
    Send *ocr_text* to GPT-4o and extract structured violation data.

    Falls back to a dictionary of empty strings on any error.
    """
    logger.info("Sending OCR text to GPT-4o for analysis (%d chars)", len(ocr_text))
    try:
        response = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": ocr_text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data: ViolationData = json.loads(raw)
        logger.info("Extraction complete: %s", data)
        return data
    except Exception as exc:
        logger.error("GPT-4o extraction failed: %s", exc)
        return ViolationData(
            case_number="",
            violation_date="",
            violation_time="",
            location="",
            violation_type="",
            fine_amount="",
            issuing_authority="",
            owner_name="",
            owner_id="",
            vehicle_number="",
        )


def generate_appeal(violation: ViolationData) -> str:
    """
    Generate a formal Hebrew appeal letter based on *violation* data.
    """
    user_prompt = (
        f"מספר תיק: {violation.get('case_number', '')}\n"
        f"תאריך עבירה: {violation.get('violation_date', '')} "
        f"{violation.get('violation_time', '')}\n"
        f"מיקום: {violation.get('location', '')}\n"
        f"סוג עבירה: {violation.get('violation_type', '')}\n"
        f"סכום קנס: {violation.get('fine_amount', '')}\n"
        f"רשות מוציאה: {violation.get('issuing_authority', '')}\n"
        f"שם בעל הרכב: {violation.get('owner_name', '')}\n"
        f"ת.ז.: {violation.get('owner_id', '')}\n"
        f"מספר רכב: {violation.get('vehicle_number', '')}\n"
    )
    logger.info("Generating appeal letter for case %s", violation.get("case_number"))
    try:
        response = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _APPEAL_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        appeal = (response.choices[0].message.content or "").strip()
        logger.info("Appeal generated (%d chars)", len(appeal))
        return appeal
    except Exception as exc:
        logger.error("GPT-4o appeal generation failed: %s", exc)
        return ""

"""OpenAI integration for fine-data extraction and Hebrew appeal generation.

Two operations are exposed:
- :func:`extract_fine_details` – parses OCR text into a structured JSON object
  with per-field confidence scores.
- :func:`generate_appeal` – writes a formal Hebrew appeal letter using only
  confirmed facts from the extracted data.
"""
import json
import logging
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY, OPENAI_MODEL
from bot.fine_number import normalize_fine_number

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are an expert at reading Israeli traffic fine notices. "
    "Extract structured data and respond ONLY with valid JSON."
)

_EXTRACTION_PROMPT = """\
Extract the following fields from the OCR text of an Israeli traffic fine notice.

Return a JSON object where each field is an object with:
  "value"      : string (the extracted value, or null if not found)
  "confidence" : number between 0.0 and 1.0

Fields:
  fine_number      – fine / ticket reference number (מספר דוח)
  fine_date        – date of the violation (תאריך העבירה)
  fine_amount      – fine amount in NIS (סכום הקנס)
  violation        – violation description (תיאור העבירה)
  vehicle_plate    – vehicle license plate (מספר רכב)
  location         – location of the violation (מיקום)
  payment_deadline – payment deadline (מועד לתשלום)

OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"

Important rules:
- fine_number must be digits only (remove spaces, dashes, punctuation).
- If fine_number is uncertain, still return your best guess with lower confidence.
- Return ALL listed fields, even when value is null.

Respond ONLY with valid JSON. No explanation, no markdown fences."""

_FINE_NUMBER_ONLY_PROMPT = """\
Extract ONLY the fine/ticket number (מספר דוח) from OCR text.

Return JSON object with exactly:
{
  "fine_number": string or null,
  "confidence": number between 0.0 and 1.0
}

Rules:
- fine_number should contain digits only.
- Use both OCR blocks if provided.
- If not found, return {"fine_number": null, "confidence": 0.0}.

GENERAL OCR:
\"\"\"
{ocr_text}
\"\"\"

NUMERIC OCR:
\"\"\"
{numeric_ocr_text}
\"\"\"

Respond ONLY with valid JSON. No explanation, no markdown fences."""

# ---------------------------------------------------------------------------
# Appeal generation
# ---------------------------------------------------------------------------

_APPEAL_SYSTEM = (
    "You are a legal expert specializing in Israeli traffic law. "
    "Write formal Hebrew appeal letters for traffic fines. "
    "Respond ONLY in Hebrew."
)

_APPEAL_PROMPT = """\
כתוב מכתב ערר רשמי בעברית לרשות המוסמכת על דוח תנועה בישראל.

הנחיות:
1. כתוב אך ורק בעברית.
2. השתמש אך ורק בעובדות המאושרות שמופיעות בנתונים שלהלן.
3. הסגנון: רשמי ומנומס.
4. מבנה המכתב: תאריך, נמען, נושא, גוף, חתימה (מקום פנוי).
5. בקש ביטול או הפחתה של הקנס.

נתוני הדוח:
{fine_details}
{reason_section}
כתוב את מכתב הערר בלבד, ללא הסברים נוספים."""


async def extract_fine_details(ocr_text: str) -> Dict[str, Any]:
    """Call OpenAI to extract structured fine fields from *ocr_text*.

    Returns a dict such as::

        {
          "fine_number":      {"value": "12345678", "confidence": 0.95},
          "fine_date":        {"value": "01/01/2024", "confidence": 0.90},
          ...
        }
    """
    prompt = _EXTRACTION_PROMPT.format(ocr_text=ocr_text[:5000])
    try:
        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=800,
        )
        result: Dict[str, Any] = json.loads(response.choices[0].message.content)
        fine_number_data = result.get("fine_number")
        if isinstance(fine_number_data, dict):
            value = fine_number_data.get("value")
            if value:
                fine_number_data["value"] = normalize_fine_number(str(value), aggressive=True)
        logger.info("Fine details extracted (fields=%d)", len(result))
        return result
    except Exception as exc:
        logger.error("extract_fine_details failed: %s", exc)
        raise


async def extract_fine_number_only(ocr_text: str, numeric_ocr_text: str = "") -> Dict[str, Any]:
    """Extract only the fine number from OCR text blocks."""
    prompt = _FINE_NUMBER_ONLY_PROMPT.format(
        ocr_text=ocr_text[:5000],
        numeric_ocr_text=numeric_ocr_text[:3000],
    )
    try:
        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )
        result: Dict[str, Any] = json.loads(response.choices[0].message.content)
        normalized = normalize_fine_number(str(result.get("fine_number") or ""), aggressive=True)
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (float, int)):
            confidence = 0.0
        return {"fine_number": normalized or None, "confidence": float(confidence)}
    except Exception as exc:
        logger.error("extract_fine_number_only failed: %s", exc)
        return {"fine_number": None, "confidence": 0.0}


async def generate_appeal(
    fine_details: Dict[str, Any], appeal_reason: Optional[str] = None
) -> str:
    """Call OpenAI to generate a formal Hebrew appeal letter.

    Only fields with non-null values and either:
    - confidence ≥ 0.5, or
    - manually edited by the user (``manual=True``)
    are passed to the model so that unconfirmed data does not influence the letter.

    Parameters
    ----------
    fine_details:
        Structured fine data as returned by :func:`extract_fine_details` (or
        updated via the edit flow).
    appeal_reason:
        Optional reason string (Hebrew or any language) that explains why the
        user is appealing.  When provided it is included in the prompt so the
        model can reference it in the letter body.
    """
    # Filter to confirmed / manually edited fields only
    confirmed: Dict[str, str] = {}
    for field, data in fine_details.items():
        if isinstance(data, dict):
            value = data.get("value")
            confidence = data.get("confidence", 0.0)
            is_manual = data.get("manual", False)
            if value and (is_manual or confidence >= 0.5):
                confirmed[field] = value

    details_str = json.dumps(confirmed, ensure_ascii=False, indent=2)
    reason_section = (
        f"\nטעם הערר:\n{appeal_reason}\n" if appeal_reason else ""
    )
    prompt = _APPEAL_PROMPT.format(
        fine_details=details_str, reason_section=reason_section
    )

    try:
        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _APPEAL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        appeal_text = response.choices[0].message.content.strip()
        logger.info("Hebrew appeal letter generated (%d chars)", len(appeal_text))
        return appeal_text
    except Exception as exc:
        logger.error("generate_appeal failed: %s", exc)
        raise

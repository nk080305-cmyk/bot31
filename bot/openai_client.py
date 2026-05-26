"""OpenAI integration for fine-data extraction and Hebrew appeal generation.

Three operations are exposed:
- :func:`extract_fine_details` – parses OCR text into a structured JSON object
  with per-field confidence scores.
- :func:`extract_vision_fields` – reads an uploaded image directly and returns
  the fine notice number / licence plate when available.
- :func:`generate_appeal` – writes a formal Hebrew appeal letter using only
  confirmed facts from the extracted data.
"""
import base64
import json
import logging
import mimetypes
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY, OPENAI_MODEL
from bot.fine_number import (
    find_fine_number_candidates,
    normalize_fine_number,
    pick_best_fine_number,
)

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Private helpers for post-validation heuristics
# ---------------------------------------------------------------------------

_HEURISTIC_PLATE_CONFIDENCE = 0.65  # cap for OCR-heuristic plate guesses
_HEURISTIC_FINE_CONFIDENCE = 0.70   # cap for OCR-heuristic fine-number guesses

_PLATE_KEYWORDS = ["מספר רכב", "לוחית", "לוחית רישוי", "רכב", "vehicle", "plate"]
_FN_KEYWORDS = [
    "מספר דוח",
    "מס' דוח",
    "מס דוח",
    "דוח",
    "מספר הודעת תשלום קנס",
    "הודעת תשלום קנס",
    "fine",
    "ticket",
]
# Fine-number length → sort priority (prefer 8-digit, then 9, 10, 7; others last)
_FINE_LEN_RANK = {8: 5, 9: 4, 10: 3, 7: 2}
_TYPE2_FINE_LABEL_RE = re.compile(
    r"מספר[ \t\-./,:_]*הודעת[ \t\-./,:_]*תשלום[ \t\-./,:_]*קנס",
    re.IGNORECASE,
)


def _digits_only(text: str) -> str:
    """Strip every non-digit character from *text*."""
    return re.sub(r"\D", "", text)


def _context_digits_near_keywords(
    text: str,
    keywords: List[str],
    min_len: int,
    max_len: int,
    window: int = 260,
) -> List[str]:
    """Return unique digit strings of length [min_len, max_len] found near any keyword.

    Searches a sliding window around each keyword occurrence so that numeric
    sequences that appear in the same sentence or line as the keyword are
    preferred over random numbers elsewhere in the document.
    """
    if not text:
        return []
    # The pattern anchors on a leading and trailing digit (accounting for the
    # two mandatory digit anchors, the repetition range is [min_len-2, max_len-2]).
    _pat = re.compile(
        r"(?<!\d)(?:\d[ \t\-./,:_]?){%d,%d}\d(?!\d)" % (min_len - 1, max_len - 1)
    )
    sep_pattern = r"[ \t\-./,:_]*"
    seen: dict = {}
    for kw in keywords:
        kw_pattern = sep_pattern.join(re.escape(part) for part in kw.split())
        for m in re.finditer(kw_pattern, text, re.IGNORECASE | re.UNICODE):
            start = max(0, m.start() - window // 2)
            end = min(len(text), m.end() + window // 2)
            snippet = text[start:end]
            for dm in _pat.finditer(snippet):
                val = _digits_only(dm.group())
                if min_len <= len(val) <= max_len and val not in seen:
                    seen[val] = None
    return list(seen)


def _best_plate_candidate(numeric_ocr_text: str) -> Optional[str]:
    """Return the first 6-8 digit string from *numeric_ocr_text* as a plate guess."""
    if not numeric_ocr_text:
        return None
    for m in re.finditer(r"\d{6,8}", numeric_ocr_text):
        return m.group()
    return None


def _best_fine_candidate(
    numeric_ocr_text: str, plate: Optional[str] = None
) -> Optional[str]:
    """Return the best fine-number candidate from *numeric_ocr_text*.

    Uses the existing keyword-aware :func:`~bot.fine_number.find_fine_number_candidates`
    helper; the plate number is excluded to avoid confusing it with the fine number.
    """
    candidates = find_fine_number_candidates(numeric_ocr_text)
    if plate:
        candidates = [c for c in candidates if c != plate]
    return pick_best_fine_number(candidates) or None


def _is_type2_notice(ocr_text: str) -> bool:
    return bool(_TYPE2_FINE_LABEL_RE.search(ocr_text or ""))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are an expert at reading Israeli traffic fine notices. "
    "Extract structured data and respond ONLY with valid JSON."
)

_VISION_PROMPT = """\
Inspect the uploaded image of an Israeli traffic fine notice and extract ONLY:
- fine_notice_number
- license_plate

Return null for a field when it is not readable.

Rules:
- fine_notice_number must contain digits only after normalization.
- If the notice is labeled "מספר הודעת תשלום קנס", fine_notice_number must be 10-11 digits and must not be a 9-digit תעודת זהות.
- license_plate must be 7 or 8 digits after removing spaces, hyphens, or punctuation.
- Never invent values.

OCR anchor text (may be noisy, use as a hint only):
\"\"\"
{ocr_text}
\"\"\"
"""

_VISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fine_notice_number": {"type": ["string", "null"]},
        "license_plate": {"type": ["string", "null"]},
    },
    "required": ["fine_notice_number", "license_plate"],
}

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
- For notices with the label "מספר הודעת תשלום קנס", prefer a 10-11 digit
  fine_number and do not confuse it with 9-digit "תעודת זהות".
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
- If "מספר הודעת תשלום קנס" is present, fine_number should be 10-11 digits.
- Do not return 9-digit "תעודת זהות" as fine_number.
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


def normalize_license_plate(value: str | None) -> str:
    """Return a digits-only normalized vehicle plate."""
    return _digits_only(value or "")


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    return ""


def _validated_vision_fields(payload: Dict[str, Any], ocr_text: str = "") -> Dict[str, str]:
    if not isinstance(payload, dict):
        return {}

    is_type2_notice = _is_type2_notice(ocr_text)
    fine_notice_number = normalize_fine_number(
        str(payload.get("fine_notice_number") or ""), aggressive=True
    )
    license_plate = normalize_license_plate(payload.get("license_plate"))

    validated: Dict[str, str] = {}
    if fine_notice_number:
        if is_type2_notice:
            if len(fine_notice_number) in (10, 11):
                validated["fine_notice_number"] = fine_notice_number
        elif 6 <= len(fine_notice_number) <= 12:
            validated["fine_notice_number"] = fine_notice_number

    if license_plate and len(license_plate) in (7, 8):
        validated["license_plate"] = license_plate

    return validated


async def extract_vision_fields(image_path: str, ocr_text: str = "") -> Dict[str, str]:
    """Extract fine notice number and licence plate directly from an image."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in {"image/jpeg", "image/png"}:
        return {}

    with open(image_path, "rb") as fh:
        image_data = fh.read()
    image_url = (
        f"data:{mime_type};base64,{base64.b64encode(image_data).decode('ascii')}"
    )

    prompt = _VISION_PROMPT.format(ocr_text=(ocr_text or "")[:3000])
    try:
        response = await _client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url, "detail": "high"},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "vision_fine_notice_extraction",
                    "schema": _VISION_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0.0,
            max_output_tokens=200,
        )
        result = json.loads(_extract_response_text(response))
    except json.JSONDecodeError:
        logger.warning("Vision extraction returned invalid JSON")
        return {}
    except Exception as exc:
        logger.warning("Vision extraction failed: %s", exc)
        return {}

    return _validated_vision_fields(result, ocr_text)


async def extract_fine_details(
    ocr_text: str, numeric_ocr_text: str = ""
) -> Dict[str, Any]:
    """Call OpenAI to extract structured fine fields from *ocr_text*.

    Parameters
    ----------
    ocr_text:
        General OCR text (Hebrew + English Tesseract pass).
    numeric_ocr_text:
        Optional numeric-focused OCR text used for post-extraction heuristics
        when the model output is missing or implausible.

    Returns a dict such as::

        {
          "fine_number":      {"value": "12345678", "confidence": 0.95},
          "fine_date":        {"value": "01/01/2024", "confidence": 0.90},
          ...
        }
    """
    prompt = _EXTRACTION_PROMPT.format(ocr_text=ocr_text[:5000])
    try:
        is_type2_notice = _is_type2_notice(ocr_text)
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

        # --- Post-validation: heuristic fallbacks for vehicle_plate / fine_number ---
        try:
            plate_data = result.get("vehicle_plate")
            plate_val2: Optional[str] = (
                _digits_only(str(plate_data.get("value") or ""))
                if isinstance(plate_data, dict)
                else ""
            ) or None

            # Context-aware candidates from the general OCR text
            ctx_plate_cands = _context_digits_near_keywords(
                ocr_text, _PLATE_KEYWORDS, 5, 8, window=200
            )
            ctx_plate_best: Optional[str] = ctx_plate_cands[0] if ctx_plate_cands else None

            ctx_fine_cands = _context_digits_near_keywords(
                ocr_text,
                _FN_KEYWORDS,
                10 if is_type2_notice else 7,
                11 if is_type2_notice else 13,
                window=260,
            )
            if ctx_fine_cands:
                ctx_fine_cands.sort(
                    key=lambda s: (_FINE_LEN_RANK.get(len(s), 1), len(s)),
                    reverse=True,
                )
            ctx_fine_best: Optional[str] = ctx_fine_cands[0] if ctx_fine_cands else None

            # Fix plate if invalid length
            if not plate_val2 or not (6 <= len(plate_val2) <= 8):
                best_plate = ctx_plate_best or _best_plate_candidate(numeric_ocr_text)
                if best_plate:
                    old_conf = (
                        (result.get("vehicle_plate") or {}).get(
                            "confidence", _HEURISTIC_PLATE_CONFIDENCE
                        )
                        if isinstance(result.get("vehicle_plate"), dict)
                        else _HEURISTIC_PLATE_CONFIDENCE
                    )
                    result["vehicle_plate"] = {
                        "value": best_plate,
                        "confidence": float(min(_HEURISTIC_PLATE_CONFIDENCE, old_conf)),
                    }
                    plate_val2 = best_plate

            # Fix fine_number if too short / clearly wrong
            if isinstance(result.get("fine_number"), dict):
                fn_val = _digits_only(str(result["fine_number"].get("value") or ""))
                if fn_val:
                    result["fine_number"]["value"] = fn_val

                # Reuse fn_val – it already reflects the (possibly updated) value above
                fn_val2: Optional[str] = fn_val or None
                if (
                    (not fn_val2)
                    or (len(fn_val2) < 7)
                    or (is_type2_notice and len(fn_val2) not in (10, 11))
                    or (plate_val2 and fn_val2 == plate_val2)
                    or (plate_val2 and len(fn_val2) <= len(plate_val2))
                ):
                    best_fn = ctx_fine_best or _best_fine_candidate(
                        numeric_ocr_text, plate=plate_val2
                    )
                    if is_type2_notice and best_fn and len(best_fn) not in (10, 11):
                        best_fn = None
                    if best_fn:
                        result["fine_number"]["value"] = best_fn
                        old_c = result["fine_number"].get("confidence", 0.5)
                        result["fine_number"]["confidence"] = float(
                            min(old_c, _HEURISTIC_FINE_CONFIDENCE)
                        )
        except Exception as _post_exc:
            logger.warning("Post-validation heuristics failed: %s", _post_exc)

        logger.info("Fine details extracted (fields=%d)", len(result))
        return result
    except Exception as exc:
        logger.error("extract_fine_details failed: %s", exc)
        raise


async def extract_fine_number_only(ocr_text: str, numeric_ocr_text: str = "") -> Dict[str, Any]:
    """Extract only the fine number from OCR text blocks.

    Parameters
    ----------
    ocr_text:
        General OCR text (heb+eng pass).
    numeric_ocr_text:
        Optional numeric-focused OCR text for difficult scans.

    Returns
    -------
    dict
        ``{"fine_number": str | None, "confidence": float}``.
    """
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

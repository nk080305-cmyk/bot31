"""OCR pipeline for traffic fine documents.

Steps
-----
1. If input is a PDF → convert first page to a 300-dpi PNG via pdf2image.
2. Preprocess the image with OpenCV:
   - Greyscale
   - Denoising (fastNlMeansDenoising)
   - Adaptive thresholding
   - Deskewing (if rotation > 0.5°)
3. Run Tesseract with language ``heb+eng`` and PSM modes 6, 4, and 11.
4. Score each result with a heuristic function and keep the best general OCR text.
5. Run an extra numeric-only OCR pass for better fine-number recovery.
"""
import logging
import os
import re
import tempfile
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

TESSERACT_LANG = "heb+eng"
TESSERACT_PSM_MODES = [6, 4, 11]
TESSERACT_NUMERIC_PSM_MODES = [7, 6, 11]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
OCR_MULTI_PREPROCESS = os.getenv("OCR_MULTI_PREPROCESS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

PLATE_KEYWORDS = ["רכב", "מספר רכב"]
FINE_KEYWORDS = ["דוח", "מספר דוח", "קנס"]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _deskew(image: np.ndarray) -> np.ndarray:
    """Rotate the image to correct small skew angles."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Return a preprocessed greyscale image suitable for Tesseract."""
    # Convert to greyscale if colour
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image.copy()

    # Denoise
    denoised = cv2.fastNlMeansDenoising(grey, h=10)

    # Adaptive threshold → clean binary image
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )

    # Deskew
    deskewed = _deskew(binary)

    return deskewed


def preprocess_variants(image: np.ndarray) -> List[np.ndarray]:
    """Return OCR preprocess variants: equalized gray + 2 binary variants."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh_global = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)[1]
    thresh_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return [gray, thresh_global, thresh_adaptive]


def preprocess_numeric_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for numeric OCR (fine number recovery)."""
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image.copy()

    upscaled = cv2.resize(grey, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 6
    )
    return _deskew(binary)


# ---------------------------------------------------------------------------
# Scoring heuristic
# ---------------------------------------------------------------------------

_ALLOWED_PUNCTUATION = set(' \n\t.,:-/()₪"\'' + "—–")


def score_ocr_result(text: str) -> float:
    """Return a quality score for an OCR result (higher is better)."""
    stripped = text.strip()
    if len(stripped) < 10:
        return 0.0

    score = 0.0

    # Reward longer meaningful output
    score += min(len(stripped) / 80.0, 15.0)

    # Reward Hebrew characters (common in Israeli traffic fines)
    score += sum(0.4 for c in stripped if "\u05d0" <= c <= "\u05ea")

    # Reward digits (amounts, dates, fine numbers)
    score += sum(0.2 for c in stripped if c.isdigit())

    # Penalise excessive non-alphanumeric, non-Hebrew, non-allowed characters
    noise = sum(
        1
        for c in stripped
        if not c.isalnum() and c not in _ALLOWED_PUNCTUATION and not ("\u05d0" <= c <= "\u05ea")
    )
    score -= noise * 0.15

    return max(score, 0.0)


# ---------------------------------------------------------------------------
# Tesseract runner
# ---------------------------------------------------------------------------

def _run_tesseract_on_file(image_path: str) -> str:
    """Run Tesseract with multiple PSM modes and return the best result."""
    results: List[Tuple[float, str]] = []
    for psm in TESSERACT_PSM_MODES:
        config = f"--psm {psm} --oem 3"
        try:
            text = pytesseract.image_to_string(image_path, lang=TESSERACT_LANG, config=config)
            sc = score_ocr_result(text)
            results.append((sc, text))
            logger.debug("PSM %d → score=%.2f, chars=%d", psm, sc, len(text))
        except Exception as exc:
            logger.warning("Tesseract PSM %d failed: %s", psm, exc)

    if not results:
        return ""

    best_score, best_text = max(results, key=lambda x: x[0])
    logger.info("OCR best score=%.2f from %d PSM configs", best_score, len(results))
    return best_text.strip()


def _run_tesseract_on_variants(images: List[np.ndarray]) -> str:
    """Run scored OCR over image variants × PSM modes and return best text."""
    results: List[Tuple[float, str]] = []
    for idx, image in enumerate(images):
        for psm in TESSERACT_PSM_MODES:
            config = f"--psm {psm} --oem 3"
            try:
                text = pytesseract.image_to_string(image, lang=TESSERACT_LANG, config=config)
                sc = score_ocr_result(text)
                results.append((sc, text))
                logger.debug(
                    "Variant %d PSM %d → score=%.2f, chars=%d", idx, psm, sc, len(text)
                )
            except Exception as exc:
                logger.warning("Tesseract variant %d PSM %d failed: %s", idx, psm, exc)
    if not results:
        return ""
    best_score, best_text = max(results, key=lambda x: x[0])
    logger.info("OCR best score=%.2f from %d variant×PSM configs", best_score, len(results))
    return best_text.strip()


def _run_tesseract_numeric_on_file(image_path: str) -> str:
    """Run numeric-focused Tesseract pass (digits and separators only)."""
    results: List[Tuple[int, str]] = []
    config_suffix = "-c tessedit_char_whitelist=0123456789/.-: "
    for psm in TESSERACT_NUMERIC_PSM_MODES:
        config = f"--psm {psm} --oem 3 {config_suffix}"
        try:
            text = pytesseract.image_to_string(image_path, lang="eng", config=config)
            digit_count = sum(1 for ch in text if ch.isdigit())
            results.append((digit_count, text))
            logger.debug("Numeric OCR PSM %d → digits=%d, chars=%d", psm, digit_count, len(text))
        except Exception as exc:
            logger.warning("Numeric OCR PSM %d failed: %s", psm, exc)

    if not results:
        return ""
    _, best_text = max(results, key=lambda x: x[0])
    return best_text.strip()


def _run_tesseract_numeric_on_variants(images: List[np.ndarray]) -> str:
    """Run numeric OCR over image variants × numeric PSM modes and return best text."""
    results: List[Tuple[int, str]] = []
    config_suffix = "-c tessedit_char_whitelist=0123456789"
    for idx, image in enumerate(images):
        for psm in TESSERACT_NUMERIC_PSM_MODES:
            config = f"--psm {psm} --oem 3 {config_suffix}"
            try:
                text = pytesseract.image_to_string(image, lang="eng", config=config)
                digit_count = sum(1 for ch in text if ch.isdigit())
                results.append((digit_count, text))
                logger.debug(
                    "Numeric variant %d PSM %d → digits=%d, chars=%d",
                    idx,
                    psm,
                    digit_count,
                    len(text),
                )
            except Exception as exc:
                logger.warning("Numeric OCR variant %d PSM %d failed: %s", idx, psm, exc)
    if not results:
        return ""
    _, best_text = max(results, key=lambda x: x[0])
    return best_text.strip()


def run_ocr_multi(image: np.ndarray) -> Tuple[str, str]:
    """Run fixed PSM6 OCR over preprocess variants and join outputs."""
    texts: List[str] = []
    nums: List[str] = []
    for variant in preprocess_variants(image):
        texts.append(
            pytesseract.image_to_string(variant, lang="heb+eng", config="--oem 3 --psm 6")
        )
        nums.append(
            pytesseract.image_to_string(
                variant,
                lang="eng",
                config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
            )
        )
    return "\n".join(t.strip() for t in texts if t.strip()), "\n".join(
        n.strip() for n in nums if n.strip()
    )


def _context_score(text: str, number: str, keywords: List[str], window: int = 150) -> int:
    score = 0
    if not number:
        return score
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            if number in text[start:end]:
                score += 2
    return score


def _candidate_score(
    candidate: str, counts: Counter, kind: str, text: str
) -> Tuple[int, bool, int, int]:
    if kind == "plate":
        return (
            counts[candidate],
            len(candidate) in (7, 8),
            _context_score(text, candidate, PLATE_KEYWORDS),
            -abs(len(candidate) - 8),
        )
    return (
        counts[candidate],
        len(candidate) in (8, 9),
        _context_score(text, candidate, FINE_KEYWORDS),
        -abs(len(candidate) - 8),
    )


def extract_plate_and_fine_candidates(ocr_text: str, numeric_text: str) -> Dict[str, Any]:
    """Extract plate/fine heuristics from OCR text blocks."""
    full_text = f"{ocr_text}\n{numeric_text}"
    raw = re.findall(r"\d[\d\s\-./,:]{4,16}\d", full_text)
    cleaned = ["".join(ch for ch in token if ch.isdigit()) for token in raw]
    cleaned = [n for n in cleaned if 6 <= len(n) <= 12 and len(set(n)) > 2]
    if not cleaned:
        return {
            "plate": None,
            "fine": None,
            "plate_confident": False,
            "fine_confident": False,
        }

    counts: Counter = Counter(cleaned)
    plate_candidates = [n for n in cleaned if len(n) in (7, 8)]
    best_plate = (
        max(plate_candidates, key=lambda n: _candidate_score(n, counts, "plate", ocr_text))
        if plate_candidates
        else None
    )

    fine_candidates = [n for n in cleaned if 7 <= len(n) <= 10 and n != best_plate]
    best_fine = (
        max(fine_candidates, key=lambda n: _candidate_score(n, counts, "fine", ocr_text))
        if fine_candidates
        else None
    )

    plate_ctx = _context_score(ocr_text, best_plate or "", PLATE_KEYWORDS) if best_plate else 0
    fine_ctx = _context_score(ocr_text, best_fine or "", FINE_KEYWORDS) if best_fine else 0
    plate_confident = bool(best_plate and counts[best_plate] >= 2 and plate_ctx >= 2)
    fine_confident = bool(best_fine and counts[best_fine] >= 2 and fine_ctx >= 2)

    return {
        "plate": best_plate,
        "fine": best_fine,
        "plate_confident": plate_confident,
        "fine_confident": fine_confident,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr_image(image_path: str) -> str:
    """OCR a raster image file (JPG / PNG)."""
    text, _ = ocr_image_with_numeric(image_path)
    return text


def ocr_image_with_numeric(image_path: str) -> Tuple[str, str]:
    """OCR a raster image and return ``(general_text, numeric_text)``.

    ``general_text`` is produced from the standard heb+eng OCR pipeline.
    ``numeric_text`` is produced from a numeric-focused OCR pass for better
    fine-number recovery.

    Returns
    -------
    Tuple[str, str]
        ``(general_text, numeric_text)``. Any failed OCR pass returns an empty
        string for that element.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    if OCR_MULTI_PREPROCESS:
        variants = preprocess_variants(img)
        best_general = _run_tesseract_on_variants(variants)
        best_numeric = _run_tesseract_numeric_on_variants(variants)
        multi_general, multi_numeric = run_ocr_multi(img)
        general_text = "\n".join(part for part in (best_general, multi_general) if part).strip()
        numeric_text = "\n".join(part for part in (best_numeric, multi_numeric) if part).strip()
        return general_text, numeric_text

    preprocessed = preprocess_image(img)
    numeric_preprocessed = preprocess_numeric_image(img)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_general:
        general_path = tmp_general.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_numeric:
        numeric_path = tmp_numeric.name
    try:
        cv2.imwrite(general_path, preprocessed)
        cv2.imwrite(numeric_path, numeric_preprocessed)
        return _run_tesseract_on_file(general_path), _run_tesseract_numeric_on_file(numeric_path)
    finally:
        for path in (general_path, numeric_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def ocr_pdf(pdf_path: str) -> str:
    """Convert the first page of a PDF to PNG and OCR it."""
    text, _ = ocr_pdf_with_numeric(pdf_path)
    return text


def ocr_pdf_with_numeric(pdf_path: str) -> Tuple[str, str]:
    """OCR the first PDF page and return ``(general_text, numeric_text)``.

    The PDF page is converted to PNG and then processed by both the standard
    OCR pipeline and a numeric-focused OCR pipeline.

    Returns
    -------
    Tuple[str, str]
        ``(general_text, numeric_text)``.
    """
    from pdf2image import convert_from_path  # imported here to keep startup light

    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
    if not pages:
        raise ValueError("PDF produced no pages")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pages[0].save(tmp_path, "PNG")
        return ocr_image_with_numeric(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_text(file_path: str) -> str:
    """Dispatch to the correct OCR function based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return ocr_pdf(file_path)
    if ext in {".jpg", ".jpeg", ".png"}:
        return ocr_image(file_path)
    raise ValueError(f"Unsupported file extension for OCR: {ext}")


def extract_text_with_numeric(file_path: str) -> Tuple[str, str]:
    """Dispatch OCR and return (general_text, numeric_text)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return ocr_pdf_with_numeric(file_path)
    if ext in {".jpg", ".jpeg", ".png"}:
        return ocr_image_with_numeric(file_path)
    raise ValueError(f"Unsupported file extension for OCR: {ext}")

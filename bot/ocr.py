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
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

TESSERACT_LANG = "heb+eng"
TESSERACT_PSM_MODES = [6, 4, 11]
TESSERACT_NUMERIC_PSM_MODES = [7, 6, 11]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


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

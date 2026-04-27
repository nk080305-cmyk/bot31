"""
OCR extraction module.

Accepts a local file path (PDF or image) and returns the raw text using
Tesseract with Hebrew + English language support.

Image pre-processing with OpenCV improves OCR accuracy on scanned documents.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply standard image clean-up steps that improve OCR quality."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding produces a clean binary image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    # Mild sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(binary, -1, kernel)
    return sharpened


def _images_from_pdf(pdf_path: str) -> list[np.ndarray]:
    """Convert every page of a PDF to an OpenCV image array."""
    from pdf2image import convert_from_path  # lazy import

    pil_images = convert_from_path(pdf_path, dpi=300)
    result: list[np.ndarray] = []
    for pil_img in pil_images:
        arr = np.array(pil_img.convert("RGB"))
        result.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return result


def extract_text(file_path: str) -> str:
    """
    Extract raw text from *file_path* (PDF or image).

    Returns the concatenated OCR text, or an empty string when nothing
    could be recognised.
    """
    import pytesseract  # lazy import

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    images: list[np.ndarray] = []

    if suffix == ".pdf":
        logger.info("Converting PDF to images: %s", file_path)
        images = _images_from_pdf(file_path)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Cannot read image file: {file_path}")
        images = [img]

    pages_text: list[str] = []
    for i, img in enumerate(images):
        preprocessed = _preprocess_image(img)
        text = pytesseract.image_to_string(
            preprocessed,
            lang="heb+eng",
            config="--oem 1 --psm 6",
        )
        logger.debug("Page %d OCR output (%d chars)", i + 1, len(text))
        pages_text.append(text)

    full_text = "\n\n".join(pages_text).strip()
    logger.info("OCR complete, total chars: %d", len(full_text))
    return full_text

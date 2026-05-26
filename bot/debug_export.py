"""Debug artifact export for OCR/LLM diagnostics."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np

from bot.config import DEBUG_DIR, OCR_DEBUG_MAX_CASES

logger = logging.getLogger(__name__)

_GROUP_RE = re.compile(
    r"^(?P<case>[^_]+)_(?P<ts>\d{8}_\d{6}Z)_.+$"
)
_debug_context: contextvars.ContextVar[Optional["DebugContext"]] = contextvars.ContextVar(
    "ocr_debug_context",
    default=None,
)


@dataclass(frozen=True)
class DebugContext:
    case_id: str
    timestamp: str


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def is_enabled() -> bool:
    return _env_flag("OCR_DEBUG", default=False) or _env_flag("DEBUG", default=False)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def set_context(case_id: str, timestamp: Optional[str] = None) -> contextvars.Token:
    return _debug_context.set(DebugContext(case_id=case_id, timestamp=timestamp or utc_timestamp()))


def reset_context(token: contextvars.Token) -> None:
    _debug_context.reset(token)


def current_context() -> Optional[DebugContext]:
    return _debug_context.get()


def _ensure_debug_dir() -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    try:
        os.chmod(DEBUG_DIR, 0o775)
    except OSError:
        pass


def _artifact_path(case_id: str, timestamp: str, suffix: str) -> str:
    return os.path.join(DEBUG_DIR, f"{case_id}_{timestamp}_{suffix}")


def _prune_old_cases() -> None:
    if OCR_DEBUG_MAX_CASES <= 0:
        return
    groups: dict[tuple[str, str], list[str]] = {}
    for entry in os.listdir(DEBUG_DIR):
        match = _GROUP_RE.match(entry)
        if not match:
            continue
        key = (match.group("case"), match.group("ts"))
        groups.setdefault(key, []).append(os.path.join(DEBUG_DIR, entry))

    if len(groups) <= OCR_DEBUG_MAX_CASES:
        return

    ordered = sorted(groups.keys(), key=lambda item: item[1], reverse=True)
    for key in ordered[OCR_DEBUG_MAX_CASES:]:
        for path in groups[key]:
            try:
                os.unlink(path)
            except OSError:
                logger.warning("Failed removing old debug artifact: %s", path)


def write_text(suffix: str, content: str, *, case_id: Optional[str] = None, timestamp: Optional[str] = None) -> Optional[str]:
    if not is_enabled():
        return None
    ctx = current_context()
    actual_case_id = case_id or (ctx.case_id if ctx else None)
    actual_ts = timestamp or (ctx.timestamp if ctx else None)
    if not actual_case_id or not actual_ts:
        return None
    _ensure_debug_dir()
    path = _artifact_path(actual_case_id, actual_ts, suffix)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    _prune_old_cases()
    return path


def write_json(suffix: str, payload: Any, *, case_id: Optional[str] = None, timestamp: Optional[str] = None) -> Optional[str]:
    return write_text(
        suffix,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        case_id=case_id,
        timestamp=timestamp,
    )


def copy_file(source_path: str, suffix: str, *, case_id: Optional[str] = None, timestamp: Optional[str] = None) -> Optional[str]:
    if not is_enabled():
        return None
    ctx = current_context()
    actual_case_id = case_id or (ctx.case_id if ctx else None)
    actual_ts = timestamp or (ctx.timestamp if ctx else None)
    if not actual_case_id or not actual_ts:
        return None
    _ensure_debug_dir()
    destination = _artifact_path(actual_case_id, actual_ts, suffix)
    shutil.copy2(source_path, destination)
    _prune_old_cases()
    return destination


def write_image(suffix: str, image: np.ndarray, *, case_id: Optional[str] = None, timestamp: Optional[str] = None) -> Optional[str]:
    if not is_enabled():
        return None
    ctx = current_context()
    actual_case_id = case_id or (ctx.case_id if ctx else None)
    actual_ts = timestamp or (ctx.timestamp if ctx else None)
    if not actual_case_id or not actual_ts:
        return None
    _ensure_debug_dir()
    path = _artifact_path(actual_case_id, actual_ts, suffix)
    cv2.imwrite(path, image)
    _prune_old_cases()
    return path

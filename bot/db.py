"""Async SQLite persistence layer.

Schema
------
users       – per-user settings (language preference)
cases       – encrypted case blobs with 7-day TTL
audit_logs  – encrypted audit records (may contain PII)
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from bot.config import DATA_TTL_DAYS, DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    user_id   INTEGER PRIMARY KEY,
    language  TEXT    NOT NULL DEFAULT 'ru',
    created_at TEXT   NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cases (
    id                   TEXT PRIMARY KEY,
    user_id              INTEGER NOT NULL,
    encrypted_data       TEXT NOT NULL,
    encrypted_file_path  TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at           TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_cases_user_id  ON cases(user_id);
CREATE INDEX IF NOT EXISTS idx_cases_expires  ON cases(expires_at);

CREATE TABLE IF NOT EXISTS audit_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type        TEXT NOT NULL,
    encrypted_payload TEXT NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def init_db() -> None:
    """Create tables if they don't exist yet."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_DDL)
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

async def get_or_create_user(user_id: int, default_language: str = "ru") -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_id, language, created_at FROM users WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
        if row:
            return dict(row)
        await db.execute(
            "INSERT OR IGNORE INTO users (user_id, language) VALUES (?, ?)",
            (user_id, default_language),
        )
        await db.commit()
        return {"user_id": user_id, "language": default_language}


async def update_user_language(user_id: int, language: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET language = ? WHERE user_id = ?", (language, user_id)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

async def save_case(
    user_id: int,
    case_data: Dict[str, Any],
    file_path: Optional[str] = None,
) -> str:
    """Persist an encrypted case and return its UUID.

    Parameters
    ----------
    file_path:
        Plaintext path to the encrypted file on disk (e.g. ``/data/cases/abc.enc``).
        The path is encrypted before being stored in the database.
    """
    from bot.encryption import encrypt  # local import avoids circular dependency at startup

    case_id = str(uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(days=DATA_TTL_DAYS)).isoformat()
    enc_data = encrypt(json.dumps(case_data, ensure_ascii=False))
    enc_file_path = encrypt(file_path) if file_path else None

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO cases (id, user_id, encrypted_data, encrypted_file_path, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (case_id, user_id, enc_data, enc_file_path, expires_at),
        )
        await db.commit()

    logger.info("Case %s saved for user_id=%s", case_id, user_id)
    return case_id


async def get_latest_case(user_id: int) -> Optional[Dict[str, Any]]:
    """Return the most recent non-expired case for a user, with decrypted data."""
    from bot.encryption import decrypt

    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, user_id, encrypted_data, encrypted_file_path, created_at, expires_at
            FROM cases
            WHERE user_id = ? AND expires_at > ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, now),
        ) as cur:
            row = await cur.fetchone()

    if not row:
        return None

    result = dict(row)
    result["data"] = json.loads(decrypt(result["encrypted_data"]))
    if result.get("encrypted_file_path"):
        result["file_path"] = decrypt(result["encrypted_file_path"])
    return result


async def delete_user_data(user_id: int) -> Tuple[int, List[str]]:
    """
    Delete all cases for *user_id*.

    Returns
    -------
    (deleted_count, file_paths)
        file_paths – plaintext paths of uploaded files that should also be removed from disk.
    """
    from bot.encryption import decrypt

    async with aiosqlite.connect(DB_PATH) as db:
        # Collect file paths before deletion so callers can purge them from disk
        async with db.execute(
            "SELECT encrypted_file_path FROM cases WHERE user_id = ? AND encrypted_file_path IS NOT NULL",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()

        file_paths: List[str] = []
        for (enc_path,) in rows:
            try:
                file_paths.append(decrypt(enc_path))
            except Exception:
                logger.warning("Could not decrypt file path during deletion (user_id=%s)", user_id)

        cur2 = await db.execute("DELETE FROM cases WHERE user_id = ?", (user_id,))
        deleted = cur2.rowcount
        await db.commit()

    logger.info("Deleted %d cases for user_id=%s", deleted, user_id)
    return deleted, file_paths


async def cleanup_expired() -> List[str]:
    """
    Remove all expired cases from the database.

    Returns
    -------
    List of plaintext file paths that should be purged from disk.
    """
    from bot.encryption import decrypt

    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT encrypted_file_path FROM cases WHERE expires_at <= ? AND encrypted_file_path IS NOT NULL",
            (now,),
        ) as cur:
            rows = await cur.fetchall()

        file_paths: List[str] = []
        for (enc_path,) in rows:
            try:
                file_paths.append(decrypt(enc_path))
            except Exception:
                logger.warning("Could not decrypt file path during expiry cleanup")

        await db.execute("DELETE FROM cases WHERE expires_at <= ?", (now,))
        await db.commit()

    if file_paths:
        logger.info("Cleaned up %d expired cases", len(file_paths))
    return file_paths


# ---------------------------------------------------------------------------
# Audit logs
# ---------------------------------------------------------------------------

async def add_audit_log(event_type: str, payload: Dict[str, Any]) -> None:
    """Write an encrypted audit log entry (payload may contain PII)."""
    from bot.encryption import encrypt

    enc_payload = encrypt(json.dumps(payload, ensure_ascii=False, default=str))
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO audit_logs (event_type, encrypted_payload) VALUES (?, ?)",
            (event_type, enc_payload),
        )
        await db.commit()

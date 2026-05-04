"""Symmetric encryption/decryption using Fernet (AES-128-CBC + HMAC-SHA256).

The encryption key is derived via PBKDF2 from:
  - ENCRYPTION_KEY env-var when set, or
  - TELEGRAM_BOT_TOKEN as a fallback secret.

A fixed application salt is used so the derived key is stable across restarts.
"""
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _build_fernet() -> Fernet:
    from bot.config import ENCRYPTION_KEY, TELEGRAM_BOT_TOKEN  # imported lazily to avoid circular imports at module load

    secret = ENCRYPTION_KEY or TELEGRAM_BOT_TOKEN
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"bot31_traffic_fines_v1",
        iterations=100_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return Fernet(key)


_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = _build_fernet()
    return _fernet


def encrypt(plaintext: str) -> str:
    """Encrypt a UTF-8 string and return a base64-URL token."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a token produced by :func:`encrypt`."""
    return _get_fernet().decrypt(token.encode()).decode()


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes."""
    return _get_fernet().encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt raw bytes produced by :func:`encrypt_bytes`."""
    return _get_fernet().decrypt(data)

"""AES-GCM-256 token encryption with a file-backed master key.

Inspired by Grafana's secureJsonData + secret_key pattern. Tokens are
never returned through the API — they live encrypted on disk and are
decrypted only at the point of use (RunPod API call, HF Hub upload,
MLflow authenticated request).

Storage layout per encrypted token::

    ~/.ryotenkai/providers/<id>/token.enc
    ~/.ryotenkai/integrations/<id>/token.enc

    File bytes: nonce(12) || ciphertext || tag(16)  (all handled by AES-GCM)

Master key::

    ~/.ryotenkai/.secret.key      # 32 random bytes, mode 0600
    RYOTENKAI_SECRET_KEY env var  # base64(32 bytes); wins over file when set

Threat model:
- File-system read access == token read access (mode 0600 mitigates
  shared-machine exposure).
- Backups can leak ``.secret.key`` + all ``token.enc`` together and
  recover plaintext — document this.
- Envelope encryption (per-token DEK wrapped by a KEK) is intentionally
  left for v2; the file layout reserves space for it.
"""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.utils.atomic_fs import atomic_write_text

MASTER_KEY_FILENAME = ".secret.key"
MASTER_KEY_BYTES = 32  # AES-256
NONCE_BYTES = 12  # AES-GCM recommended


class TokenCryptoError(RuntimeError):
    """Raised for recoverable crypto/storage errors."""


def default_root() -> Path:
    return Path.home() / ".ryotenkai"


def master_key_path(root: Path | None = None) -> Path:
    return (root or default_root()) / MASTER_KEY_FILENAME


def _load_master_key_from_env() -> bytes | None:
    raw = os.environ.get("RYOTENKAI_SECRET_KEY")
    if not raw:
        return None
    try:
        key = base64.b64decode(raw)
    except (ValueError, TypeError) as exc:
        raise TokenCryptoError(
            "RYOTENKAI_SECRET_KEY is set but is not valid base64"
        ) from exc
    if len(key) != MASTER_KEY_BYTES:
        raise TokenCryptoError(
            f"RYOTENKAI_SECRET_KEY must decode to {MASTER_KEY_BYTES} bytes, got {len(key)}"
        )
    return key


def load_or_create_master_key(root: Path | None = None) -> bytes:
    """Return the master key, generating + persisting it on first call.

    Precedence: ``RYOTENKAI_SECRET_KEY`` env var > on-disk file. When the
    env var is set the on-disk file is ignored (allows CI/prod to supply
    the key from a KMS without touching the filesystem).
    """
    env_key = _load_master_key_from_env()
    if env_key is not None:
        return env_key

    path = master_key_path(root)
    if path.is_file():
        raw = path.read_bytes()
        # Files we write are base64-text (trailing newline). Accept raw
        # 32-byte blobs too so a hand-crafted key works.
        if len(raw) == MASTER_KEY_BYTES:
            return raw
        try:
            decoded = base64.b64decode(raw.strip())
        except (ValueError, TypeError) as exc:
            raise TokenCryptoError(f"master key at {path} is corrupt") from exc
        if len(decoded) != MASTER_KEY_BYTES:
            raise TokenCryptoError(
                f"master key at {path} decodes to {len(decoded)} bytes, expected {MASTER_KEY_BYTES}"
            )
        return decoded

    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(MASTER_KEY_BYTES)
    atomic_write_text(path, base64.b64encode(key).decode("ascii") + "\n")
    path.chmod(0o600)
    return key


class TokenCrypto:
    """AES-GCM-256 wrapper bound to the master key."""

    def __init__(self, key: bytes | None = None, *, root: Path | None = None):
        if key is None:
            key = load_or_create_master_key(root)
            # load_or_create_master_key always returns exactly 32 bytes.
        if len(key) != MASTER_KEY_BYTES:
            raise TokenCryptoError(f"master key must be {MASTER_KEY_BYTES} bytes")
        self._aes = AESGCM(key)

    def encrypt(self, plaintext: str) -> bytes:
        if not isinstance(plaintext, str):
            raise TypeError("plaintext must be str")
        nonce = secrets.token_bytes(NONCE_BYTES)
        ciphertext = self._aes.encrypt(nonce, plaintext.encode("utf-8"), None)
        return nonce + ciphertext

    def decrypt(self, blob: bytes) -> str:
        if len(blob) < NONCE_BYTES + 16:
            raise TokenCryptoError("token blob too short")
        nonce, ciphertext = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        try:
            plaintext = self._aes.decrypt(nonce, ciphertext, None)
        except Exception as exc:  # cryptography raises InvalidTag
            raise TokenCryptoError(
                "failed to decrypt token (master key changed or file corrupt)"
            ) from exc
        return plaintext.decode("utf-8")


def write_token_file(path: Path, plaintext: str, crypto: TokenCrypto) -> None:
    """Atomically write an encrypted token blob at mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = crypto.encrypt(plaintext)
    # Binary write — atomic_write_text won't fit; do our own atomic dance.
    import tempfile

    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=path.parent
    ) as tmp:
        tmp.write(blob)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).chmod(0o600)
    Path(tmp_name).replace(path)


def read_token_file(path: Path, crypto: TokenCrypto) -> str | None:
    """Return the plaintext token, or ``None`` when the file is absent."""
    if not path.is_file():
        return None
    return crypto.decrypt(path.read_bytes())


def delete_token_file(path: Path) -> bool:
    if path.is_file():
        path.unlink()
        return True
    return False


__all__ = [
    "MASTER_KEY_BYTES",
    "MASTER_KEY_FILENAME",
    "NONCE_BYTES",
    "TokenCrypto",
    "TokenCryptoError",
    "default_root",
    "delete_token_file",
    "load_or_create_master_key",
    "master_key_path",
    "read_token_file",
    "write_token_file",
]

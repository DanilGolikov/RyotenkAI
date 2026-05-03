"""Generic cryptographic helpers shared across control + pod sides.

Lived in ``ryotenkai_shared.utils.crypto.token_crypto`` until the
post-Phase-B audit (ADR row 8) noticed it had no API-process logic in
it — it's a pure file/env-keyed AES-GCM token wrapper. Moved here so
``ryotenkai_shared.config.secrets.model`` can use it without crossing
the shared→control boundary.
"""

from __future__ import annotations

from ryotenkai_shared.utils.crypto.token_crypto import (
    TokenCrypto,
    TokenCryptoError,
    default_root,
    delete_token_file,
    load_or_create_master_key,
    master_key_path,
    read_token_file,
    write_token_file,
)

__all__ = [
    "TokenCrypto",
    "TokenCryptoError",
    "default_root",
    "delete_token_file",
    "load_or_create_master_key",
    "master_key_path",
    "read_token_file",
    "write_token_file",
]

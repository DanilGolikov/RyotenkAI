from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

import pytest

from src.api.services.token_crypto import (
    MASTER_KEY_BYTES,
    TokenCrypto,
    TokenCryptoError,
    delete_token_file,
    load_or_create_master_key,
    master_key_path,
    read_token_file,
    write_token_file,
)


def test_master_key_generated_on_first_call(tmp_path: Path) -> None:
    key = load_or_create_master_key(tmp_path)
    assert len(key) == MASTER_KEY_BYTES
    assert master_key_path(tmp_path).is_file()
    mode = master_key_path(tmp_path).stat().st_mode & 0o777
    # 0600 is the invariant; allow anyone who tightens it further.
    assert mode & 0o077 == 0, f"master key is world/group-readable: {oct(mode)}"


def test_master_key_stable_across_calls(tmp_path: Path) -> None:
    a = load_or_create_master_key(tmp_path)
    b = load_or_create_master_key(tmp_path)
    assert a == b


def test_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = os.urandom(MASTER_KEY_BYTES)
    monkeypatch.setenv("RYOTENKAI_SECRET_KEY", base64.b64encode(key).decode("ascii"))
    # Write a different file key — env should still win.
    (tmp_path / ".secret.key").write_bytes(os.urandom(MASTER_KEY_BYTES))
    loaded = load_or_create_master_key(tmp_path)
    assert loaded == key


def test_env_rejects_bad_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RYOTENKAI_SECRET_KEY", "???not base64???")
    with pytest.raises(TokenCryptoError):
        load_or_create_master_key()


def test_env_rejects_wrong_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RYOTENKAI_SECRET_KEY", base64.b64encode(b"short").decode("ascii"))
    with pytest.raises(TokenCryptoError):
        load_or_create_master_key()


def test_encrypt_decrypt_roundtrip(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    blob = crypto.encrypt("hf_secret_value")
    assert crypto.decrypt(blob) == "hf_secret_value"


def test_encrypt_produces_different_ciphertext_each_call(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    a = crypto.encrypt("same")
    b = crypto.encrypt("same")
    assert a != b  # nonce randomised


def test_decrypt_rejects_tampered_blob(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    blob = bytearray(crypto.encrypt("token"))
    blob[-1] ^= 0x01  # flip one bit in the tag
    with pytest.raises(TokenCryptoError):
        crypto.decrypt(bytes(blob))


def test_decrypt_rejects_short_blob(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    with pytest.raises(TokenCryptoError):
        crypto.decrypt(b"short")


def test_token_file_roundtrip_and_permissions(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    token_path = tmp_path / "workspace" / "token.enc"

    assert read_token_file(token_path, crypto) is None

    write_token_file(token_path, "xyz-token", crypto)
    assert token_path.is_file()
    mode = token_path.stat().st_mode
    assert stat.S_IMODE(mode) & 0o077 == 0, "token.enc must not be group/other readable"

    assert read_token_file(token_path, crypto) == "xyz-token"


def test_token_file_overwrite_and_delete(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    token_path = tmp_path / "t.enc"
    write_token_file(token_path, "a", crypto)
    write_token_file(token_path, "b", crypto)
    assert read_token_file(token_path, crypto) == "b"

    assert delete_token_file(token_path) is True
    assert delete_token_file(token_path) is False
    assert read_token_file(token_path, crypto) is None


def test_plaintext_type_checked(tmp_path: Path) -> None:
    crypto = TokenCrypto(root=tmp_path)
    with pytest.raises(TypeError):
        crypto.encrypt(b"bytes")  # type: ignore[arg-type]

"""Tests for the per-resource token resolver.

The HF token is now sourced exclusively from ``HF_TOKEN`` (env or
``secrets.env``). Per-integration token files for HF were removed
along with the ``integration: <id>`` shorthand. Provider tokens
(``RUNPOD_API_KEY``) keep the per-provider workspace lookup —
those still flow through Settings → Providers.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from ryotenkai_shared.utils.crypto.token_crypto import TokenCrypto, write_token_file
from ryotenkai_shared.config.secrets.model import Secrets


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ``~/.ryotenkai`` per test and pin a master key."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "RYOTENKAI_SECRET_KEY", base64.b64encode(os.urandom(32)).decode("ascii")
    )


def _write_provider_token(provider_id: str, value: str) -> None:
    path = Path.home() / ".ryotenkai" / "providers" / provider_id / "token.enc"
    write_token_file(path, value, TokenCrypto())


def test_hf_token_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    secrets = Secrets()
    assert secrets.get_hf_token() == "env-token"


def test_hf_token_none_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secrets = Secrets()
    assert secrets.get_hf_token() is None


def test_provider_token_prefers_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "env-runpod")
    _write_provider_token("runpod-prod", "ws-runpod")

    secrets = Secrets()
    assert secrets.get_provider_token("runpod-prod") == "ws-runpod"


def test_provider_token_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "env-runpod")
    secrets = Secrets()
    assert secrets.get_provider_token("runpod-none") == "env-runpod"


def test_hf_token_field_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Secrets`` constructs successfully without ``HF_TOKEN`` in env."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secrets = Secrets()
    assert secrets.hf_token is None

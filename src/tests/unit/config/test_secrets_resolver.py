"""Tests for the per-resource token resolver (PR4)."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from src.api.services.token_crypto import TokenCrypto, write_token_file
from src.config.secrets.model import Secrets


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ``~/.ryotenkai`` per test and pin a master key."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "RYOTENKAI_SECRET_KEY", base64.b64encode(os.urandom(32)).decode("ascii")
    )


def _write_integration_token(integration_id: str, value: str) -> None:
    path = Path.home() / ".ryotenkai" / "integrations" / integration_id / "token.enc"
    write_token_file(path, value, TokenCrypto())


def _write_provider_token(provider_id: str, value: str) -> None:
    path = Path.home() / ".ryotenkai" / "providers" / provider_id / "token.enc"
    write_token_file(path, value, TokenCrypto())


def test_hf_token_prefers_integration_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    _write_integration_token("hf-prod", "workspace-token")

    secrets = Secrets()
    assert secrets.get_hf_token("hf-prod") == "workspace-token"


def test_hf_token_falls_back_to_env_when_workspace_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    secrets = Secrets()
    assert secrets.get_hf_token("nonexistent") == "env-token"


def test_hf_token_none_without_any_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secrets = Secrets()
    assert secrets.get_hf_token("nonexistent") is None
    assert secrets.get_hf_token(None) is None


def test_hf_token_no_integration_id_returns_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-only")
    secrets = Secrets()
    assert secrets.get_hf_token() == "env-only"


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


def test_hf_token_field_now_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """PR4: Secrets constructs successfully without ``HF_TOKEN`` in env."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secrets = Secrets()
    assert secrets.hf_token is None

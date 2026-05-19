"""Tests for :class:`MlflowPromptRegistry`."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_shared.infrastructure.mlflow.prompt_registry import (
    MlflowPromptRegistry,
)


def test_constructor_rejects_empty_tracking_uri() -> None:
    """Empty URI must raise; otherwise mlflow would silently default."""
    with pytest.raises(ValueError, match="tracking_uri must be non-empty"):
        MlflowPromptRegistry(tracking_uri="")


def test_load_returns_artifact_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: mlflow.genai.load_prompt returns artifact -> registry returns it."""
    fake_prompt = MagicMock()
    fake_prompt.template = "hello"
    fake_prompt.name = "p"
    fake_prompt.version = "1"

    set_uri_called: dict[str, Any] = {}

    fake_mlflow = types.ModuleType("mlflow")

    def _set_tracking_uri(uri: str) -> None:
        set_uri_called["uri"] = uri

    fake_mlflow.set_tracking_uri = _set_tracking_uri  # type: ignore[attr-defined]
    fake_genai = types.ModuleType("mlflow.genai")
    fake_genai.load_prompt = lambda _name: fake_prompt  # type: ignore[attr-defined]
    fake_mlflow.genai = fake_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.genai", fake_genai)

    registry = MlflowPromptRegistry(tracking_uri="http://mlflow.test")
    result = registry.load("my_prompt", timeout_s=5.0)

    assert result is fake_prompt
    assert set_uri_called["uri"] == "http://mlflow.test"


def test_load_returns_none_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """When mlflow.genai.load_prompt raises, return None (soft failure)."""
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda _uri: None  # type: ignore[attr-defined]
    fake_genai = types.ModuleType("mlflow.genai")

    def _raise(_name: str) -> Any:
        raise RuntimeError("boom")

    fake_genai.load_prompt = _raise  # type: ignore[attr-defined]
    fake_mlflow.genai = fake_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.genai", fake_genai)

    registry = MlflowPromptRegistry(tracking_uri="http://mlflow.test")
    result = registry.load("my_prompt", timeout_s=5.0)

    assert result is None


def test_load_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the registry call exceeds the deadline, return None."""
    import time as _time

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda _uri: None  # type: ignore[attr-defined]
    fake_genai = types.ModuleType("mlflow.genai")

    def _slow(_name: str) -> Any:
        _time.sleep(2.0)
        return None

    fake_genai.load_prompt = _slow  # type: ignore[attr-defined]
    fake_mlflow.genai = fake_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.genai", fake_genai)

    registry = MlflowPromptRegistry(tracking_uri="http://mlflow.test")
    result = registry.load("slow_prompt", timeout_s=0.1)

    assert result is None

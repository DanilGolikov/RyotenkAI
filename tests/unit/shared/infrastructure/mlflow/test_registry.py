"""Unit tests for :class:`MlflowModelRegistry` (Phase M5).

The adapter is a thin wrapper around ``mlflow.MlflowClient``; we
stub the ``mlflow`` module to assert the right calls land at the
right times without depending on a live MLflow server.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_stub_mlflow(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install a stub ``mlflow`` + ``mlflow.tracking`` pair.

    Returns the stub ``mlflow`` module and the stub ``MlflowClient``
    class (whose return value is the client instance the adapter
    will see).
    """
    stub_mlflow = MagicMock(name="mlflow")
    stub_tracking = MagicMock(name="mlflow.tracking")
    stub_client_instance = MagicMock(name="MlflowClient")
    stub_tracking.MlflowClient = MagicMock(return_value=stub_client_instance)
    monkeypatch.setitem(sys.modules, "mlflow", stub_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", stub_tracking)
    return stub_mlflow, stub_client_instance


def test_register_calls_mlflow_register_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from ryotenkai_shared.infrastructure.mlflow.registry import (
        MlflowModelRegistry,
        MlflowModelVersion,
    )

    stub_mlflow, _ = _install_stub_mlflow(monkeypatch)
    stub_mlflow.register_model.return_value = SimpleNamespace(
        name="ryo/exp/family", version=5, run_id="r-1",
    )

    reg = MlflowModelRegistry(tracking_uri="http://mlflow:5000")
    mv = reg.register("runs:/r-1/model", "ryo/exp/family")

    stub_mlflow.register_model.assert_called_once_with(
        "runs:/r-1/model", "ryo/exp/family",
    )
    assert isinstance(mv, MlflowModelVersion)
    assert mv.name == "ryo/exp/family"
    assert mv.version == "5"  # stringified
    assert mv.run_id == "r-1"


def test_set_alias_delegates_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from ryotenkai_shared.infrastructure.mlflow.registry import MlflowModelRegistry

    _, stub_client = _install_stub_mlflow(monkeypatch)

    reg = MlflowModelRegistry(tracking_uri="http://mlflow:5000")
    reg.set_alias("ryo/exp/family", "champion", "3")

    stub_client.set_registered_model_alias.assert_called_once_with(
        "ryo/exp/family", "champion", "3",
    )


def test_resolve_alias_returns_typed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    from ryotenkai_shared.infrastructure.mlflow.registry import (
        MlflowModelRegistry,
        MlflowModelVersion,
    )

    _, stub_client = _install_stub_mlflow(monkeypatch)
    stub_client.get_model_version_by_alias.return_value = SimpleNamespace(
        name="ryo/exp/family", version=11, run_id="r-9",
    )

    reg = MlflowModelRegistry(tracking_uri="http://mlflow:5000")
    mv = reg.resolve_alias("ryo/exp/family", "champion")

    stub_client.get_model_version_by_alias.assert_called_once_with(
        "ryo/exp/family", "champion",
    )
    assert isinstance(mv, MlflowModelVersion)
    assert mv.version == "11"
    assert mv.run_id == "r-9"


def test_constructor_rejects_empty_uri() -> None:
    from ryotenkai_shared.infrastructure.mlflow.registry import MlflowModelRegistry

    with pytest.raises(ValueError, match="tracking_uri must be non-empty"):
        MlflowModelRegistry(tracking_uri="")
    with pytest.raises(ValueError, match="tracking_uri must be non-empty"):
        MlflowModelRegistry(tracking_uri="   ")


def test_client_constructed_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy MlflowClient is built on first call and cached thereafter."""
    from ryotenkai_shared.infrastructure.mlflow.registry import MlflowModelRegistry

    stub_mlflow, stub_client = _install_stub_mlflow(monkeypatch)
    # Replace the imported attribute access with our stub.
    client_cls = sys.modules["mlflow.tracking"].MlflowClient

    reg = MlflowModelRegistry(tracking_uri="http://mlflow:5000")
    reg.set_alias("m", "a", "1")
    reg.set_alias("m", "a", "2")
    assert client_cls.call_count == 1


def test_tracking_uri_property() -> None:
    from ryotenkai_shared.infrastructure.mlflow.registry import MlflowModelRegistry

    reg = MlflowModelRegistry(tracking_uri="http://mlflow:5000")
    assert reg.tracking_uri == "http://mlflow:5000"

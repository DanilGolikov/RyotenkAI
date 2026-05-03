"""Tests for MLflowEnvironment — single owner of MLflow process-wide state."""

from __future__ import annotations

import atexit
import os
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.infrastructure.mlflow.environment import MLflowEnvironment

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake = ModuleType("mlflow")
    fake.set_tracking_uri = MagicMock()  # type: ignore[attr-defined]
    fluent = ModuleType("mlflow.tracking.fluent")
    fluent._safe_end_run = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.tracking.fluent", fluent)
    return fake


class TestActivateDeactivate:
    def test_activate_sets_env_vars_when_ca_bundle_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002", ca_bundle_path="/path/to/ca.pem")

        env.activate()

        assert os.environ["REQUESTS_CA_BUNDLE"] == "/path/to/ca.pem"
        assert os.environ["SSL_CERT_FILE"] == "/path/to/ca.pem"
        assert env.is_active is True

    def test_activate_does_not_set_env_vars_without_ca_bundle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002")

        env.activate()

        assert "REQUESTS_CA_BUNDLE" not in os.environ
        assert "SSL_CERT_FILE" not in os.environ
        assert env.is_active is True

    def test_activate_calls_mlflow_set_tracking_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002")

        env.activate()

        fake.set_tracking_uri.assert_called_once_with("http://localhost:5002")

    def test_deactivate_restores_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        os.environ["REQUESTS_CA_BUNDLE"] = "/original/bundle"
        os.environ["SSL_CERT_FILE"] = "/original/cert"

        env = MLflowEnvironment("http://localhost:5002", ca_bundle_path="/new/ca.pem")
        env.activate()
        assert os.environ["REQUESTS_CA_BUNDLE"] == "/new/ca.pem"

        env.deactivate()
        assert os.environ["REQUESTS_CA_BUNDLE"] == "/original/bundle"
        assert os.environ["SSL_CERT_FILE"] == "/original/cert"
        assert env.is_active is False

    def test_deactivate_removes_env_vars_if_not_previously_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002", ca_bundle_path="/path/to/ca.pem")

        env.activate()
        assert "REQUESTS_CA_BUNDLE" in os.environ

        env.deactivate()
        assert "REQUESTS_CA_BUNDLE" not in os.environ
        assert "SSL_CERT_FILE" not in os.environ


class TestIdempotency:
    def test_double_activate_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002")

        env.activate()
        env.activate()

        assert fake.set_tracking_uri.call_count == 1

    def test_double_deactivate_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002", ca_bundle_path="/ca.pem")

        env.activate()
        env.deactivate()
        env.deactivate()

        assert env.is_active is False
        assert "REQUESTS_CA_BUNDLE" not in os.environ

    def test_deactivate_without_activate_is_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        env = MLflowEnvironment("http://localhost:5002")
        env.deactivate()
        assert env.is_active is False


class TestForceUnregisterAtexit:
    def test_force_unregister_is_safe_without_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "mlflow.tracking.fluent", raising=False)
        MLflowEnvironment.force_unregister_atexit()

    def test_force_unregister_calls_atexit_unregister(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)

        calls: list[Any] = []
        monkeypatch.setattr(atexit, "unregister", lambda fn: calls.append(fn))

        MLflowEnvironment.force_unregister_atexit()

        assert len(calls) == 1

    def test_deactivate_also_unregisters_atexit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)

        calls: list[Any] = []
        monkeypatch.setattr(atexit, "unregister", lambda fn: calls.append(fn))

        env = MLflowEnvironment("http://localhost:5002")
        env.activate()
        env.deactivate()

        assert len(calls) == 1


class TestProperties:
    def test_tracking_uri_property(self) -> None:
        env = MLflowEnvironment("http://custom-uri:5002")
        assert env.tracking_uri == "http://custom-uri:5002"

    def test_is_active_initially_false(self) -> None:
        env = MLflowEnvironment("http://localhost:5002")
        assert env.is_active is False

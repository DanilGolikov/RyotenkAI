"""Unit tests for :class:`HFMlflowWiring` (Phase M4)."""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_pod.trainer.mlflow.hf_wiring import HFMlflowWiring
from ryotenkai_shared.errors import ConfigInvalidError


class _FakeTrainingArgs:
    """Minimal stand-in for :class:`transformers.TrainingArguments`."""

    def __init__(self, report_to: Any = "none") -> None:
        self.report_to = report_to


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the MLFLOW_* env vars before each test."""
    for k in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_RUN_ID",
        "MLFLOW_NESTED_RUN",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
    ):
        monkeypatch.delenv(k, raising=False)


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-abc")
    monkeypatch.setenv("MLFLOW_NESTED_RUN", "TRUE")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp-1")


def test_validate_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    HFMlflowWiring.validate_env()  # does not raise


def test_validate_env_missing_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only set URI; the other three must trigger fail-fast.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    with pytest.raises(ConfigInvalidError) as exc:
        HFMlflowWiring.validate_env()
    assert "missing required env vars" in str(exc.value.detail).lower()
    assert "MLFLOW_RUN_ID" in exc.value.context["missing_keys"]
    assert "MLFLOW_NESTED_RUN" in exc.value.context["missing_keys"]
    assert "MLFLOW_EXPERIMENT_NAME" in exc.value.context["missing_keys"]


def test_validate_env_lowercase_nested_run_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MLFLOW_NESTED_RUN`` must be the literal 'TRUE' (R-01)."""
    _set_env(monkeypatch)
    monkeypatch.setenv("MLFLOW_NESTED_RUN", "true")
    with pytest.raises(ConfigInvalidError) as exc:
        HFMlflowWiring.validate_env()
    assert exc.value.context["legacy_code"] == "MLFLOW_NESTED_RUN_INVALID"


def test_validate_env_empty_value_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env(monkeypatch)
    monkeypatch.setenv("MLFLOW_RUN_ID", "   ")
    with pytest.raises(ConfigInvalidError):
        HFMlflowWiring.validate_env()


def test_configure_training_args_forces_report_to_mlflow() -> None:
    args = _FakeTrainingArgs(report_to=["tensorboard", "wandb"])
    HFMlflowWiring.configure_training_args(args, local_rank=None)
    assert args.report_to == ["mlflow"]


def test_configure_training_args_no_mlflow_module_does_not_crash() -> None:
    """The wiring must degrade gracefully if mlflow is not installed."""
    args = _FakeTrainingArgs()
    # Doesn't matter whether mlflow is installed; the helper must
    # never crash when ``set_system_metrics_node_id`` is unavailable.
    HFMlflowWiring.configure_training_args(args, local_rank=3)
    assert args.report_to == ["mlflow"]


def test_configure_training_args_local_rank_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``None`` local_rank should pass through as node id 0."""
    captured: dict[str, int] = {}

    class _StubMlflow:
        @staticmethod
        def set_system_metrics_node_id(node_id: int) -> None:
            captured["node_id"] = node_id

    monkeypatch.setitem(__import__("sys").modules, "mlflow", _StubMlflow())
    args = _FakeTrainingArgs()
    HFMlflowWiring.configure_training_args(args, local_rank=None)
    assert captured["node_id"] == 0


def test_configure_training_args_local_rank_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    class _StubMlflow:
        @staticmethod
        def set_system_metrics_node_id(node_id: int) -> None:
            captured["node_id"] = node_id

    monkeypatch.setitem(__import__("sys").modules, "mlflow", _StubMlflow())
    args = _FakeTrainingArgs()
    HFMlflowWiring.configure_training_args(args, local_rank=7)
    assert captured["node_id"] == 7

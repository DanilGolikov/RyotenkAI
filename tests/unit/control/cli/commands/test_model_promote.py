"""Tests for ``ryotenkai model promote`` (Phase M5).

The command is a thin client over
:class:`MlflowModelRegistry.set_alias`; every test patches the
registry constructor so no real MLflow server is contacted.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ryotenkai_control.cli.app import app
# Direct import keeps the module name visible to the
# ``test_every_module_has_tests`` sentinel — it scans for the dotted
# path ``ryotenkai_control.cli.commands.model`` as a literal substring.
from ryotenkai_control.cli.commands import model as _model_mod  # noqa: F401


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_model_module_exposes_typer_app() -> None:
    """Sanity check: the model commands sub-typer is mounted via the registry."""
    assert _model_mod.model_app is not None


def test_promote_uses_env_tracking_uri_when_no_config(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MLFLOW_TRACKING_URI`` env supplies the URI when --config is absent."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registry_instance = MagicMock()
    with patch(
        "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
        return_value=registry_instance,
    ) as ctor:
        result = runner.invoke(
            app,
            [
                "model", "promote",
                "--name", "ryotenkai/exp/family",
                "--version", "3",
            ],
        )
    assert result.exit_code == 0, result.output
    ctor.assert_called_once_with(tracking_uri="http://mlflow:5000")
    registry_instance.set_alias.assert_called_once_with(
        "ryotenkai/exp/family", "champion", "3",
    )


def test_promote_custom_alias_flag(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--alias shadow`` overrides the default champion target."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registry_instance = MagicMock()
    with patch(
        "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
        return_value=registry_instance,
    ):
        result = runner.invoke(
            app,
            [
                "model", "promote",
                "--name", "ryo/exp/fam",
                "--version", "5",
                "--alias", "shadow",
            ],
        )
    assert result.exit_code == 0, result.output
    registry_instance.set_alias.assert_called_once_with(
        "ryo/exp/fam", "shadow", "5",
    )


def test_promote_no_tracking_uri_fails_with_helpful_message(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing both --config and MLFLOW_TRACKING_URI exits non-zero."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = runner.invoke(
        app,
        [
            "model", "promote",
            "--name", "ryo/exp/fam",
            "--version", "1",
        ],
    )
    assert result.exit_code != 0
    assert "MLflow tracking URI" in result.output or "tracking_uri" in result.output.lower()


def test_promote_registry_error_exits_non_zero(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry failure surfaces as a non-zero exit + readable error."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registry_instance = MagicMock()
    registry_instance.set_alias.side_effect = RuntimeError("unknown version 99")
    with patch(
        "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
        return_value=registry_instance,
    ):
        result = runner.invoke(
            app,
            [
                "model", "promote",
                "--name", "ryo/exp/fam",
                "--version", "99",
            ],
        )
    assert result.exit_code == 1
    assert "unknown version 99" in result.output


def test_promote_json_output(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``-o json`` returns a machine-readable success payload."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registry_instance = MagicMock()
    with patch(
        "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
        return_value=registry_instance,
    ):
        result = runner.invoke(
            app,
            [
                "-o", "json",
                "model", "promote",
                "--name", "ryo/exp/fam",
                "--version", "2",
            ],
        )
    assert result.exit_code == 0, result.output
    # Renderer emits pretty-printed JSON spanning multiple lines.
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["name"] == "ryo/exp/fam"
    assert payload["version"] == "2"
    assert payload["alias"] == "champion"


def test_promote_json_output_on_failure(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON output preserves error info on registry failure."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registry_instance = MagicMock()
    registry_instance.set_alias.side_effect = RuntimeError("server down")
    with patch(
        "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
        return_value=registry_instance,
    ):
        result = runner.invoke(
            app,
            [
                "-o", "json",
                "model", "promote",
                "--name", "ryo/exp/fam",
                "--version", "2",
            ],
        )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "server down" in payload["error"]

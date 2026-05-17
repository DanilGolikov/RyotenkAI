"""Unit tests for :mod:`ryotenkai_control.pipeline.stages.dataset_validator.plugin_runner`.

Phase A2 Batch 8 — raise-based migration. ``PluginRunner.run`` returns a
metrics ``dict`` on success and raises
:class:`DatasetValidationFailedError` on failure
(``context["critical"]`` distinguishes the legacy
``DATASET_VALIDATION_CRITICAL_FAILURE`` /
``DATASET_VALIDATION_ERROR`` codes).

Phase 4 (event-system unification, 2026-05-16): the previous
``DatasetValidatorEventCallbacks`` dataclass was replaced with a direct
:class:`ValidationArtifactManager` collaborator. Tests now spy on
``MagicMock(spec=ValidationArtifactManager)`` — that class is a regular
class (not a Protocol) so this mocking pattern remains compliant with
the no-Protocol-mocking sentinel.
"""

from __future__ import annotations

from types import SimpleNamespace

from unittest.mock import MagicMock

import pytest

from ryotenkai_control.data.validation.base import ValidationResult
from ryotenkai_control.pipeline.stages.dataset_validator.artifact_manager import (
    ValidationArtifactManager,
)
from ryotenkai_control.pipeline.stages.dataset_validator.constants import (
    VALIDATION_STATUS_KEY,
    VALIDATION_STATUS_PASSED,
)
from ryotenkai_control.pipeline.stages.dataset_validator.plugin_runner import PluginRunner
from ryotenkai_shared.errors import DatasetValidationFailedError

pytestmark = pytest.mark.unit


def _mock_artifact_recorder() -> MagicMock:
    """Return a ``MagicMock(spec=ValidationArtifactManager)`` so the
    test assertion surface mirrors the real ``on_*`` recorder method
    set without dragging a real :class:`StageArtifactCollector` into
    the fixture.

    ``ValidationArtifactManager`` is a concrete class (not a Protocol),
    so this stays compliant with :mod:`tests._lint.test_no_protocol_mocking`.
    """
    return MagicMock(spec=ValidationArtifactManager)


# ------------------------------------------------------------------
# Test plugin doubles
# ------------------------------------------------------------------


class _OkPlugin:
    name = "p"
    params: dict = {"x": 1}
    thresholds: dict = {"threshold": 1}

    def get_description(self) -> str:
        return "d"

    def validate(self, dataset):
        _ = dataset
        return ValidationResult(
            plugin_name="p",
            passed=True,
            params={"x": 1},
            thresholds={"threshold": 1},
            metrics={"m": 1.0},
            warnings=["w"],
            errors=[],
            execution_time_ms=1.0,
        )

    def get_recommendations(self, result):
        _ = result
        return []


class _FailPlugin:
    name = "p"
    params: dict = {"x": 1}
    thresholds: dict = {"threshold": 1}

    def get_description(self) -> str:
        return "d"

    def validate(self, dataset):
        _ = dataset
        return ValidationResult(
            plugin_name="p",
            passed=False,
            params={"x": 1},
            thresholds={"threshold": 1},
            metrics={"m": 0.0},
            warnings=[],
            errors=["e"],
            execution_time_ms=1.0,
        )

    def get_recommendations(self, result):
        _ = result
        return ["r"]


class _CrashPlugin:
    name = "p"
    params: dict = {"x": 1}
    thresholds: dict = {"threshold": 1}

    def get_description(self) -> str:
        return "d"

    def validate(self, dataset):
        _ = dataset
        raise RuntimeError("boom")

    def get_recommendations(self, result):
        _ = result
        return []


# ------------------------------------------------------------------
# success path
# ------------------------------------------------------------------


def test_run_success_records_plugin_start_complete_and_validation_completed():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=0))

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _OkPlugin(), {"train"})],
        split_name="train",
    )
    assert isinstance(res, dict)
    assert res[VALIDATION_STATUS_KEY] == VALIDATION_STATUS_PASSED
    # Metrics carry split-prefixed key.
    assert res["train.p_main.m"] == 1.0
    assert res["warnings"] == ["w"]
    rec.on_plugin_start.assert_called_once()
    rec.on_plugin_complete.assert_called_once()
    rec.on_validation_completed.assert_called_once()
    rec.on_plugin_failed.assert_not_called()
    rec.on_validation_failed.assert_not_called()


# ------------------------------------------------------------------
# failure path + critical threshold + recommendations
# ------------------------------------------------------------------


def test_run_failure_raises_and_records_plugin_failed_and_validation_failed():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=1))

    with pytest.raises(DatasetValidationFailedError):
        runner.run(
            "d",
            "ref",
            dataset=object(),
            dataset_config=dataset_config,
            plugins=[("p_main", "p", _FailPlugin(), {"train"})],
            split_name="train",
        )
    rec.on_plugin_failed.assert_called()
    failed_call = rec.on_plugin_failed.call_args
    assert failed_call is not None
    # Positional args layout: dataset_name, dataset_path, plugin_id,
    # plugin_name, params, thresholds, metrics, duration_ms, errors,
    # recommendations.
    assert failed_call.args[7] == pytest.approx(1.0)
    rec.on_validation_failed.assert_called()
    rec.on_plugin_complete.assert_not_called()
    rec.on_validation_completed.assert_not_called()


def test_run_critical_threshold_raises_with_critical_flag():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=1))

    with pytest.raises(DatasetValidationFailedError) as excinfo:
        runner.run(
            "d",
            "ref",
            dataset=object(),
            dataset_config=dataset_config,
            plugins=[("p_main", "p", _FailPlugin(), {"train"})],
            split_name="train",
        )
    assert excinfo.value.context.get("critical") is True
    # Non-empty error list also surfaced on context for the stage.
    assert excinfo.value.context.get("errors")


def test_run_below_critical_threshold_raises_with_critical_false():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=2))

    with pytest.raises(DatasetValidationFailedError) as excinfo:
        runner.run(
            "d",
            "ref",
            dataset=object(),
            dataset_config=dataset_config,
            plugins=[("p_main", "p", _FailPlugin(), {"train"})],
            split_name="train",
        )
    assert excinfo.value.context.get("critical") is False


def test_run_critical_threshold_breaks_loop_early():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=1))

    second = _OkPlugin()
    second_validate_mock = MagicMock(wraps=second.validate)
    second.validate = second_validate_mock

    with pytest.raises(DatasetValidationFailedError):
        runner.run(
            "d",
            "ref",
            dataset=object(),
            dataset_config=dataset_config,
            plugins=[
                ("p_fail", "p", _FailPlugin(), {"train"}),
                ("p_ok", "p", second, {"train"}),
            ],
            split_name="train",
        )
    # critical threshold reached after first failure → second plugin not called
    second_validate_mock.assert_not_called()


# ------------------------------------------------------------------
# crash path
# ------------------------------------------------------------------


def test_run_plugin_crash_raises_and_records_failure():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=1))

    with pytest.raises(DatasetValidationFailedError):
        runner.run(
            "d",
            "ref",
            dataset=object(),
            dataset_config=dataset_config,
            plugins=[("p_main", "p", _CrashPlugin(), {"train"})],
            split_name="train",
        )
    rec.on_plugin_failed.assert_called_once()
    failed_call = rec.on_plugin_failed.call_args
    assert failed_call is not None
    assert failed_call.args[7] >= 0.0
    # error in args[8] mentions crash
    errors_arg = failed_call.args[8]
    assert any("crashed" in e for e in errors_arg)


def test_run_no_recorder_does_not_blow_up():
    """``artifact_recorder=None`` is the default — every recorder hook
    is silently skipped so plugin loops work in tests that don't care
    about the accumulator surface."""
    runner = PluginRunner(artifact_recorder=None)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=0))

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _OkPlugin(), {"train"})],
        split_name="train",
    )
    assert isinstance(res, dict)
    assert res[VALIDATION_STATUS_KEY] == VALIDATION_STATUS_PASSED


def test_run_empty_plugins_list_is_success():
    rec = _mock_artifact_recorder()
    runner = PluginRunner(artifact_recorder=rec)
    dataset_config = SimpleNamespace(validations=MagicMock(critical_failures=0))

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[],
        split_name="train",
    )
    assert isinstance(res, dict)
    assert res[VALIDATION_STATUS_KEY] == VALIDATION_STATUS_PASSED
    rec.on_validation_completed.assert_called_once()

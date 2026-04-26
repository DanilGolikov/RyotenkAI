"""Unit tests for src.pipeline.stages.dataset_validator.plugin_runner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.validation.base import ValidationResult
from src.pipeline.stages.dataset_validator.plugin_runner import PluginRunner
from src.pipeline.stages.dataset_validator.stage import DatasetValidatorEventCallbacks

pytestmark = pytest.mark.unit


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


def test_run_success_fires_complete_and_validation_completed_callbacks():
    cb = DatasetValidatorEventCallbacks(
        on_plugin_start=MagicMock(),
        on_plugin_complete=MagicMock(),
        on_plugin_failed=MagicMock(),
        on_validation_completed=MagicMock(),
        on_validation_failed=MagicMock(),
    )
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=0)

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _OkPlugin(), {"train"})],
        split_name="train",
    )
    assert res.is_success()
    cb.on_plugin_start.assert_called_once()
    cb.on_plugin_complete.assert_called_once()
    cb.on_validation_completed.assert_called_once()
    cb.on_plugin_failed.assert_not_called()
    cb.on_validation_failed.assert_not_called()


# ------------------------------------------------------------------
# failure path + critical threshold + recommendations
# ------------------------------------------------------------------


def test_run_failure_fires_plugin_failed_and_validation_failed_callbacks():
    cb = DatasetValidatorEventCallbacks(
        on_plugin_complete=MagicMock(),
        on_plugin_failed=MagicMock(),
        on_validation_completed=MagicMock(),
        on_validation_failed=MagicMock(),
    )
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=1)

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _FailPlugin(), {"train"})],
        split_name="train",
    )
    assert res.is_failure()
    cb.on_plugin_failed.assert_called()
    failed_call = cb.on_plugin_failed.call_args
    assert failed_call is not None
    assert failed_call.args[7] == pytest.approx(1.0)
    cb.on_validation_failed.assert_called()
    cb.on_plugin_complete.assert_not_called()
    cb.on_validation_completed.assert_not_called()


def test_run_critical_threshold_returns_critical_failure_code():
    cb = DatasetValidatorEventCallbacks(on_plugin_failed=MagicMock(), on_validation_failed=MagicMock())
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=1)

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _FailPlugin(), {"train"})],
        split_name="train",
    )
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DATASET_VALIDATION_CRITICAL_FAILURE"


def test_run_below_critical_threshold_returns_validation_error_code():
    cb = DatasetValidatorEventCallbacks(on_plugin_failed=MagicMock(), on_validation_failed=MagicMock())
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=2)  # need 2 failures, only 1

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _FailPlugin(), {"train"})],
        split_name="train",
    )
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "DATASET_VALIDATION_ERROR"


def test_run_critical_threshold_breaks_loop_early():
    cb = DatasetValidatorEventCallbacks()
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=1)

    second = _OkPlugin()
    second_validate_mock = MagicMock(wraps=second.validate)
    second.validate = second_validate_mock

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


def test_run_plugin_crash_fires_failed_callback():
    cb = DatasetValidatorEventCallbacks(on_plugin_failed=MagicMock(), on_validation_failed=MagicMock())
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=1)

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _CrashPlugin(), {"train"})],
        split_name="train",
    )
    assert res.is_failure()
    cb.on_plugin_failed.assert_called_once()
    failed_call = cb.on_plugin_failed.call_args
    assert failed_call is not None
    assert failed_call.args[7] >= 0.0
    # error in args[8] mentions crash
    errors_arg = failed_call.args[8]
    assert any("crashed" in e for e in errors_arg)


def test_run_no_callbacks_does_not_blow_up():
    """Default DatasetValidatorEventCallbacks() leaves all 7 callbacks None."""
    runner = PluginRunner(callbacks=DatasetValidatorEventCallbacks())
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=0)

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[("p_main", "p", _OkPlugin(), {"train"})],
        split_name="train",
    )
    assert res.is_success()


def test_run_empty_plugins_list_is_success():
    cb = DatasetValidatorEventCallbacks(on_validation_completed=MagicMock())
    runner = PluginRunner(callbacks=cb)
    dataset_config = MagicMock()
    dataset_config.validations = MagicMock(critical_failures=0)

    res = runner.run(
        "d",
        "ref",
        dataset=object(),
        dataset_config=dataset_config,
        plugins=[],
        split_name="train",
    )
    assert res.is_success()
    cb.on_validation_completed.assert_called_once()

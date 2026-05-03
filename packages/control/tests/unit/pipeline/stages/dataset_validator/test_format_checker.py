"""Unit tests for src.pipeline.stages.dataset_validator.format_checker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.validation.standalone import FormatCheckResult
from src.pipeline.stages.dataset_validator.format_checker import FormatChecker
from src.utils.result import DatasetError, Failure, Ok

pytestmark = pytest.mark.unit


@pytest.fixture
def checker() -> FormatChecker:
    return FormatChecker(config=MagicMock())


@pytest.fixture
def dataset() -> MagicMock:
    return MagicMock()


def test_check_returns_ok_when_no_phases(monkeypatch, checker, dataset):
    """Empty strategy_phases → Ok (no formats to check)."""
    monkeypatch.setattr(
        "src.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: Ok([]),
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[])
    assert result.is_ok()


def test_check_returns_ok_when_all_phases_pass(monkeypatch, checker, dataset):
    monkeypatch.setattr(
        "src.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: Ok(
            [
                FormatCheckResult(strategy_type="sft", ok=True, message=""),
                FormatCheckResult(strategy_type="dpo", ok=True, message=""),
            ]
        ),
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[MagicMock(), MagicMock()])
    assert result.is_ok()


def test_check_early_fails_on_first_failed_strategy(monkeypatch, checker, dataset):
    """First failed strategy short-circuits with DATASET_FORMAT_ERROR."""
    monkeypatch.setattr(
        "src.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: Ok(
            [
                FormatCheckResult(strategy_type="sft", ok=True, message=""),
                FormatCheckResult(strategy_type="dpo", ok=False, message="missing chosen"),
                FormatCheckResult(strategy_type="cpt", ok=False, message="never reached"),
            ]
        ),
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[MagicMock()])
    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, DatasetError)
    assert err.code == "DATASET_FORMAT_ERROR"
    assert "ds_a" in str(err)
    assert "dpo" in str(err)
    assert "missing chosen" in str(err)
    # Third entry "never reached" must NOT appear (early-fail).
    assert "never reached" not in str(err)


def test_check_propagates_underlying_failure(monkeypatch, checker, dataset):
    """If standalone helper returns Failure (e.g. unknown strategy), bubble up."""
    underlying = DatasetError(message="strategy 'xyz' not registered", code="UNKNOWN_STRATEGY")
    monkeypatch.setattr(
        "src.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: Failure(underlying),
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[MagicMock()])
    assert result.is_err()
    assert result.unwrap_err() is underlying

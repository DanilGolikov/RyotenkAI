"""Unit tests for src.pipeline.stages.dataset_validator.format_checker.

Phase A2 Batch 7: ``standalone.check_dataset_format`` now returns a bare
list (raising :class:`DatasetValidationFailedError` on the unknown-strategy
path). The :class:`FormatChecker` adapter still surfaces a ``Result``
shape for its (Batch 8) consumers; these stubs match the new standalone
contract so the adapter logic is exercised end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ryotenkai_control.data.validation.standalone import FormatCheckResult
from ryotenkai_control.pipeline.stages.dataset_validator.format_checker import FormatChecker
from ryotenkai_shared.errors import DatasetValidationFailedError
from ryotenkai_shared.utils.result import DatasetError

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
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: [],
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[])
    assert result.is_ok()


def test_check_returns_ok_when_all_phases_pass(monkeypatch, checker, dataset):
    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: [
            FormatCheckResult(strategy_type="sft", ok=True, message=""),
            FormatCheckResult(strategy_type="dpo", ok=True, message=""),
        ],
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[MagicMock(), MagicMock()])
    assert result.is_ok()


def test_check_early_fails_on_first_failed_strategy(monkeypatch, checker, dataset):
    """First failed strategy short-circuits with DATASET_FORMAT_ERROR."""
    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: [
            FormatCheckResult(strategy_type="sft", ok=True, message=""),
            FormatCheckResult(strategy_type="dpo", ok=False, message="missing chosen"),
            FormatCheckResult(strategy_type="cpt", ok=False, message="never reached"),
        ],
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
    """If standalone helper raises (e.g. unknown strategy), the adapter
    surfaces a :class:`DatasetError` with the legacy DATASET_FORMAT_ERROR
    code so its Batch 8 callers keep seeing the same outer shape."""

    def _raise(*_a, **_k):
        raise DatasetValidationFailedError(
            detail="strategy 'xyz' not registered",
            context={"legacy_code": "DATASET_FORMAT_ERROR"},
        )

    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        _raise,
    )
    result = checker.check(dataset, "ds_a", strategy_phases=[MagicMock()])
    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, DatasetError)
    assert err.code == "DATASET_FORMAT_ERROR"
    assert "strategy 'xyz' not registered" in err.message

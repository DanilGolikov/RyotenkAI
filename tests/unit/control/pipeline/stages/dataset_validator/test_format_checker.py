"""Unit tests for src.pipeline.stages.dataset_validator.format_checker.

Phase A2 Batch 8 — raise-based migration. ``FormatChecker.check`` returns
``None`` on success and raises :class:`DatasetValidationFailedError` on
the first failed strategy (or propagates the same error type from
``check_dataset_format`` on the "unknown strategy" branch).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ryotenkai_control.data.validation.standalone import FormatCheckResult
from ryotenkai_control.pipeline.stages.dataset_validator.format_checker import FormatChecker
from ryotenkai_shared.errors import DatasetValidationFailedError

pytestmark = pytest.mark.unit


@pytest.fixture
def checker() -> FormatChecker:
    return FormatChecker(config=MagicMock())


@pytest.fixture
def dataset() -> MagicMock:
    return MagicMock()


def test_check_returns_none_when_no_phases(monkeypatch, checker, dataset):
    """Empty strategy_phases → returns None (no formats to check)."""
    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: [],
    )
    assert checker.check(dataset, "ds_a", strategy_phases=[]) is None


def test_check_returns_none_when_all_phases_pass(monkeypatch, checker, dataset):
    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: [
            FormatCheckResult(strategy_type="sft", ok=True, message=""),
            FormatCheckResult(strategy_type="dpo", ok=True, message=""),
        ],
    )
    assert checker.check(dataset, "ds_a", strategy_phases=[MagicMock(), MagicMock()]) is None


def test_check_early_fails_on_first_failed_strategy(monkeypatch, checker, dataset):
    """First failed strategy short-circuits with DatasetValidationFailedError."""
    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        lambda *_a, **_k: [
            FormatCheckResult(strategy_type="sft", ok=True, message=""),
            FormatCheckResult(strategy_type="dpo", ok=False, message="missing chosen"),
            FormatCheckResult(strategy_type="cpt", ok=False, message="never reached"),
        ],
    )
    with pytest.raises(DatasetValidationFailedError) as excinfo:
        checker.check(dataset, "ds_a", strategy_phases=[MagicMock()])

    exc = excinfo.value
    detail = exc.detail or ""
    assert "ds_a" in detail
    assert "dpo" in detail
    assert "missing chosen" in detail
    # Third entry "never reached" must NOT appear (early-fail).
    assert "never reached" not in detail
    # Context carries metadata for downstream handling.
    assert exc.context.get("strategy_type") == "dpo"
    assert exc.context.get("dataset_name") == "ds_a"
    assert exc.context.get("legacy_code") == "DATASET_FORMAT_ERROR"


def test_check_propagates_underlying_failure(monkeypatch, checker, dataset):
    """``check_dataset_format`` raise (e.g. unknown strategy) propagates verbatim."""

    def _raise(*_a, **_k):
        raise DatasetValidationFailedError(
            detail="strategy 'xyz' not registered",
            context={"legacy_code": "DATASET_FORMAT_ERROR"},
        )

    monkeypatch.setattr(
        "ryotenkai_control.data.validation.standalone.check_dataset_format",
        _raise,
    )
    with pytest.raises(DatasetValidationFailedError) as excinfo:
        checker.check(dataset, "ds_a", strategy_phases=[MagicMock()])

    assert "strategy 'xyz' not registered" in (excinfo.value.detail or "")

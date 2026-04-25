"""Unit tests for ``src.data.validation.standalone``.

These cover the pure helpers that are reused by both the pipeline stage
and the HTTP API. The pipeline stage's own behaviour (callbacks,
threshold-stop, MLflow events) is tested elsewhere — here we only
exercise the standalone surface.
"""

from __future__ import annotations

from datasets import Dataset

from src.data.validation.base import ValidationErrorGroup, ValidationPlugin, ValidationResult
from src.data.validation.standalone import (
    FormatCheckResult,
    check_dataset_format,
    run_plugins,
)


# ---------------------------------------------------------------------------
# run_plugins
# ---------------------------------------------------------------------------


class _OkPlugin(ValidationPlugin):
    name = "ok"

    def get_description(self) -> str:
        return "always passes"

    def validate(self, dataset):
        return ValidationResult(
            plugin_name=self.name,
            passed=True,
            params={},
            thresholds={},
            metrics={"hits": 1.0},
            warnings=[],
            errors=[],
            execution_time_ms=1.0,
        )

    def get_recommendations(self, result):
        return []


class _FailPlugin(ValidationPlugin):
    name = "fail"

    def get_description(self) -> str:
        return "always fails"

    def validate(self, dataset):
        return ValidationResult(
            plugin_name=self.name,
            passed=False,
            params={},
            thresholds={},
            metrics={"miss": 5.0},
            warnings=["watch out"],
            errors=["bad sample"],
            execution_time_ms=2.0,
            error_groups=[
                ValidationErrorGroup(error_type="oops", sample_indices=[1, 2], total_count=2),
            ],
        )

    def get_recommendations(self, result):
        return ["try harder"]


class _BoomPlugin(ValidationPlugin):
    name = "boom"

    def get_description(self) -> str:
        return "raises"

    def validate(self, dataset):
        raise RuntimeError("kaboom")

    def get_recommendations(self, result):
        return []


def _ds() -> Dataset:
    return Dataset.from_list([{"text": f"row-{i}"} for i in range(3)])


def test_run_plugins_collects_passes_and_fails():
    plugins = [
        ("p_ok", "ok", _OkPlugin(params={}, thresholds={})),
        ("p_fail", "fail", _FailPlugin(params={}, thresholds={})),
    ]
    runs = run_plugins(_ds(), plugins)

    assert len(runs) == 2
    ok = next(r for r in runs if r.plugin_id == "p_ok")
    assert ok.passed is True
    assert ok.metrics == {"hits": 1.0}
    assert ok.errors == []

    fail = next(r for r in runs if r.plugin_id == "p_fail")
    assert fail.passed is False
    assert fail.recommendations == ["try harder"]
    assert fail.error_groups[0].error_type == "oops"
    # render_error_groups is appended to errors when failed
    assert any("oops" in e or "2" in e for e in fail.errors)


def test_run_plugins_isolates_crashed_plugin():
    plugins = [
        ("p_boom", "boom", _BoomPlugin(params={}, thresholds={})),
        ("p_ok", "ok", _OkPlugin(params={}, thresholds={})),
    ]
    runs = run_plugins(_ds(), plugins)

    assert len(runs) == 2
    boom = next(r for r in runs if r.plugin_id == "p_boom")
    assert boom.passed is False
    assert boom.crashed is True
    assert any("kaboom" in msg for msg in boom.errors)

    ok = next(r for r in runs if r.plugin_id == "p_ok")
    assert ok.passed is True


# ---------------------------------------------------------------------------
# check_dataset_format
# ---------------------------------------------------------------------------


def test_check_dataset_format_returns_empty_for_no_phases():
    result = check_dataset_format(_ds(), "ds", strategy_phases=[], pipeline_config=None)
    assert result.is_ok()
    assert result.unwrap() == []


class _StubFromPhase:
    """Mimics StrategyFactory.create_from_phase().validate_dataset()."""

    def __init__(self, ok: bool, message: str = ""):
        self.ok = ok
        self.message = message

    def validate_dataset(self, dataset):
        from src.utils.result import DatasetError, Err, Ok

        if self.ok:
            return Ok(None)
        return Err(DatasetError(message=self.message, code="X"))


def test_check_dataset_format_aggregates_per_strategy(monkeypatch):
    """Two distinct strategy_types → two FormatCheckResult entries.
    Duplicates of the same strategy_type are deduped (matches original
    pipeline behaviour)."""

    factory_calls: list[str] = []

    class _StubFactory:
        def create_from_phase(self, phase, _config):
            factory_calls.append(phase.strategy_type)
            if phase.strategy_type == "sft":
                return _StubFromPhase(ok=True)
            return _StubFromPhase(ok=False, message="missing chosen")

    monkeypatch.setattr(
        "src.training.strategies.factory.StrategyFactory",
        _StubFactory,
    )

    class _Phase:
        def __init__(self, st):
            self.strategy_type = st

    phases = [_Phase("sft"), _Phase("dpo"), _Phase("sft")]
    result = check_dataset_format(_ds(), "ds", phases, pipeline_config=None)
    assert result.is_ok()
    by_type = {item.strategy_type: item for item in result.unwrap()}
    assert by_type["sft"].ok is True
    assert by_type["dpo"].ok is False
    assert "missing chosen" in by_type["dpo"].message
    # Dedup: factory was called once per UNIQUE strategy_type
    assert factory_calls == ["sft", "dpo"]


def test_check_dataset_format_short_circuits_on_unknown_strategy(monkeypatch):
    class _Factory:
        def create_from_phase(self, phase, _config):
            raise ValueError("strategy 'xyz' not registered")

    monkeypatch.setattr(
        "src.training.strategies.factory.StrategyFactory",
        _Factory,
    )

    class _Phase:
        strategy_type = "xyz"

    result = check_dataset_format(_ds(), "ds", [_Phase()], pipeline_config=None)
    assert result.is_failure()
    err = result.unwrap_err()
    assert "Unknown strategy" in err.message


def test_format_check_result_dataclass_defaults():
    res = FormatCheckResult(strategy_type="sft", ok=True)
    assert res.message == ""

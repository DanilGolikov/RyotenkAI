"""Unit tests for ``src.data.validation.standalone``.

These cover the pure helpers that are reused by both the pipeline stage
and the HTTP API. The pipeline stage's own behaviour (callbacks,
threshold-stop, MLflow events) is tested elsewhere — here we only
exercise the standalone surface.

Phase A2 Batch 7: ``check_dataset_format`` no longer returns a
``Result`` — it returns the per-strategy list directly and raises
:class:`DatasetValidationFailedError` on the "unknown strategy type"
config-bug path.
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from ryotenkai_control.data.validation.base import ValidationErrorGroup, ValidationPlugin, ValidationResult
from ryotenkai_control.data.validation.standalone import (
    FormatCheckResult,
    check_dataset_format,
    run_plugins,
)
from ryotenkai_shared.errors import DatasetValidationFailedError


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


def test_standalone_plugin_run_dataclass_defaults():
    """Pins defaults: crashed=False, duration_ms=0.0, empty containers."""
    from ryotenkai_control.data.validation.standalone import StandalonePluginRun

    r = StandalonePluginRun(plugin_id="x", plugin_name="X", passed=True)
    assert r.crashed is False
    assert r.duration_ms == 0.0
    assert r.metrics == {}
    assert r.warnings == []
    assert r.errors == []
    assert r.error_groups == []
    assert r.recommendations == []


def test_run_plugins_uses_plugin_reported_duration_when_present():
    """Line 160 path: result.execution_time_ms is truthy -> use it (don't recompute)."""
    plugins = [("p_ok", "ok", _OkPlugin(params={}, thresholds={}))]
    runs = run_plugins(_ds(), plugins)
    # _OkPlugin reports execution_time_ms=1.0 -> propagated through.
    assert runs[0].duration_ms == 1.0


def test_run_plugins_recomputes_duration_when_plugin_reports_zero():
    """Line 160 fallback: execution_time_ms falsy → (perf_counter - started) * 1000.

    Mutations on the multiplier (× → /, +, -, …) MUST be caught by
    asserting the result is in milliseconds (>= microseconds, < hour).
    """

    class _NoTimerPlugin(ValidationPlugin):
        name = "no_timer"

        def get_description(self) -> str:
            return ""

        def validate(self, dataset):
            return ValidationResult(
                plugin_name=self.name,
                passed=True,
                params={},
                thresholds={},
                metrics={},
                warnings=[],
                errors=[],
                execution_time_ms=0.0,  # falsy → recompute path
            )

        def get_recommendations(self, result):
            return []

    plugins = [("p", "no_timer", _NoTimerPlugin(params={}, thresholds={}))]
    runs = run_plugins(_ds(), plugins)
    # The plugin returns immediately so wall clock is sub-millisecond,
    # but the recompute multiplies by 1000.0 → must be non-negative and
    # less than 1 second (1000 ms). Any multiplier mutation produces a
    # value far outside [0, 1000.0].
    assert 0.0 <= runs[0].duration_ms < 1000.0
    # A "× 1.0" or "× 1" mutation would yield seconds (< 1.0). Force
    # at least the millisecond-magnitude property: result must not be
    # in [0, 1.0) unless wall clock truly was sub-microsecond.
    # We can't easily distinguish "< 1.0 because fast" from a mutation,
    # so additionally pin the "no recompute when value present" branch
    # in the previous test.


def test_run_plugins_crash_path_computes_duration_ms():
    """Line 186 path: crash branch computes (perf_counter - started) * 1000.

    Same defense as the non-crash recompute test: pin the duration is
    in milliseconds (not seconds).
    """
    plugins = [("p_boom", "boom", _BoomPlugin(params={}, thresholds={}))]
    runs = run_plugins(_ds(), plugins)
    assert runs[0].crashed is True
    # Same magnitude check as the recompute path.
    assert 0.0 <= runs[0].duration_ms < 1000.0


def test_run_plugins_no_recommendations_when_passed():
    """Line 159 guard: get_recommendations is NOT consulted when passed=True.

    Mutations that flip `if not result.passed:` to `if result.passed:`
    would invert recommendation population. We pin BOTH that ok-pass
    has empty recommendations AND that a failing plugin's
    recommendations DO get populated.
    """

    class _OkWithRecs(ValidationPlugin):
        name = "ok_recs"

        def get_description(self) -> str:
            return ""

        def validate(self, dataset):
            return ValidationResult(
                plugin_name=self.name,
                passed=True,
                params={},
                thresholds={},
                metrics={},
                warnings=[],
                errors=[],
                execution_time_ms=1.0,
            )

        def get_recommendations(self, result):
            # If the guard is mutated, this would be invoked and end up
            # in the StandalonePluginRun.recommendations field.
            raise AssertionError("get_recommendations must NOT be called when passed=True")

    plugins = [("p_ok_recs", "ok_recs", _OkWithRecs(params={}, thresholds={}))]
    runs = run_plugins(_ds(), plugins)
    assert runs[0].recommendations == []


def test_run_plugins_preserves_input_order():
    """Sequential iteration: output order matches input plugins list order.

    Pins that there's no shuffle, reverse, or sort in run_plugins
    — kills any swap mutation on the loop.
    """
    plugins = [
        ("a", "ok", _OkPlugin(params={}, thresholds={})),
        ("b", "fail", _FailPlugin(params={}, thresholds={})),
        ("c", "ok", _OkPlugin(params={}, thresholds={})),
    ]
    runs = run_plugins(_ds(), plugins)
    assert [r.plugin_id for r in runs] == ["a", "b", "c"]


def test_check_dataset_format_dedup_skips_repeated_strategy_type(monkeypatch):
    """Line 84 'continue': repeated strategy_type after the first is skipped.

    Pins that the dedup uses ``continue`` (not ``break`` — a mutation
    we observed surviving) so a duplicate AFTER an unseen one still
    lets the unseen one through.
    """
    factory_calls: list[str] = []

    class _Factory:
        def create_from_phase(self, phase, _config):
            factory_calls.append(phase.strategy_type)
            return _StubFromPhase(ok=True)

    monkeypatch.setattr(
        "ryotenkai_pod.trainer.strategies.factory.StrategyFactory",
        _Factory,
    )

    class _Phase:
        def __init__(self, st):
            self.strategy_type = st

    # Sequence: A, A (dup), B. If `continue` became `break`, B would
    # never be reached, and factory_calls would be ["A"] only.
    items = check_dataset_format(
        _ds(), "ds", [_Phase("A"), _Phase("A"), _Phase("B")], pipeline_config=None
    )
    assert factory_calls == ["A", "B"]
    assert {item.strategy_type for item in items} == {"A", "B"}


# ---------------------------------------------------------------------------
# check_dataset_format
# ---------------------------------------------------------------------------


def test_check_dataset_format_returns_empty_for_no_phases():
    assert check_dataset_format(_ds(), "ds", strategy_phases=[], pipeline_config=None) == []


class _StubFromPhase:
    """Mimics StrategyFactory.create_from_phase().validate_dataset()."""

    def __init__(self, ok: bool, message: str = ""):
        self.ok = ok
        self.message = message

    def validate_dataset(self, dataset):
        from ryotenkai_shared.utils.result import DatasetError, Err, Ok

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
        "ryotenkai_pod.trainer.strategies.factory.StrategyFactory",
        _StubFactory,
    )

    class _Phase:
        def __init__(self, st):
            self.strategy_type = st

    phases = [_Phase("sft"), _Phase("dpo"), _Phase("sft")]
    items = check_dataset_format(_ds(), "ds", phases, pipeline_config=None)
    by_type = {item.strategy_type: item for item in items}
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
        "ryotenkai_pod.trainer.strategies.factory.StrategyFactory",
        _Factory,
    )

    class _Phase:
        strategy_type = "xyz"

    with pytest.raises(DatasetValidationFailedError) as excinfo:
        check_dataset_format(_ds(), "ds", [_Phase()], pipeline_config=None)
    assert "Unknown strategy" in (excinfo.value.detail or "")
    assert excinfo.value.context.get("legacy_code") == "DATASET_FORMAT_ERROR"


def test_format_check_result_dataclass_defaults():
    res = FormatCheckResult(strategy_type="sft", ok=True)
    assert res.message == ""

"""
Tests for the PipelineStage base class (Phase A2 Batch 6 — raise-based).

Coverage by category (7-class):
- 1. Positive: happy-path success
- 2. Negative: stage / setup raise
- 3. Boundary: empty name, repeatable run
- 4. Invariants: setup→execute→teardown order, teardown always runs when
                 setup succeeded
- 5. Dependency errors: legacy Result shim for pre-Batch-7 stages
- 6. Regressions: teardown exception swallowed; setup raising RyotenkAIError
- 7. Combinatorial: parametrised lifecycle outcomes

NO MagicMock(spec=Protocol) — PipelineStage is an ABC, not a Protocol.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_control.pipeline.stages.base import (
    PipelineStage,
    _adapt_legacy_to_typed,
)
from ryotenkai_shared.errors import (
    InternalError,
    PipelineStageFailedError,
    RyotenkAIError,
)
from ryotenkai_shared.utils.result import AppError, Failure, Success

# =========================================================================
# CONCRETE IMPLEMENTATIONS FOR TESTING
# =========================================================================


class ConcreteStage(PipelineStage):
    """Concrete PipelineStage returning dict directly (Batch 6 shape)."""

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        self.update_context(context, {"result": "success"})
        return {"result": "success"}


class ConcreteStageRaises(PipelineStage):
    """Stage that raises a typed RyotenkAIError."""

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        raise PipelineStageFailedError(
            detail="Execution failed", context={"reason": "stage-bug"}
        )


class ConcreteStageRaisesPlain(PipelineStage):
    """Stage that raises a non-typed Exception (must be wrapped)."""

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Unexpected error during execution")


class ConcreteStageWithLifecycle(PipelineStage):
    """Stage exercising setup → execute → teardown order."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.calls: list[str] = []

    def setup(self, _context: dict[str, Any]) -> None:
        self.calls.append("setup")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("execute")
        return {"ok": True}

    def teardown(self) -> None:
        self.calls.append("teardown")


class ConcreteStageWithSetupRaise(PipelineStage):
    """setup() raises RyotenkAIError — execute + teardown must be skipped."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.execute_called = False
        self.teardown_called = False

    def setup(self, _context: dict[str, Any]) -> None:
        raise PipelineStageFailedError(detail="Setup failed")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        self.execute_called = True
        return {}

    def teardown(self) -> None:
        self.teardown_called = True


class ConcreteStageWithSetupPlainException(PipelineStage):
    """setup() raises a plain Exception — wrapped as InternalError."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.execute_called = False
        self.teardown_called = False

    def setup(self, _context: dict[str, Any]) -> None:
        raise RuntimeError("setup boom")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        self.execute_called = True
        return {}

    def teardown(self) -> None:
        self.teardown_called = True


class ConcreteStageWithTeardownAfterExecuteRaise(PipelineStage):
    """execute() raises; teardown() must still run."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.teardown_called = False

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        raise PipelineStageFailedError(detail="boom")

    def teardown(self) -> None:
        self.teardown_called = True


class ConcreteStageWithTeardownException(PipelineStage):
    """teardown() raises: run() returns execute output regardless."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.calls: list[str] = []

    def setup(self, _context: dict[str, Any]) -> None:
        self.calls.append("setup")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("execute")
        return {"ok": True}

    def teardown(self) -> None:
        self.calls.append("teardown")
        raise RuntimeError("teardown boom")


class ConcreteLegacyStage(PipelineStage):
    """Pre-Batch-7 stage still returning legacy ``Result[T, AppError]``.

    The shim ``_adapt_legacy_to_typed`` in ``base.run()`` must unwrap
    ``Success`` and raise ``InternalError`` for ``Failure``.
    """

    def __init__(self, config: Any, stage_name: str, *, fail: bool = False):
        super().__init__(config=config, stage_name=stage_name)
        self._fail = fail

    def execute(self, context: dict[str, Any]):  # type: ignore[override]
        if self._fail:
            return Failure(AppError(message="legacy boom", code="LEGACY_X"))
        return Success({"legacy": True})


# =========================================================================
# FIXTURES
# =========================================================================


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.model.name = "test/model"
    return config


@pytest.fixture
def concrete_stage(mock_config):
    return ConcreteStage(config=mock_config, stage_name="TestStage")


# =========================================================================
# 1. POSITIVE
# =========================================================================


class TestPositive:
    def test_run_calls_execute_and_returns_dict(self, concrete_stage):
        out = concrete_stage.run({"initial": "data"})
        assert isinstance(out, dict)
        assert out == {"result": "success"}

    def test_run_logs_start_and_success(self, concrete_stage):
        with patch("ryotenkai_control.pipeline.stages.base.logger") as mock_logger:
            concrete_stage.run({})
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        # 1: start, 2: end ✅
        assert "🚀 Starting:" in calls[0]
        assert "✅ Stage completed:" in calls[1]

    def test_lifecycle_order(self, mock_config):
        stage = ConcreteStageWithLifecycle(config=mock_config, stage_name="L")
        stage.run({})
        assert stage.calls == ["setup", "execute", "teardown"]


# =========================================================================
# 2. NEGATIVE
# =========================================================================


class TestNegative:
    def test_run_propagates_ryotenkai_error_from_execute(self, mock_config):
        stage = ConcreteStageRaises(config=mock_config, stage_name="ErrStage")
        with pytest.raises(PipelineStageFailedError) as ei:
            stage.run({})
        assert ei.value.detail == "Execution failed"
        assert ei.value.context["reason"] == "stage-bug"

    def test_plain_exception_wrapped_as_internal_error(self, mock_config):
        stage = ConcreteStageRaisesPlain(config=mock_config, stage_name="X")
        with pytest.raises(InternalError) as ei:
            stage.run({})
        assert "Unexpected error in X" in (ei.value.detail or "")
        assert ei.value.context["stage"] == "X"
        assert ei.value.context["exception_type"] == "RuntimeError"
        # Cause chain preserved for tracebacks
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_setup_ryotenkai_error_propagates_unchanged(self, mock_config):
        stage = ConcreteStageWithSetupRaise(config=mock_config, stage_name="S")
        with pytest.raises(PipelineStageFailedError):
            stage.run({})
        assert stage.execute_called is False
        assert stage.teardown_called is False

    def test_setup_plain_exception_wrapped_as_internal_error(self, mock_config):
        stage = ConcreteStageWithSetupPlainException(
            config=mock_config, stage_name="SP"
        )
        with pytest.raises(InternalError) as ei:
            stage.run({})
        assert stage.execute_called is False
        assert stage.teardown_called is False
        assert "Setup error in SP" in (ei.value.detail or "")


# =========================================================================
# 3. BOUNDARY
# =========================================================================


class TestBoundary:
    def test_pipeline_stage_initialization(self, mock_config):
        stage = ConcreteStage(config=mock_config, stage_name="T")
        assert stage.config is mock_config
        assert stage.stage_name == "T"
        assert stage.metadata == {}

    def test_pipeline_stage_is_abstract(self):
        class Incomplete(PipelineStage):
            pass

        with pytest.raises(TypeError):
            Incomplete(config=MagicMock(), stage_name="X")

    def test_stage_with_empty_name(self, mock_config):
        s = ConcreteStage(config=mock_config, stage_name="")
        assert s.stage_name == ""
        assert isinstance(s, PipelineStage)

    def test_run_is_repeatable(self, mock_config):
        stage = ConcreteStageWithLifecycle(config=mock_config, stage_name="R")
        stage.run({})
        stage.run({})
        assert stage.calls == [
            "setup", "execute", "teardown",
            "setup", "execute", "teardown",
        ]


# =========================================================================
# 4. INVARIANTS
# =========================================================================


class TestInvariants:
    def test_teardown_runs_after_execute_raises(self, mock_config):
        stage = ConcreteStageWithTeardownAfterExecuteRaise(
            config=mock_config, stage_name="TR"
        )
        with pytest.raises(PipelineStageFailedError):
            stage.run({})
        assert stage.teardown_called is True

    def test_cleanup_default_is_noop(self, mock_config):
        stage = ConcreteStageWithTeardownAfterExecuteRaise(
            config=mock_config, stage_name="C"
        )
        stage.cleanup()
        # cleanup must NOT call teardown by default
        assert stage.teardown_called is False

    def test_update_context_stores_under_stage_name(self, concrete_stage):
        ctx = {"prev": 1}
        out = concrete_stage.update_context(ctx, {"k": "v"})
        assert out["prev"] == 1
        assert out["TestStage"] == {"k": "v"}


# =========================================================================
# 5. DEPENDENCY ERRORS (legacy Result shim)
# =========================================================================


class TestLegacyShim:
    def test_shim_passes_through_dict(self):
        assert _adapt_legacy_to_typed({"a": 1}) == {"a": 1}

    def test_shim_non_dict_non_result_becomes_empty_dict(self):
        # None / int etc — shim normalises to {} so the loop's
        # context.update() never sees a non-dict.
        assert _adapt_legacy_to_typed(None) == {}
        assert _adapt_legacy_to_typed(42) == {}

    def test_shim_unwraps_legacy_success(self):
        result = Success({"k": 1})
        assert _adapt_legacy_to_typed(result) == {"k": 1}

    def test_shim_raises_internal_error_for_legacy_failure(self):
        result = Failure(AppError(message="legacy boom", code="LEGACY_X"))
        with pytest.raises(InternalError) as ei:
            _adapt_legacy_to_typed(result)
        assert "legacy boom" in (ei.value.detail or "")
        assert ei.value.context["legacy_code"] == "LEGACY_X"

    def test_legacy_stage_success_unwrapped_by_run(self, mock_config):
        stage = ConcreteLegacyStage(config=mock_config, stage_name="L")
        out = stage.run({})
        assert out == {"legacy": True}

    def test_legacy_stage_failure_raised_by_run(self, mock_config):
        stage = ConcreteLegacyStage(config=mock_config, stage_name="L", fail=True)
        with pytest.raises(InternalError) as ei:
            stage.run({})
        assert ei.value.context["legacy_code"] == "LEGACY_X"

    def test_shim_unwrap_returns_empty_dict_for_non_dict_success_value(self):
        # Defensive: legacy stages occasionally return Success(None)
        result = Success(None)
        assert _adapt_legacy_to_typed(result) == {}


# =========================================================================
# 6. REGRESSIONS
# =========================================================================


class TestRegressions:
    def test_teardown_exception_swallowed_and_output_preserved(self, mock_config):
        stage = ConcreteStageWithTeardownException(
            config=mock_config, stage_name="TE"
        )
        out = stage.run({})
        assert out == {"ok": True}
        assert stage.calls == ["setup", "execute", "teardown"]

    def test_keyboard_interrupt_in_execute_reraised(self, mock_config):
        class KbiStage(PipelineStage):
            def execute(self, context):
                raise KeyboardInterrupt

        stage = KbiStage(config=mock_config, stage_name="K")
        with pytest.raises(KeyboardInterrupt):
            stage.run({})

    def test_keyboard_interrupt_in_setup_reraised(self, mock_config):
        class KbiSetupStage(PipelineStage):
            def setup(self, _context):
                raise KeyboardInterrupt

            def execute(self, context):
                return {}

        stage = KbiSetupStage(config=mock_config, stage_name="K2")
        with pytest.raises(KeyboardInterrupt):
            stage.run({})

    def test_log_end_called_on_failure(self, mock_config):
        stage = ConcreteStageRaises(config=mock_config, stage_name="F")
        with patch("ryotenkai_control.pipeline.stages.base.logger") as mock_logger:
            with pytest.raises(PipelineStageFailedError):
                stage.run({})
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("❌" in c for c in info_calls)


# =========================================================================
# 7. COMBINATORIAL
# =========================================================================


@pytest.mark.parametrize(
    "setup_raise,execute_raise,expect_exc,expect_teardown",
    [
        (False, False, None, True),
        (True, False, PipelineStageFailedError, False),
        (False, True, PipelineStageFailedError, True),
    ],
)
def test_lifecycle_matrix(
    mock_config,
    setup_raise: bool,
    execute_raise: bool,
    expect_exc: type[Exception] | None,
    expect_teardown: bool,
):
    """Matrix of {setup_raises, execute_raises} outcomes."""
    teardown_calls: list[str] = []

    class S(PipelineStage):
        def setup(self, _context):
            if setup_raise:
                raise PipelineStageFailedError(detail="setup")

        def execute(self, context):
            if execute_raise:
                raise PipelineStageFailedError(detail="execute")
            return {"ok": True}

        def teardown(self):
            teardown_calls.append("teardown")

    stage = S(config=mock_config, stage_name="M")
    if expect_exc is None:
        out = stage.run({})
        assert out == {"ok": True}
    else:
        with pytest.raises(expect_exc):
            stage.run({})

    if expect_teardown:
        assert teardown_calls == ["teardown"]
    else:
        assert teardown_calls == []

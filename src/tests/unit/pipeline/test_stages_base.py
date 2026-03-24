"""
Tests for the PipelineStage base class.

Covers:
- Stage initialization
- Logging (log_start, log_end)
- Context updates (update_context)
- run() lifecycle with execute()
- Exception handling
- Invariants
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.base import PipelineStage
from src.utils.result import Failure, Success

# =========================================================================
# CONCRETE IMPLEMENTATIONS FOR TESTING
# =========================================================================


class ConcreteStage(PipelineStage):
    """Concrete PipelineStage for testing."""

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        """Returns Success with updated context."""
        updated = self.update_context(context, {"result": "success"})
        return Success(updated)


class ConcreteStageWithError(PipelineStage):
    """PipelineStage that returns Failure."""

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        """Returns Failure."""
        return Failure("Execution failed")


class ConcreteStageWithException(PipelineStage):
    """PipelineStage that raises."""

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        """Raises an exception."""
        raise RuntimeError("Unexpected error during execution")


class ConcreteStageWithLifecycle(PipelineStage):
    """PipelineStage to exercise setup/teardown lifecycle."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.calls: list[str] = []

    def setup(self, context: dict[str, Any]) -> Success[None] | Failure[str]:
        self.calls.append("setup")
        return Success(None)

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        self.calls.append("execute")
        return Success(self.update_context(context, {"result": "ok"}))

    def teardown(self) -> None:
        self.calls.append("teardown")


class ConcreteStageWithSetupFailure(PipelineStage):
    """PipelineStage whose setup() fails (fail-fast)."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.execute_called = False
        self.teardown_called = False

    def setup(self, context: dict[str, Any]) -> Success[None] | Failure[str]:
        return Failure("Setup failed")

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        self.execute_called = True
        return Success(context)

    def teardown(self) -> None:
        self.teardown_called = True


class ConcreteStageWithTeardownOnException(PipelineStage):
    """PipelineStage where execute() raises; teardown() must still run."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.teardown_called = False

    def setup(self, context: dict[str, Any]) -> Success[None] | Failure[str]:
        return Success(None)

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        raise RuntimeError("boom")

    def teardown(self) -> None:
        self.teardown_called = True


class ConcreteStageWithSetupException(PipelineStage):
    """setup() raises: execute/teardown must not run."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.execute_called = False
        self.teardown_called = False

    def setup(self, context: dict[str, Any]) -> Success[None] | Failure[str]:
        raise RuntimeError("setup boom")

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        self.execute_called = True
        return Success(context)

    def teardown(self) -> None:
        self.teardown_called = True


class ConcreteStageWithTeardownException(PipelineStage):
    """teardown() raises: run() must not crash and keeps execute Result."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.calls: list[str] = []

    def setup(self, context: dict[str, Any]) -> Success[None] | Failure[str]:
        self.calls.append("setup")
        return Success(None)

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        self.calls.append("execute")
        return Success(self.update_context(context, {"ok": True}))

    def teardown(self) -> None:
        self.calls.append("teardown")
        raise RuntimeError("teardown boom")


class ConcreteStageWithTeardownTrackingOnFailure(PipelineStage):
    """execute() returns Failure: teardown must still run (invariant)."""

    def __init__(self, config: Any, stage_name: str):
        super().__init__(config=config, stage_name=stage_name)
        self.teardown_called = False

    def setup(self, context: dict[str, Any]) -> Success[None] | Failure[str]:
        return Success(None)

    def execute(self, context: dict[str, Any]) -> Success[dict[str, Any]] | Failure[str]:
        return Failure("Execution failed")

    def teardown(self) -> None:
        self.teardown_called = True


# =========================================================================
# FIXTURES
# =========================================================================


@pytest.fixture
def mock_config():
    """Minimal mock config."""
    config = MagicMock()
    config.model.name = "test/model"
    return config


@pytest.fixture
def concrete_stage(mock_config):
    """Builds ConcreteStage instance."""
    return ConcreteStage(config=mock_config, stage_name="TestStage")


@pytest.fixture
def error_stage(mock_config):
    """Builds ConcreteStageWithError instance."""
    return ConcreteStageWithError(config=mock_config, stage_name="ErrorStage")


@pytest.fixture
def exception_stage(mock_config):
    """Builds ConcreteStageWithException instance."""
    return ConcreteStageWithException(config=mock_config, stage_name="ExceptionStage")


# =========================================================================
# INITIALIZATION TESTS
# =========================================================================


def test_pipeline_stage_initialization(mock_config):
    """PipelineStage init with config and name."""
    stage = ConcreteStage(config=mock_config, stage_name="TestStage")

    assert stage.config is mock_config
    assert stage.stage_name == "TestStage"
    assert stage.metadata == {}
    assert isinstance(stage.metadata, dict)


def test_pipeline_stage_is_abstract():
    """PipelineStage requires execute() implementation."""

    class IncompleteStage(PipelineStage):
        pass  # No execute() implementation

    with pytest.raises(TypeError) as exc_info:
        IncompleteStage(config=MagicMock(), stage_name="Test")

    assert "abstract" in str(exc_info.value).lower()


def test_metadata_mutable(concrete_stage):
    """metadata can be updated after construction."""
    concrete_stage.metadata["key1"] = "value1"
    concrete_stage.metadata["key2"] = 42

    assert concrete_stage.metadata["key1"] == "value1"
    assert concrete_stage.metadata["key2"] == 42
    assert len(concrete_stage.metadata) == 2


# =========================================================================
# LOGGING TESTS
# =========================================================================


@patch("src.pipeline.stages.base.logger")
def test_log_start_logs_correctly(mock_logger, concrete_stage):
    """log_start logs stage start with expected format."""
    concrete_stage.log_start()

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "🚀" in call_args
    assert "Starting:" in call_args
    assert "TestStage" in call_args


@patch("src.pipeline.stages.base.logger")
def test_log_end_success(mock_logger, concrete_stage):
    """log_end logs success with checkmark."""
    concrete_stage.log_end(success=True)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "✅" in call_args
    assert "Stage completed:" in call_args
    assert "TestStage" in call_args


@patch("src.pipeline.stages.base.logger")
def test_log_end_failure(mock_logger, concrete_stage):
    """log_end logs failure with cross."""
    concrete_stage.log_end(success=False)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "❌" in call_args
    assert "Stage completed:" in call_args
    assert "TestStage" in call_args


# =========================================================================
# UPDATE_CONTEXT TESTS
# =========================================================================


def test_update_context_adds_data(concrete_stage):
    """update_context stores data under stage_name."""
    context = {"previous_stage": "data"}
    updates = {"result": "success", "value": 42}

    updated = concrete_stage.update_context(context, updates)

    assert updated["previous_stage"] == "data"
    assert updated["TestStage"] == updates
    assert updated["TestStage"]["result"] == "success"
    assert updated["TestStage"]["value"] == 42


def test_update_context_empty_context(concrete_stage):
    """update_context with empty context."""
    context = {}
    updates = {"data": "value"}

    updated = concrete_stage.update_context(context, updates)

    assert "TestStage" in updated
    assert updated["TestStage"] == updates


def test_update_context_overwrites_existing(concrete_stage):
    """update_context overwrites existing stage data."""
    context = {"TestStage": {"old": "data"}}
    updates = {"new": "data"}

    updated = concrete_stage.update_context(context, updates)

    assert updated["TestStage"] == updates
    assert "old" not in updated["TestStage"]
    assert updated["TestStage"]["new"] == "data"


# =========================================================================
# RUN() LIFECYCLE TESTS
# =========================================================================


@patch("src.pipeline.stages.base.logger")
def test_run_calls_execute_and_logs(mock_logger, concrete_stage):
    """run() calls execute() and logs start/end."""
    context = {"initial": "data"}

    result = concrete_stage.run(context)

    assert result.is_success()
    # Logging called twice: start and end
    assert mock_logger.info.call_count == 2

    # Call order
    calls = [call[0][0] for call in mock_logger.info.call_args_list]
    assert "🚀 Starting:" in calls[0]
    assert "✅ Stage completed:" in calls[1]


def test_run_success_returns_success_result(concrete_stage):
    """Successful execute() yields Success."""
    context = {"test": "data"}

    result = concrete_stage.run(context)

    assert result.is_success()
    result_value = result.unwrap()
    assert "TestStage" in result_value
    assert result_value["TestStage"]["result"] == "success"


def test_run_failure_propagates_error(error_stage):
    """Failed execute() returns Failure."""
    context = {}

    result = error_stage.run(context)

    assert result.is_failure()
    error_msg = str(result.unwrap_err())
    assert error_msg == "Execution failed"


@patch("src.pipeline.stages.base.logger")
def test_run_exception_handling(mock_logger, exception_stage):
    """Exception in execute() is caught as Err."""
    context = {}

    result = exception_stage.run(context)

    assert result.is_failure()
    error_msg = str(result.unwrap_err())
    assert "Unexpected error in ExceptionStage" in error_msg
    assert "Unexpected error during execution" in error_msg

    # Exception was logged
    mock_logger.exception.assert_called_once()


@patch("src.pipeline.stages.base.logger")
def test_run_always_calls_log_end(mock_logger, exception_stage):
    """log_end runs even on exception."""
    context = {}

    result = exception_stage.run(context)

    assert result.is_failure()

    # log_end called with success=False
    calls = [call[0][0] for call in mock_logger.info.call_args_list]
    assert any("❌" in call for call in calls)


# =========================================================================
# INVARIANT TESTS
# =========================================================================


@patch("src.pipeline.stages.base.logger")
def test_run_logging_order_invariant(mock_logger, concrete_stage):
    """log_start before execute; log_end after."""
    context = {}

    # Reset mock for accurate ordering
    mock_logger.reset_mock()

    result = concrete_stage.run(context)

    assert result.is_success()

    # Call order
    info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
    assert len(info_calls) == 2

    # First call: log_start
    assert "🚀 Starting:" in info_calls[0]
    # Second call: log_end
    assert "Stage completed:" in info_calls[1]


@patch("src.pipeline.stages.base.logger")
def test_run_logging_order_on_exception(mock_logger, exception_stage):
    """Logging order on exception."""
    context = {}

    mock_logger.reset_mock()

    result = exception_stage.run(context)

    assert result.is_failure()

    # log_start called
    info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
    assert any("🚀 Starting:" in call for call in info_calls)
    # log_end called
    assert any("❌" in call for call in info_calls)
    # exception logged
    mock_logger.exception.assert_called_once()


def test_run_calls_setup_execute_teardown_in_order(mock_config):
    """run() orders setup -> execute -> teardown."""
    stage = ConcreteStageWithLifecycle(config=mock_config, stage_name="LifecycleStage")
    ctx: dict[str, Any] = {}

    result = stage.run(ctx)

    assert result.is_success()
    assert stage.calls == ["setup", "execute", "teardown"]


def test_run_does_not_call_execute_or_teardown_when_setup_fails(mock_config):
    """If setup() returns Failure, execute/teardown are skipped."""
    stage = ConcreteStageWithSetupFailure(config=mock_config, stage_name="SetupFailStage")
    ctx: dict[str, Any] = {}

    result = stage.run(ctx)

    assert result.is_failure()
    assert stage.execute_called is False
    assert stage.teardown_called is False


def test_run_calls_teardown_on_execute_exception(mock_config):
    """teardown() runs even if execute() raises."""
    stage = ConcreteStageWithTeardownOnException(config=mock_config, stage_name="TeardownOnExceptionStage")
    ctx: dict[str, Any] = {}

    result = stage.run(ctx)

    assert result.is_failure()
    assert stage.teardown_called is True


def test_cleanup_default_is_noop(mock_config):
    """cleanup() does not call teardown() by default."""
    stage = ConcreteStageWithTeardownOnException(config=mock_config, stage_name="CleanupNoopStage")

    stage.cleanup()

    assert stage.teardown_called is False


@patch("src.pipeline.stages.base.logger")
def test_run_setup_exception_is_handled_and_no_execute_teardown(mock_logger, mock_config):
    """Negative: setup() raises → Err; execute/teardown skipped."""
    stage = ConcreteStageWithSetupException(config=mock_config, stage_name="SetupExceptionStage")
    ctx: dict[str, Any] = {}

    result = stage.run(ctx)

    assert result.is_failure()
    assert stage.execute_called is False
    assert stage.teardown_called is False
    mock_logger.exception.assert_called_once()


@patch("src.pipeline.stages.base.logger")
def test_run_teardown_exception_is_swallowed_and_result_preserved(mock_logger, mock_config):
    """Edge: teardown() raises → swallowed; execute Result preserved."""
    stage = ConcreteStageWithTeardownException(config=mock_config, stage_name="TeardownExceptionStage")
    ctx: dict[str, Any] = {}

    result = stage.run(ctx)

    assert result.is_success()
    assert stage.calls == ["setup", "execute", "teardown"]
    mock_logger.warning.assert_called_once()


def test_run_calls_teardown_when_execute_returns_failure(mock_config):
    """Invariant: teardown() runs even when execute() returned Failure."""
    stage = ConcreteStageWithTeardownTrackingOnFailure(config=mock_config, stage_name="FailureTeardownStage")
    ctx: dict[str, Any] = {}

    result = stage.run(ctx)

    assert result.is_failure()
    assert stage.teardown_called is True


def test_run_is_repeatable_no_cross_run_state(mock_config):
    """Regression/invariant: repeated run() does not depend on prior run state."""
    stage = ConcreteStageWithLifecycle(config=mock_config, stage_name="RepeatableStage")
    ctx1: dict[str, Any] = {}
    ctx2: dict[str, Any] = {}

    r1 = stage.run(ctx1)
    r2 = stage.run(ctx2)

    assert r1.is_success()
    assert r2.is_success()
    assert stage.calls == ["setup", "execute", "teardown", "setup", "execute", "teardown"]

# =========================================================================
# EDGE CASE TESTS
# =========================================================================


def test_stage_with_empty_name(mock_config):
    """Stage with empty name."""
    stage = ConcreteStage(config=mock_config, stage_name="")

    assert stage.stage_name == ""
    assert isinstance(stage, PipelineStage)


def test_run_with_none_context_via_run(concrete_stage):
    """run() handles bad context exceptions."""

    # Stage uses None as context
    class BadContextStage(PipelineStage):
        def execute(self, context):
            # Intentionally use None as dict
            context["key"] = "value"  # type: ignore
            return Success(context)

    bad_stage = BadContextStage(config=concrete_stage.config, stage_name="BadStage")

    # run() catches and returns Err
    result = bad_stage.run(None)  # type: ignore

    assert result.is_failure()
    error_msg = str(result.unwrap_err())
    assert "Unexpected error" in error_msg

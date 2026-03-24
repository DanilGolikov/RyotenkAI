"""
Tests for Result pattern and error types.

Verifies Result pattern functionality for explicit error handling.
"""

from __future__ import annotations

import pytest

from src.utils.result import (
    AppError,
    ConfigError,
    DatasetError,
    Failure,
    InferenceError,
    ModelError,
    OOMError,
    ProviderError,
    Result,
    StrategyError,
    Success,
    TrainingError,
    err,
    ok,
)

# =============================================================================
# TESTS: Basic Result Operations
# =============================================================================


class TestBasicResult:
    """Tests for basic Success/Failure operations."""

    def test_ok_creates_success(self):
        """ok() should create Success."""
        result = ok(42)

        assert isinstance(result, Success)
        assert result.is_success()
        assert not result.is_failure()
        assert result.unwrap() == 42

    def test_err_creates_failure(self):
        """err() should create Failure."""
        result = err("error message")

        assert isinstance(result, Failure)
        assert not result.is_success()
        assert result.is_failure()
        assert result.unwrap_err() == "error message"

    def test_success_unwrap_or_returns_value(self):
        """Success.unwrap_or() should return actual value."""
        result = ok(42)

        assert result.unwrap_or(0) == 42

    def test_failure_unwrap_or_returns_default(self):
        """Failure.unwrap_or() should return default."""
        result = err("error")

        assert result.unwrap_or(0) == 0

    def test_success_map(self):
        """Success.map() should transform value."""
        result = ok(10)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_success()
        assert mapped.unwrap() == 20

    def test_failure_map_returns_self(self):
        """Failure.map() should return itself unchanged."""
        result = err("error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_failure()
        assert mapped.unwrap_err() == "error"

    def test_success_unwrap_err_raises(self):
        """Success.unwrap_err() should raise."""
        result = ok(42)

        with pytest.raises(ValueError, match="Cannot unwrap_err on Success"):
            result.unwrap_err()

    def test_failure_unwrap_raises(self):
        """Failure.unwrap() should raise."""
        result = err("error")

        with pytest.raises(ValueError, match="Cannot unwrap Failure"):
            result.unwrap()


# =============================================================================
# TESTS: Error Types
# =============================================================================


class TestErrorTypes:
    """Tests for error type hierarchy."""

    def test_app_error_is_base(self):
        """AppError is the base for all domain errors."""
        app_err = AppError(message="Base error")

        assert app_err.message == "Base error"
        assert app_err.code == "APP_ERROR"
        assert "[APP_ERROR] Base error" == str(app_err)

    def test_app_error_to_log_dict(self):
        """AppError.to_log_dict() returns structured dict."""
        error = AppError(message="Something failed", code="CUSTOM_CODE", details={"key": "val"})

        log_dict = error.to_log_dict()

        assert log_dict == {"code": "CUSTOM_CODE", "message": "Something failed", "details": {"key": "val"}}

    def test_to_log_dict_with_none_details(self):
        """to_log_dict() works with None details."""
        error = AppError(message="err")
        log_dict = error.to_log_dict()

        assert log_dict["details"] is None

    def test_training_error_basic(self):
        """TrainingError should have message and code."""
        error = TrainingError("Something failed")

        assert error.message == "Something failed"
        assert error.code == "TRAINING_ERROR"
        assert "[TRAINING_ERROR]" in str(error)

    def test_training_error_is_app_error(self):
        """TrainingError should be an AppError."""
        error = TrainingError("failed")

        assert isinstance(error, AppError)

    def test_config_error_code(self):
        """ConfigError should have CONFIG_ERROR code and be under AppError (NOT TrainingError)."""
        error = ConfigError("Invalid config")

        assert error.code == "CONFIG_ERROR"
        assert isinstance(error, AppError)
        # ConfigError is now directly under AppError, NOT under TrainingError
        assert not isinstance(error, TrainingError)

    def test_dataset_error_code(self):
        """DatasetError should have DATASET_ERROR code."""
        error = DatasetError("Dataset not found")

        assert error.code == "DATASET_ERROR"
        assert isinstance(error, TrainingError)

    def test_model_error_code(self):
        """ModelError should have MODEL_ERROR code."""
        error = ModelError("Model loading failed")

        assert error.code == "MODEL_ERROR"
        assert isinstance(error, TrainingError)

    def test_strategy_error_code(self):
        """StrategyError should have STRATEGY_ERROR code."""
        error = StrategyError("Strategy failed")

        assert error.code == "STRATEGY_ERROR"

    def test_oom_error_code(self):
        """OOMError should have OOM_ERROR code."""
        error = OOMError("Out of memory")

        assert error.code == "OOM_ERROR"

    def test_provider_error_is_app_error(self):
        """ProviderError should be a direct AppError child."""
        error = ProviderError("SSH connection failed")

        assert error.code == "PROVIDER_ERROR"
        assert isinstance(error, AppError)
        assert not isinstance(error, TrainingError)

    def test_inference_error_is_app_error(self):
        """InferenceError should be a direct AppError child."""
        error = InferenceError("vLLM startup failed")

        assert error.code == "INFERENCE_ERROR"
        assert isinstance(error, AppError)
        assert not isinstance(error, TrainingError)

    def test_error_hierarchy_structure(self):
        """Verify full error hierarchy."""
        # AppError is root
        assert isinstance(AppError("x"), AppError)
        # Direct AppError children
        assert isinstance(ConfigError("x"), AppError)
        assert isinstance(ProviderError("x"), AppError)
        assert isinstance(InferenceError("x"), AppError)
        # TrainingError subtree
        assert isinstance(TrainingError("x"), AppError)
        assert isinstance(DatasetError("x"), AppError)
        assert isinstance(ModelError("x"), AppError)
        # TrainingError children are NOT ConfigError/ProviderError/InferenceError
        assert not isinstance(DatasetError("x"), ConfigError)
        assert not isinstance(DatasetError("x"), ProviderError)

    def test_error_with_details(self):
        """Error should store optional details."""
        error = TrainingError(
            message="Failed",
            details={"phase": 2, "reason": "OOM"},
        )

        assert error.details == {"phase": 2, "reason": "OOM"}


# =============================================================================
# TESTS: Result.from_exception()
# =============================================================================


class TestFromException:
    """Tests for Result.from_exception() helper."""

    def test_from_exception_success(self):
        """from_exception should wrap successful execution."""
        result = Result.from_exception(lambda: 1 + 1)

        assert result.is_success()
        assert result.unwrap() == 2

    def test_from_exception_failure(self):
        """from_exception should catch exceptions."""

        def failing_func():
            raise ValueError("Test error")

        result = Result.from_exception(failing_func)

        assert result.is_failure()
        error = result.unwrap_err()
        assert "Test error" in error.message
        assert error.details["exception_type"] == "ValueError"

    def test_from_exception_with_error_type(self):
        """from_exception should use specified error type."""

        def failing_func():
            raise RuntimeError("Runtime error")

        result = Result.from_exception(failing_func, error_type=ModelError)

        assert result.is_failure()
        error = result.unwrap_err()
        assert isinstance(error, ModelError)

    def test_from_exception_captures_traceback(self):
        """from_exception should capture traceback."""

        def failing_func():
            raise Exception("Error")

        result = Result.from_exception(failing_func)

        error = result.unwrap_err()
        assert "traceback" in error.details


# =============================================================================
# TESTS: Result.try_or_error()
# =============================================================================


class TestTryOrError:
    """Tests for Result.try_or_error() helper."""

    def test_try_or_error_success(self):
        """try_or_error should return Success on success."""
        result = Result.try_or_error(
            lambda: 42,
            "Operation failed",
        )

        assert result.is_success()
        assert result.unwrap() == 42

    def test_try_or_error_failure(self):
        """try_or_error should use custom message on failure."""
        result = Result.try_or_error(
            lambda: int("not a number"),
            "Conversion failed",
            ConfigError,
        )

        assert result.is_failure()
        error = result.unwrap_err()
        assert "Conversion failed" in error.message
        assert isinstance(error, ConfigError)


# =============================================================================
# TESTS: Result.collect()
# =============================================================================


class TestCollect:
    """Tests for Result.collect() helper."""

    def test_collect_all_success(self):
        """collect should return Success with all values."""
        results = [ok(1), ok(2), ok(3)]
        collected = Result.collect(results)

        assert collected.is_success()
        assert collected.unwrap() == [1, 2, 3]

    def test_collect_with_failure(self):
        """collect should return first Failure."""
        results = [ok(1), err(TrainingError("fail")), ok(3)]
        collected = Result.collect(results)

        assert collected.is_failure()
        assert "fail" in collected.unwrap_err().message

    def test_collect_empty_list(self):
        """collect should return empty list for empty input."""
        results: list = []
        collected = Result.collect(results)

        assert collected.is_success()
        assert collected.unwrap() == []


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_result_with_none_value(self):
        """Success can wrap None."""
        result = ok(None)

        assert result.is_success()
        assert result.unwrap() is None

    def test_result_with_empty_string_error(self):
        """Failure can have empty error."""
        result = err("")

        assert result.is_failure()
        assert result.unwrap_err() == ""

    def test_nested_result(self):
        """Result can wrap another Result."""
        inner = ok(42)
        outer = ok(inner)

        assert outer.is_success()
        unwrapped = outer.unwrap()
        assert unwrapped.is_success()
        assert unwrapped.unwrap() == 42

    def test_error_immutability(self):
        """TrainingError should be immutable (frozen)."""
        error = TrainingError("Test")

        with pytest.raises(AttributeError):
            error.message = "Changed"  # type: ignore


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for Result pattern usage."""

    def test_chained_operations(self):
        """Test chaining multiple Result operations."""

        def parse_number(s: str) -> Success[int] | Failure[TrainingError]:
            return Result.from_exception(
                lambda: int(s),
                error_type=ConfigError,
            )

        def double(x: int) -> Success[int] | Failure[TrainingError]:
            return ok(x * 2)

        # Success path
        result1 = parse_number("21")
        if result1.is_success():
            result2 = double(result1.unwrap())
            assert result2.unwrap() == 42

        # Failure path
        result3 = parse_number("not a number")
        assert result3.is_failure()

    def test_real_world_config_loading(self):
        """Test Result pattern for config loading scenario."""

        def load_config(path: str) -> Success[dict] | Failure[ConfigError]:
            """Simulate config loading."""
            if path == "valid.yaml":
                return ok({"model": "qwen2.5"})
            return err(ConfigError(f"Config not found: {path}"))

        # Success case
        result = load_config("valid.yaml")
        assert result.is_success()
        assert result.unwrap()["model"] == "qwen2.5"

        # Failure case
        result = load_config("invalid.yaml")
        assert result.is_failure()
        assert "not found" in result.unwrap_err().message

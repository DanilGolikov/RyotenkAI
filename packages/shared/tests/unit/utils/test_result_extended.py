"""
Additional tests for Result pattern to improve coverage.

Target: Increase result.py coverage from 63.92% to >80%.

Tests cover:
- Rust-style aliases (is_ok, is_err)
- Failure methods (unwrap_or, map)
- ResultHelpers edge cases
- Error code customization
"""

import pytest

from src.utils.result import (
    AppError,
    ConfigError,
    DatasetError,
    InferenceError,
    ModelError,
    OOMError,
    ProviderError,
    Result,
    ResultHelpers,
    StrategyError,
    TrainingError,
    err,
    ok,
)

# =============================================================================
# TESTS: Rust-style API Aliases
# =============================================================================


class TestRustStyleAliases:
    """Test Rust-style method aliases (is_ok, is_err)."""

    def test_success_is_ok(self):
        """Success.is_ok() should return True."""
        result = ok(42)

        assert result.is_ok() is True
        assert result.is_err() is False

    def test_failure_is_ok(self):
        """Failure.is_ok() should return False."""
        result = err(TrainingError("fail"))

        assert result.is_ok() is False
        assert result.is_err() is True

    def test_success_is_err(self):
        """Success.is_err() should return False."""
        result = ok(42)

        assert result.is_err() is False

    def test_failure_is_err(self):
        """Failure.is_err() should return True."""
        result = err(TrainingError("fail"))

        assert result.is_err() is True


# =============================================================================
# TESTS: Failure Methods Coverage
# =============================================================================


class TestFailureMethods:
    """Test Failure-specific methods."""

    def test_failure_unwrap_or(self):
        """Failure.unwrap_or() should return default."""
        result: Result[int, TrainingError] = err(TrainingError("error"))

        value = result.unwrap_or(999)

        assert value == 999

    def test_failure_map(self):
        """Failure.map() should return self unchanged."""
        result: Result[int, TrainingError] = err(TrainingError("error"))

        mapped = result.map(lambda x: x * 2)

        assert mapped.is_failure()
        assert mapped.unwrap_err().message == "error"

    def test_failure_unwrap_raises(self):
        """Failure.unwrap() should raise ValueError."""
        result = err(TrainingError("fail"))

        with pytest.raises(ValueError, match="Cannot unwrap Failure"):
            result.unwrap()

    def test_success_unwrap_err_raises(self):
        """Success.unwrap_err() should raise ValueError."""
        result = ok(42)

        with pytest.raises(ValueError, match="Cannot unwrap_err on Success"):
            result.unwrap_err()


# =============================================================================
# TESTS: ResultHelpers Edge Cases
# =============================================================================


class TestResultHelpersEdgeCases:
    """Test ResultHelpers edge cases and error handling."""

    def test_from_exception_captures_exception_type(self):
        """from_exception should capture exception type in details."""

        def raise_value_error():
            raise ValueError("Test value error")

        result = ResultHelpers.from_exception(raise_value_error, error_type=ConfigError)

        assert result.is_failure()
        error = result.unwrap_err()
        assert error.details["exception_type"] == "ValueError"
        assert "traceback" in error.details

    def test_from_exception_with_custom_error_code(self):
        """from_exception should use custom error_code if provided."""

        def raise_error():
            raise Exception("Error")

        result = ResultHelpers.from_exception(raise_error, error_type=TrainingError, error_code="CUSTOM_ERROR")

        assert result.is_failure()
        error = result.unwrap_err()
        assert error.code == "CUSTOM_ERROR"

    def test_try_or_error_includes_original_exception(self):
        """try_or_error should include original exception in details."""

        def raise_runtime_error():
            raise RuntimeError("Runtime error details")

        result = ResultHelpers.try_or_error(raise_runtime_error, "Operation failed", DatasetError)

        assert result.is_failure()
        error = result.unwrap_err()
        assert "Operation failed" in error.message
        assert "original_exception" in error.details
        assert "Runtime error details" in error.details["original_exception"]

    def test_collect_empty_list(self):
        """collect() should return Success([]) for empty list."""
        results: list[Result] = []

        collected = ResultHelpers.collect(results)

        assert collected.is_success()
        assert collected.unwrap() == []

    def test_collect_first_failure(self):
        """collect() should return first Failure encountered."""
        results = [ok(1), err(TrainingError("first error")), err(TrainingError("second error")), ok(3)]

        collected = ResultHelpers.collect(results)

        assert collected.is_failure()
        assert "first error" in collected.unwrap_err().message

    def test_collect_all_success(self):
        """collect() should collect all values if all Success."""
        results = [ok(10), ok(20), ok(30)]

        collected = ResultHelpers.collect(results)

        assert collected.is_success()
        assert collected.unwrap() == [10, 20, 30]


# =============================================================================
# TESTS: Error Code Customization
# =============================================================================


class TestErrorCodeCustomization:
    """Test error code generation and customization."""

    def test_default_error_codes(self):
        """Each error type should have correct default code."""
        errors = [
            (AppError("test"), "APP_ERROR"),
            (TrainingError("test"), "TRAINING_ERROR"),
            (ConfigError("test"), "CONFIG_ERROR"),
            (DatasetError("test"), "DATASET_ERROR"),
            (ModelError("test"), "MODEL_ERROR"),
            (StrategyError("test"), "STRATEGY_ERROR"),
            (OOMError("test"), "OOM_ERROR"),
            (ProviderError("test"), "PROVIDER_ERROR"),
            (InferenceError("test"), "INFERENCE_ERROR"),
        ]

        for error, expected_code in errors:
            assert error.code == expected_code

    def test_error_str_includes_code(self):
        """Error.__str__() should include code."""
        error = ConfigError("Invalid config")

        error_str = str(error)

        assert "[CONFIG_ERROR]" in error_str
        assert "Invalid config" in error_str

    def test_custom_error_code_in_from_exception(self):
        """from_exception should allow custom error codes."""

        def failing():
            raise Exception("fail")

        result = ResultHelpers.from_exception(failing, error_type=ModelError, error_code="MODEL_LOAD_FAILED")

        error = result.unwrap_err()
        assert error.code == "MODEL_LOAD_FAILED"


# =============================================================================
# TESTS: Error Details
# =============================================================================


class TestErrorDetails:
    """Test error details handling."""

    def test_error_with_none_details(self):
        """Error can have None details."""
        error = TrainingError("Message")

        assert error.details is None

    def test_error_with_empty_details(self):
        """Error can have empty dict details."""
        error = TrainingError("Message", details={})

        assert error.details == {}

    def test_error_with_complex_details(self):
        """Error can have complex details dict."""
        details = {"phase": 2, "epoch": 5, "batch": 100, "loss": 0.5, "metadata": {"gpu": "A100", "batch_size": 32}}
        error = StrategyError("Training failed", details=details)

        assert error.details == details
        assert error.details["metadata"]["gpu"] == "A100"


# =============================================================================
# TESTS: Result Pattern in Real Scenarios
# =============================================================================


class TestRealWorldScenarios:
    """Test Result pattern in realistic scenarios."""

    def test_dataset_loading_pipeline(self):
        """Test Result pattern for dataset loading."""

        def load_dataset(path: str) -> Result[dict, DatasetError]:
            if not path.endswith(".jsonl"):
                return err(
                    DatasetError(f"Invalid format: {path}", details={"expected": ".jsonl", "got": path.split(".")[-1]})
                )
            return ok({"path": path, "loaded": True})

        # Success case
        result1 = load_dataset("train.jsonl")
        assert result1.is_success()
        assert result1.unwrap()["loaded"] is True

        # Failure case
        result2 = load_dataset("train.csv")
        assert result2.is_failure()
        assert "Invalid format" in result2.unwrap_err().message
        assert result2.unwrap_err().details["expected"] == ".jsonl"

    def test_model_loading_with_error_recovery(self):
        """Test error recovery in model loading."""

        def load_model(name: str, fallback: str | None = None) -> Result[str, ModelError]:
            if name == "invalid":
                if fallback:
                    return ok(f"Loaded fallback: {fallback}")
                return err(ModelError(f"Model not found: {name}"))
            return ok(f"Loaded: {name}")

        # Try with fallback
        result = load_model("invalid", fallback="default-model")
        assert result.is_success()
        assert "fallback" in result.unwrap()

        # Without fallback
        result2 = load_model("invalid", fallback=None)
        assert result2.is_failure()

    def test_chained_operations_with_error_propagation(self):
        """Test chaining operations with error propagation."""

        def step1() -> Result[int, ConfigError]:
            return ok(10)

        def step2(x: int) -> Result[int, ConfigError]:
            if x < 5:
                return err(ConfigError("Value too small"))
            return ok(x * 2)

        def step3(x: int) -> Result[int, ConfigError]:
            return ok(x + 5)

        # Success chain
        r1 = step1()
        if r1.is_success():
            r2 = step2(r1.unwrap())
            if r2.is_success():
                r3 = step3(r2.unwrap())
                assert r3.is_success()
                assert r3.unwrap() == 25  # (10*2)+5

        # Failure chain
        r1_fail = ok(2)
        r2_fail = step2(r1_fail.unwrap())
        assert r2_fail.is_failure()
        assert "too small" in r2_fail.unwrap_err().message


# =============================================================================
# TESTS: Type Safety
# =============================================================================


class TestTypeSafety:
    """Test type annotations and generic behavior."""

    def test_success_generic_type(self):
        """Success should preserve type."""
        int_result: Result[int, TrainingError] = ok(42)
        str_result: Result[str, TrainingError] = ok("hello")
        list_result: Result[list[int], TrainingError] = ok([1, 2, 3])

        assert isinstance(int_result.unwrap(), int)
        assert isinstance(str_result.unwrap(), str)
        assert isinstance(list_result.unwrap(), list)

    def test_failure_generic_type(self):
        """Failure should preserve error type."""
        config_err: Result[int, ConfigError] = err(ConfigError("fail"))
        model_err: Result[int, ModelError] = err(ModelError("fail"))

        assert isinstance(config_err.unwrap_err(), ConfigError)
        assert isinstance(model_err.unwrap_err(), ModelError)


# =============================================================================
# TESTS: repr() and str()
# =============================================================================


class TestStringRepresentation:
    """Test string representations."""

    def test_success_repr(self):
        """Success.__repr__() should show value."""
        result = ok(42)

        repr_str = repr(result)

        assert "Success" in repr_str
        assert "42" in repr_str

    def test_failure_repr(self):
        """Failure.__repr__() should show error."""
        result = err(TrainingError("error message"))

        repr_str = repr(result)

        assert "Failure" in repr_str
        assert "error message" in repr_str

    def test_error_str_format(self):
        """TrainingError.__str__() should be formatted."""
        error = OOMError("Out of memory during training")

        error_str = str(error)

        assert error_str == "[OOM_ERROR] Out of memory during training"


# =============================================================================
# TESTS: AppError Base Class
# =============================================================================


class TestAppErrorBase:
    """Test AppError as the universal base class."""

    def test_all_errors_are_app_errors(self):
        """All domain error types must inherit from AppError."""
        assert isinstance(TrainingError("x"), AppError)
        assert isinstance(ConfigError("x"), AppError)
        assert isinstance(DatasetError("x"), AppError)
        assert isinstance(ModelError("x"), AppError)
        assert isinstance(StrategyError("x"), AppError)
        assert isinstance(OOMError("x"), AppError)
        assert isinstance(ProviderError("x"), AppError)
        assert isinstance(InferenceError("x"), AppError)

    def test_config_error_not_under_training_error(self):
        """ConfigError is under AppError directly, NOT TrainingError."""
        error = ConfigError("bad config")

        assert isinstance(error, AppError)
        assert not isinstance(error, TrainingError)

    def test_provider_error_not_under_training_error(self):
        """ProviderError is independent from TrainingError."""
        error = ProviderError("SSH timeout")

        assert isinstance(error, AppError)
        assert not isinstance(error, TrainingError)

    def test_inference_error_not_under_training_error(self):
        """InferenceError is independent from TrainingError."""
        error = InferenceError("vLLM crashed")

        assert isinstance(error, AppError)
        assert not isinstance(error, TrainingError)


# =============================================================================
# TESTS: to_log_dict Structured Logging
# =============================================================================


class TestToLogDict:
    """Test AppError.to_log_dict() for structured logging."""

    def test_to_log_dict_keys(self):
        """to_log_dict() must return code, message, details keys."""
        error = AppError(message="Something went wrong", code="APP_ERROR")
        log_dict = error.to_log_dict()

        assert set(log_dict.keys()) == {"code", "message", "details"}

    def test_to_log_dict_without_details(self):
        """to_log_dict() details is None when not provided."""
        error = ProviderError("SSH failed")
        log_dict = error.to_log_dict()

        assert log_dict["code"] == "PROVIDER_ERROR"
        assert log_dict["message"] == "SSH failed"
        assert log_dict["details"] is None

    def test_to_log_dict_with_details(self):
        """to_log_dict() includes details when provided."""
        error = InferenceError("startup failed", details={"port": 8080, "timeout": 30})
        log_dict = error.to_log_dict()

        assert log_dict["details"] == {"port": 8080, "timeout": 30}

    def test_to_log_dict_is_serializable(self):
        """to_log_dict() output should be JSON-serializable."""
        import json

        error = ConfigError("missing key", details={"key": "model_name", "file": "config.yaml"})
        log_dict = error.to_log_dict()

        serialized = json.dumps(log_dict)
        assert isinstance(serialized, str)

    def test_to_log_dict_subclasses_use_own_code(self):
        """Subclasses report their own code in to_log_dict."""
        errors_and_codes = [
            (TrainingError("x"), "TRAINING_ERROR"),
            (DatasetError("x"), "DATASET_ERROR"),
            (ModelError("x"), "MODEL_ERROR"),
            (ConfigError("x"), "CONFIG_ERROR"),
            (ProviderError("x"), "PROVIDER_ERROR"),
            (InferenceError("x"), "INFERENCE_ERROR"),
        ]
        for error, expected_code in errors_and_codes:
            assert error.to_log_dict()["code"] == expected_code

"""
Result pattern for explicit error handling.

Inspired by Rust's Result type, provides explicit error handling
without exceptions for recoverable errors.

Features:
- Type-safe Success/Failure values
- Error type hierarchy for categorized errors
- from_exception() for wrapping try/except blocks
- Chainable operations (map, and_then)

Usage:
    result = do_something()
    if result.is_success():
        value = result.unwrap()
    else:
        error = result.unwrap_err()

    # With from_exception
    result = Result.from_exception(lambda: risky_operation())
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

T = TypeVar("T")  # Success value type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Mapped success value type


# =============================================================================
# ERROR TYPES
# =============================================================================


@dataclass(frozen=True)
class AppError:
    """
    Base error type for the entire application.

    All domain-specific errors inherit from this.
    Provides a single type to use in Result[T, AppError] signatures.

    Attributes:
        message: Human-readable error message
        code: Error code for categorization
        details: Optional additional context
    """

    message: str
    code: str = "APP_ERROR"
    details: dict[str, Any] | None = field(default=None)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_log_dict(self) -> dict[str, Any]:
        """Return structured dict suitable for logger extra= or JSON logging."""
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass(frozen=True)
class ConfigError(AppError):
    """Error in configuration validation or loading (universal across all flows)."""

    code: str = "CONFIG_ERROR"


@dataclass(frozen=True)
class ConfigDriftError(ConfigError):
    """Configuration drift detected for an existing logical run."""

    code: str = "CONFIG_DRIFT"


@dataclass(frozen=True)
class TrainingError(AppError):
    """
    Base error type for training pipeline.

    All training-specific errors inherit from this.
    """

    code: str = "TRAINING_ERROR"


@dataclass(frozen=True)
class DatasetError(TrainingError):
    """Error in dataset loading or processing."""

    code: str = "DATASET_ERROR"


@dataclass(frozen=True)
class ModelError(TrainingError):
    """Error in model loading or inference."""

    code: str = "MODEL_ERROR"


@dataclass(frozen=True)
class StrategyError(TrainingError):
    """Error in training strategy execution."""

    code: str = "STRATEGY_ERROR"


@dataclass(frozen=True)
class OOMError(TrainingError):
    """Out of memory error during training."""

    code: str = "OOM_ERROR"


@dataclass(frozen=True)
class DataLoaderError(TrainingError):
    """Error in data loading, validation or transformation."""

    code: str = "DATA_LOADER_ERROR"


@dataclass(frozen=True)
class ProviderError(AppError):
    """Error in infrastructure provider (SSH, Docker, RunPod, pod lifecycle)."""

    code: str = "PROVIDER_ERROR"


@dataclass(frozen=True)
class InferenceError(AppError):
    """Error in inference serving (vLLM startup, health checks, endpoints)."""

    code: str = "INFERENCE_ERROR"


@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful result"""

    value: T

    @staticmethod
    def is_success() -> Literal[True]:
        return True

    @staticmethod
    def is_failure() -> Literal[False]:
        return False

    # Aliases for Rust-style API
    @staticmethod
    def is_ok() -> Literal[True]:
        return True

    @staticmethod
    def is_err() -> Literal[False]:
        return False

    def unwrap(self) -> T:
        """Get the success value"""
        return self.value

    def unwrap_err(self) -> NoReturn:
        """Raise error - Success has no error"""
        raise ValueError("Cannot unwrap_err on Success")

    def unwrap_or(self, _default: T) -> T:
        """Get value or default"""
        return self.value

    def map(self, func: Callable[[T], U]) -> Result[U, E]:
        """Map the success value"""
        return Success(func(self.value))

    def __repr__(self) -> str:
        return f"Success({self.value!r})"


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed result"""

    error: E

    @staticmethod
    def is_success() -> Literal[False]:
        return False

    @staticmethod
    def is_failure() -> Literal[True]:
        return True

    # Aliases for Rust-style API
    @staticmethod
    def is_ok() -> Literal[False]:
        return False

    @staticmethod
    def is_err() -> Literal[True]:
        return True

    def unwrap(self) -> NoReturn:
        """Raise error on unwrap"""
        raise ValueError(f"Cannot unwrap Failure: {self.error}")

    def unwrap_err(self) -> E:
        """Get the error value"""
        return self.error

    @staticmethod
    def unwrap_or(default: T) -> T:
        """Get default value"""
        return default

    def map(self, _func: Callable[[T], U]) -> Result[U, E]:
        """Return self (no mapping on failure)"""
        return self

    def __repr__(self) -> str:
        return f"Failure({self.error!r})"


if TYPE_CHECKING:
    Result: TypeAlias = Success[T] | Failure[E]
else:

    class Result(Generic[T, E]):
        """
        Result namespace + typing helper.

        Runtime:
        - Provides helper constructors/operations like `Result.from_exception(...)`
          expected by higher-level tests and docs.

        Typing:
        - For type checkers, `Result[T, E]` is defined as `Success[T] | Failure[E]`
          (see the `TYPE_CHECKING` branch above). This avoids attr/return mismatches
          and allows proper narrowing with `is_success()` / `is_failure()`.

        Note:
        - Actual runtime values are always instances of `Success` or `Failure`.
        """

        @staticmethod
        def from_exception(
            func: Callable[[], T],
            error_type: type[AppError] = AppError,
            error_code: str | None = None,
        ) -> Success[T] | Failure[AppError]:
            """Alias for ResultHelpers.from_exception()."""
            return ResultHelpers.from_exception(func, error_type=error_type, error_code=error_code)

        @staticmethod
        def try_or_error(
            func: Callable[[], T],
            error_message: str,
            error_type: type[AppError] = AppError,
        ) -> Success[T] | Failure[AppError]:
            """Alias for ResultHelpers.try_or_error()."""
            return ResultHelpers.try_or_error(func, error_message, error_type=error_type)

        @staticmethod
        def collect(results: list[Success[T] | Failure[E]]) -> Success[list[T]] | Failure[E]:
            """Alias for ResultHelpers.collect()."""
            return ResultHelpers.collect(results)


def ok(value: T) -> Success[T]:
    """Create a Success result (alias)"""
    return Success(value)


def err(error: E) -> Failure[E]:
    """Create a Failure result (alias)"""
    return Failure(error)


# Backward compatibility aliases (deprecated)
Ok = ok
Err = err


# =============================================================================
# RESULT HELPERS
# =============================================================================


class ResultHelpers:
    """
    Static helper methods for working with Results.

    Usage:
        result = ResultHelpers.from_exception(lambda: risky_operation())
        result = ResultHelpers.from_exception(
            lambda: load_config(),
            error_type=ConfigError,
        )
    """

    @staticmethod
    def from_exception(
        func: Callable[[], T],
        error_type: type[AppError] = AppError,
        error_code: str | None = None,
    ) -> Success[T] | Failure[AppError]:
        """
        Execute function and wrap result or exception in Result.

        Converts exceptions to Failure with AppError (or subclass).

        Args:
            func: Function to execute
            error_type: Error class to use (default: AppError)
            error_code: Optional error code override

        Returns:
            Success with value or Failure with AppError

        Example:
            result = ResultHelpers.from_exception(
                lambda: load_model("path"),
                error_type=ModelError,
            )
        """
        try:
            value = func()
            return Success(value)
        except Exception as e:
            error = error_type(
                message=str(e),
                code=error_code or error_type.__name__.upper().replace("ERROR", "_ERROR"),
                details={"exception_type": type(e).__name__, "traceback": traceback.format_exc()},
            )
            return Failure(error)

    @staticmethod
    def try_or_error(
        func: Callable[[], T],
        error_message: str,
        error_type: type[AppError] = AppError,
    ) -> Success[T] | Failure[AppError]:
        """
        Execute function with custom error message on failure.

        Args:
            func: Function to execute
            error_message: Message to use on failure
            error_type: Error class to use

        Returns:
            Success or Failure

        Example:
            result = ResultHelpers.try_or_error(
                lambda: config.validate(),
                "Configuration validation failed",
                ConfigError,
            )
        """
        try:
            return Success(func())
        except Exception as e:
            return Failure(
                error_type(
                    message=f"{error_message}: {e}",
                    details={"original_exception": str(e)},
                )
            )

    @staticmethod
    def collect(
        results: list[Success[T] | Failure[E]],
    ) -> Success[list[T]] | Failure[E]:
        """
        Collect list of Results into single Result.

        Returns first Failure if any, otherwise Success with all values.

        Args:
            results: List of Results to collect

        Returns:
            Success[list[T]] or first Failure

        Example:
            items = [ok(1), ok(2), ok(3)]
            collected = ResultHelpers.collect(items)
            collected.unwrap()  # [1, 2, 3]
        """
        values: list[T] = []
        for result in results:
            if isinstance(result, Failure):
                return result
            values.append(result.unwrap())
        return Success(values)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base error
    "AppError",
    # Domain errors
    "ConfigError",
    "DataLoaderError",
    "DatasetError",
    "Err",  # deprecated alias
    "Failure",
    "InferenceError",
    "ModelError",
    "OOMError",
    "Ok",  # deprecated alias
    "ProviderError",
    # Helpers
    "Result",
    "ResultHelpers",
    "StrategyError",
    # Core types
    "Success",
    # Error types
    "TrainingError",
    "err",
    # Factories
    "ok",
]

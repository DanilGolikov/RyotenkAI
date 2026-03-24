"""
MLflowAutologManager — MLflow autologging and LLM tracing management.

Responsibilities (Single Responsibility):
  - Enable/disable HuggingFace Transformers autologging
  - Enable/disable PyTorch autologging
  - Enable/disable MLflow LLM tracing
  - Provide context manager for tracing LLM calls
  - Log trace I/O data on active spans
  - Create trace decorators

Does not depend on run state (_run_id, _run, etc.) — operates at the
global MLflow module level.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

logger = get_logger(__name__)


class MLflowAutologManager:
    """
    Manages MLflow autologging and LLM tracing.

    All methods operate at the global MLflow module level and do not
    depend on an active run or gateway state.

    Args:
        mlflow_module: The imported mlflow module (or None if unavailable)
        tracking_uri: Resolved tracking URI (used only for get_trace_url)
    """

    def __init__(self, mlflow_module: Any, tracking_uri: str | None = None) -> None:
        self._mlflow = mlflow_module
        self._tracking_uri = tracking_uri

    # =========================================================================
    # AUTOLOGGING
    # =========================================================================

    def enable_autolog(
        self,
        log_models: bool = False,
        log_input_examples: bool = False,
        log_model_signatures: bool = True,
        log_every_n_steps: int | None = None,
        disable_for_unsupported_versions: bool = True,
        silent: bool = False,
    ) -> bool:
        """
        Enable MLflow autologging for HuggingFace Transformers.

        Automatically logs training arguments, metrics, and optionally model checkpoints.

        Args:
            log_models: Log model checkpoints (default: False — saves space)
            log_input_examples: Log input examples with model
            log_model_signatures: Log model input/output signatures
            log_every_n_steps: Reserved for future use
            disable_for_unsupported_versions: Disable for unsupported transformers versions
            silent: Suppress autolog warnings

        Returns:
            True if autolog enabled successfully
        """
        if self._mlflow is None:
            logger.warning("MLflow not initialized, cannot enable autolog")
            return False
        _ = log_every_n_steps

        try:
            try:
                import mlflow.transformers

                mlflow.transformers.autolog(
                    log_models=log_models,
                    log_input_examples=log_input_examples,
                    log_model_signatures=log_model_signatures,
                    disable=False,
                    silent=silent,
                )
                logger.info("MLflow transformers autolog enabled")
                return True
            except (ImportError, AttributeError):
                pass

            self._mlflow.autolog(
                log_models=log_models,
                log_input_examples=log_input_examples,
                log_model_signatures=log_model_signatures,
                disable_for_unsupported_versions=disable_for_unsupported_versions,
                silent=silent,
            )
            logger.info("MLflow generic autolog enabled")
            return True

        except Exception as e:
            logger.warning(f"Failed to enable autolog: {e}")
            return False

    def disable_autolog(self) -> bool:
        """
        Disable MLflow autologging.

        Returns:
            True if disabled successfully
        """
        if self._mlflow is None:
            return False

        try:
            try:
                import mlflow.transformers

                mlflow.transformers.autolog(disable=True)
                logger.info("MLflow transformers autolog disabled")
                return True
            except (ImportError, AttributeError):
                pass

            self._mlflow.autolog(disable=True)
            logger.info("MLflow autolog disabled")
            return True

        except Exception as e:
            logger.warning(f"Failed to disable autolog: {e}")
            return False

    def enable_pytorch_autolog(
        self,
        log_models: bool = False,
        log_every_n_epoch: int = 1,
        log_every_n_step: int | None = None,
    ) -> bool:
        """
        Enable MLflow autologging specifically for PyTorch.

        Args:
            log_models: Log model checkpoints
            log_every_n_epoch: Log metrics every N epochs
            log_every_n_step: Log metrics every N steps (overrides epoch)

        Returns:
            True if enabled successfully
        """
        if self._mlflow is None:
            return False

        try:
            import mlflow.pytorch

            mlflow.pytorch.autolog(
                log_models=log_models,
                log_every_n_epoch=log_every_n_epoch,
                log_every_n_step=log_every_n_step,
            )
            logger.info("MLflow PyTorch autolog enabled")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable PyTorch autolog: {e}")
            return False

    # =========================================================================
    # TRACING
    # =========================================================================

    def enable_tracing(self) -> bool:
        """
        Enable MLflow tracing for LLM observability.

        Returns:
            True if tracing enabled successfully
        """
        if self._mlflow is None:
            return False

        try:
            self._mlflow.tracing.enable()
            logger.info("MLflow tracing enabled")
            return True
        except AttributeError:
            logger.debug("MLflow tracing not available (requires MLflow 2.x)")
            return False
        except Exception as e:
            logger.warning(f"Failed to enable tracing: {e}")
            return False

    def disable_tracing(self) -> bool:
        """
        Disable MLflow tracing.

        Returns:
            True if disabled successfully
        """
        if self._mlflow is None:
            return False

        try:
            self._mlflow.tracing.disable()
            logger.info("MLflow tracing disabled")
            return True
        except AttributeError:
            return False
        except Exception as e:
            logger.warning(f"Failed to disable tracing: {e}")
            return False

    @contextmanager
    def trace_llm_call(
        self,
        name: str,
        model_name: str | None = None,
        span_type: str = "LLM",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager for tracing LLM inference calls.

        Args:
            name: Name of the trace span
            model_name: Model being used
            span_type: Type of span ("LLM", "CHAIN", "TOOL", "RETRIEVER")
            attributes: Additional attributes to log

        Yields:
            Trace span object (or None if tracing unavailable)
        """
        if self._mlflow is None:
            yield None
            return

        try:
            trace_attrs = attributes or {}
            if model_name:
                trace_attrs["model_name"] = model_name
            trace_attrs["span_type"] = span_type

            with self._mlflow.start_span(name=name, attributes=trace_attrs) as span:
                yield span

        except AttributeError:
            logger.debug("MLflow tracing API not available")
            yield None
        except Exception as e:
            logger.debug(f"Tracing error: {e}")
            yield None

    def log_trace_io(
        self,
        input_data: str | dict[str, Any] | None = None,
        output_data: str | dict[str, Any] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log input/output data for current trace span.

        Args:
            input_data: Input prompt or data
            output_data: Output response or data
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
        """
        if self._mlflow is None:
            return

        try:
            span = self._mlflow.get_current_active_span()
            if span is None:
                return

            if input_data is not None:
                span.set_inputs({"input": input_data} if isinstance(input_data, str) else input_data)

            if output_data is not None:
                span.set_outputs({"output": output_data} if isinstance(output_data, str) else output_data)

            attrs: dict[str, Any] = {}
            if input_tokens is not None:
                attrs["input_tokens"] = input_tokens
            if output_tokens is not None:
                attrs["output_tokens"] = output_tokens
            if latency_ms is not None:
                attrs["latency_ms"] = latency_ms
            if metadata:
                attrs.update(metadata)

            if attrs:
                span.set_attributes(attrs)

        except AttributeError:
            logger.debug("MLflow span API not available")
        except Exception as e:
            logger.debug(f"Failed to log trace I/O: {e}")

    def create_trace_decorator(
        self,
        name: str | None = None,
        span_type: str = "LLM",
    ) -> Any:
        """
        Create a decorator for tracing function calls.

        Args:
            name: Span name (default: function name)
            span_type: Type of span

        Returns:
            Decorator function (no-op if tracing unavailable)
        """
        if self._mlflow is None:

            def noop_decorator(func: Any) -> Any:
                return func

            return noop_decorator

        try:
            return self._mlflow.trace(name=name, span_type=span_type)
        except AttributeError:

            def noop_decorator(func: Any) -> Any:
                return func

            return noop_decorator

    def get_trace_url(self, trace_id: str | None = None) -> str | None:
        """
        Get URL to view trace in MLflow UI.

        Args:
            trace_id: Trace ID (default: current trace)

        Returns:
            URL string or None
        """
        if self._mlflow is None or not self._tracking_uri:
            return None

        try:
            if trace_id is None:
                span = self._mlflow.get_current_active_span()
                if span:
                    trace_id = span.trace_id

            if trace_id:
                return f"{self._tracking_uri}/#/traces/{trace_id}"

            return None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None


__all__ = ["MLflowAutologManager"]

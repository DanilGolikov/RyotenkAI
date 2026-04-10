"""
Resilient MLflow transport shim for provider-side training runtime.

Protects native HF/TRL -> MLflow callback delivery from transient transport
failures without disabling `report_to=["mlflow"]`.
"""

from __future__ import annotations

import functools
import socket
import ssl
import time
from collections.abc import Callable
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PATCHED_MARKER = "_ryotenkai_resilient_transport_owner"
_FAILURE_THRESHOLD = 3
_RECOVERY_COOLDOWN_S = 60.0
_WARNING_INTERVAL_S = 30.0
_FLUENT_METHODS = (
    "log_metric",
    "log_metrics",
    "log_param",
    "log_params",
    "set_tag",
    "set_tags",
    "log_dict",
    "log_text",
)
_CLIENT_METHODS = (
    "log_batch",
    "log_metric",
    "log_param",
    "log_params",
    "set_tag",
    "set_tags",
    "log_dict",
    "log_text",
)
_TRANSPORT_MESSAGE_MARKERS = (
    "API request to",
    "HTTPSConnectionPool",
    "HTTPConnectionPool",
    "Max retries exceeded",
    "Failed to establish a new connection",
    "Connection aborted",
    "Connection refused",
    "Connection reset",
    "Read timed out",
    "ConnectTimeout",
    "SSLEOFError",
    "UNEXPECTED_EOF_WHILE_READING",
)


def _optional_exception_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [
        ssl.SSLError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
        ConnectionError,
        ConnectionResetError,
        BrokenPipeError,
    ]
    try:
        from requests import exceptions as requests_exceptions

        types.extend(
            [
                requests_exceptions.RequestException,
                requests_exceptions.Timeout,
                requests_exceptions.ConnectionError,
                requests_exceptions.SSLError,
            ]
        )
    except Exception:
        pass
    try:
        from urllib3 import exceptions as urllib3_exceptions

        types.extend(
            [
                urllib3_exceptions.HTTPError,
                urllib3_exceptions.MaxRetryError,
                urllib3_exceptions.ConnectTimeoutError,
                urllib3_exceptions.ReadTimeoutError,
                urllib3_exceptions.SSLError,
            ]
        )
    except Exception:
        pass
    return tuple(dict.fromkeys(types))


_TRANSPORT_EXCEPTION_TYPES = _optional_exception_types()


class MLflowTransportCircuitBreaker:
    """Small explicit state machine for MLflow transport failures."""

    def __init__(
        self,
        *,
        failure_threshold: int = _FAILURE_THRESHOLD,
        recovery_cooldown_s: float = _RECOVERY_COOLDOWN_S,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_cooldown_s = max(0.0, recovery_cooldown_s)
        self.state = "closed"
        self.consecutive_failures = 0
        self._opened_at: float | None = None

    def allow_call(self) -> bool:
        if self.state != "open":
            return True
        now = time.monotonic()
        if self._opened_at is None or (now - self._opened_at) >= self.recovery_cooldown_s:
            self.state = "half_open"
            return True
        return False

    def record_success(self) -> None:
        self.state = "closed"
        self.consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.state == "half_open" or self.consecutive_failures >= self.failure_threshold:
            self.state = "open"
            self._opened_at = time.monotonic()


class ResilientMLflowTransport:
    """Installs best-effort wrappers around MLflow fluent and client logging APIs.

    When a MetricsBuffer is attached, metrics are buffered to disk instead of
    being silently dropped while the circuit breaker is open. Buffered metrics
    are flushed automatically when connectivity recovers.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = _FAILURE_THRESHOLD,
        recovery_cooldown_s: float = _RECOVERY_COOLDOWN_S,
        warning_interval_s: float = _WARNING_INTERVAL_S,
    ) -> None:
        self._breaker = MLflowTransportCircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_cooldown_s=recovery_cooldown_s,
        )
        self._warning_interval_s = max(0.0, warning_interval_s)
        self._last_warning_at: dict[str, float] = {}
        self._module: Any = None
        self._originals: dict[tuple[str, str], Any] = {}
        self._buffer: Any | None = None  # Optional MetricsBuffer

    def attach_buffer(self, buffer: Any) -> None:
        """Attach a MetricsBuffer for offline metric storage."""
        self._buffer = buffer
        logger.info("Metrics buffer attached: %s", getattr(buffer, "path", "unknown"))

    @property
    def breaker_state(self) -> str:
        return self._breaker.state

    def install(self, mlflow_module: Any) -> bool:
        owner = getattr(mlflow_module, _PATCHED_MARKER, None)
        if owner is self:
            return False
        if owner is not None:
            logger.debug("MLflow resilient transport already installed by another owner")
            return False

        self._module = mlflow_module
        setattr(mlflow_module, _PATCHED_MARKER, self)

        for method_name in _FLUENT_METHODS:
            self._patch_method("module", mlflow_module, method_name)

        client_cls = getattr(mlflow_module, "MlflowClient", None)
        if client_cls is not None:
            for method_name in _CLIENT_METHODS:
                self._patch_method("client", client_cls, method_name)

        logger.info("Installed resilient MLflow transport shim")
        return True

    def uninstall(self) -> None:
        if self._module is None:
            return

        for (scope, method_name), original in reversed(tuple(self._originals.items())):
            target = self._module if scope == "module" else getattr(self._module, "MlflowClient", None)
            if target is None:
                continue
            setattr(target, method_name, original)

        if getattr(self._module, _PATCHED_MARKER, None) is self:
            delattr(self._module, _PATCHED_MARKER)

        self._originals.clear()
        self._module = None

    def _patch_method(self, scope: str, target: Any, method_name: str) -> None:
        original = getattr(target, method_name, None)
        key = (scope, method_name)
        if original is None or key in self._originals:
            return
        self._originals[key] = original
        setattr(target, method_name, self._make_wrapper(method_name, original))

    def _buffer_metric_call(self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Try to buffer a metric call when circuit is open."""
        if self._buffer is None:
            return
        try:
            if operation == "log_metric":
                # mlflow.log_metric(key, value, step=None, timestamp=None, ...)
                key = args[0] if len(args) > 0 else kwargs.get("key")
                value = args[1] if len(args) > 1 else kwargs.get("value")
                step = args[2] if len(args) > 2 else kwargs.get("step", 0)
                if key is not None and value is not None:
                    self._buffer.write_metric(str(key), float(value), int(step or 0))
            elif operation == "log_metrics":
                # mlflow.log_metrics(metrics, step=None, ...)
                metrics = args[0] if len(args) > 0 else kwargs.get("metrics", {})
                step = args[1] if len(args) > 1 else kwargs.get("step", 0)
                if isinstance(metrics, dict):
                    self._buffer.write_metrics(metrics, int(step or 0))
        except Exception:
            pass  # Buffer is best-effort

    def _flush_buffer(self, original: Callable[..., Any], operation: str) -> None:
        """Flush buffered metrics on circuit recovery. Best-effort."""
        if self._buffer is None or self._buffer.count == 0:
            return
        if operation not in ("log_metric", "log_metrics"):
            return
        # Use the original (unpatched) log_metric to flush
        log_metric_original = self._originals.get(("module", "log_metric"))
        if log_metric_original is None:
            log_metric_original = original if operation == "log_metric" else None
        if log_metric_original is not None:
            try:
                self._buffer.flush(log_metric_original)
            except Exception as e:
                logger.debug("[BUFFER] Flush error: %s", e)

    def _make_wrapper(self, operation: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self._breaker.allow_call():
                self._warn_rate_limited(
                    operation,
                    f"[MLFLOW:DEGRADED] skip op={operation} state=open reason=circuit_breaker_open",
                )
                self._buffer_metric_call(operation, args, kwargs)
                return None

            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                if not self._is_transport_exception(exc):
                    raise
                self._breaker.record_failure()
                self._warn_rate_limited(
                    operation,
                    "[MLFLOW:DEGRADED] "
                    f"op={operation} state={self._breaker.state} failures={self._breaker.consecutive_failures} "
                    f"reason={type(exc).__name__}: {exc}",
                )
                self._buffer_metric_call(operation, args, kwargs)
                return None

            self._breaker.record_success()
            self._flush_buffer(original, operation)
            return result

        return wrapped

    def _warn_rate_limited(self, key: str, message: str) -> None:
        now = time.monotonic()
        last_logged = self._last_warning_at.get(key)
        if last_logged is not None and (now - last_logged) < self._warning_interval_s:
            return
        self._last_warning_at[key] = now
        logger.warning(message)

    def _is_transport_exception(self, exc: BaseException) -> bool:
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, _TRANSPORT_EXCEPTION_TYPES):
                return True

            message = str(current)
            if any(marker in message for marker in _TRANSPORT_MESSAGE_MARKERS):
                return True

            current = current.__cause__ or current.__context__

        return False


__all__ = [
    "MLflowTransportCircuitBreaker",
    "ResilientMLflowTransport",
]

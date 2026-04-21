"""
MLflowGateway — single entry point for the MLflow HTTP API.

All blocking SDK calls are wrapped in ThreadPoolExecutor.submit().result(timeout=N),
which provides an explicit timeout without global env vars.

Architecture:
    IMLflowGateway   — Protocol (for tests and DI)
    MLflowGateway    — concrete implementation
    NullMLflowGateway — stub (MLflow disabled / not needed)

Consumers:
    MLflowManager, SystemPromptLoader, MLflowAdapter, ExperimentReportGenerator
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from typing import Any, Protocol

from src.utils.logger import get_logger
from src.utils.result import AppError, ConfigError

logger = get_logger(__name__)

class IMLflowGateway(Protocol):
    """
    Protocol for MLflow infrastructure layer.

    All concrete implementations must provide these methods so that
    callers can depend on the abstraction, not the implementation.
    """

    @property
    def uri(self) -> str:
        """Configured tracking URI."""
        ...

    @property
    def last_connectivity_error(self) -> AppError | None:
        """Last connectivity probe error, if any."""
        ...

    def get_client(self) -> Any:
        """
        Return a fully configured MlflowClient bound to this gateway's URI.

        Always returns a fresh client (MlflowClient is lightweight).
        """
        ...

    def load_prompt(self, name: str, timeout: float = 10.0) -> str | None:
        """
        Load prompt text from MLflow Prompt Registry.

        Args:
            name: Prompt name or URI (e.g. 'my-prompt', 'prompts:/my-prompt/3').
            timeout: Max seconds to wait for the HTTP call.

        Returns:
            Prompt template string, or None on any error / timeout.
        """
        ...

    def check_connectivity(self, timeout: float = 5.0) -> bool:
        """
        Check if the MLflow server is reachable.

        Args:
            timeout: Max seconds for the connectivity probe.

        Returns:
            True if reachable, False otherwise.
        """
        ...


class MLflowGateway:
    """
    Default MLflowGateway implementation.

    Every blocking SDK / HTTP call is submitted to a single-use
    ThreadPoolExecutor so that ``future.result(timeout=N)`` provides
    a hard upper-bound on wait time without relying on env vars.
    """

    DEFAULT_CALL_TIMEOUT: float = 10.0
    DEFAULT_CONNECTIVITY_TIMEOUT: float = 5.0

    def __init__(self, tracking_uri: str, *, ca_bundle_path: str | None = None) -> None:
        """
        Args:
            tracking_uri: Explicit tracking URI from config/runtime resolver.
            ca_bundle_path: Optional CA bundle for HTTPS verification.
        """
        self._uri = tracking_uri
        self._ca_bundle_path = ca_bundle_path
        self._last_connectivity_error: AppError | None = None

        logger.debug(
            "[MLFLOW:GATEWAY] Initialized with URI=%r (ca_bundle_path=%r)",
            self._uri,
            self._ca_bundle_path,
        )

    # -------------------------------------------------------------------------
    # IMLflowGateway implementation
    # -------------------------------------------------------------------------

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def last_connectivity_error(self) -> AppError | None:
        return self._last_connectivity_error

    def get_client(self) -> Any:
        """
        Create a MlflowClient bound to this gateway's URI.

        Uses explicit tracking_uri to force REST API usage (avoids permission
        errors when remote machines try to write to local artifact paths).
        """
        from mlflow import MlflowClient

        logger.debug(f"[MLFLOW:GATEWAY] Creating MlflowClient with URI={self._uri!r}")
        return MlflowClient(tracking_uri=self._uri)

    def load_prompt(self, name: str, timeout: float = DEFAULT_CALL_TIMEOUT) -> str | None:
        """
        Load prompt from MLflow Prompt Registry with an explicit timeout.

        mlflow.genai.load_prompt performs an HTTP request; without a timeout
        it can block indefinitely when the server is unreachable.
        """
        try:
            import mlflow
            import mlflow.genai

            def _call() -> Any:
                mlflow.set_tracking_uri(self._uri)
                return mlflow.genai.load_prompt(name)

            # Carry ContextVars (per-stage logging context etc.) into worker.
            parent_ctx = copy_context()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(parent_ctx.run, _call)
                return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(f"[MLFLOW:GATEWAY] load_prompt({name!r}) timed out after {timeout}s " f"(URI={self._uri!r})")
            return None
        except Exception as exc:
            logger.warning(f"[MLFLOW:GATEWAY] load_prompt({name!r}) failed: {exc} " f"(URI={self._uri!r})")
            return None

    def check_connectivity(self, timeout: float = DEFAULT_CONNECTIVITY_TIMEOUT) -> bool:
        """
        Probe the MLflow server with an HTTP HEAD request.

        Uses stdlib urllib (no SDK overhead) so this call can be safely
        used before the MLflow module is imported.
        """
        if not self._uri or not self._uri.startswith("http"):
            self._last_connectivity_error = None
            return True  # file:// or empty — assume available

        try:
            context = self._build_ssl_context()
            request = urllib.request.Request(self._uri, method="HEAD")
            urlopen_kwargs: dict[str, Any] = {"timeout": timeout}
            if context is not None:
                urlopen_kwargs["context"] = context
            with urllib.request.urlopen(request, **urlopen_kwargs) as response:
                self._last_connectivity_error = None
                return response.status < 500
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                self._last_connectivity_error = AppError(
                    message=f"MLflow connectivity probe returned HTTP {exc.code} for {self._uri}",
                    code="MLFLOW_PREFLIGHT_HTTP_ERROR",
                    details={"status_code": exc.code, "tracking_uri": self._uri},
                )
            else:
                self._last_connectivity_error = None
            return exc.code < 500  # 4xx means server is reachable
        except ssl.SSLCertVerificationError as exc:
            self._last_connectivity_error = AppError(
                message=f"MLflow TLS certificate verification failed for {self._uri}: {exc}",
                code="MLFLOW_TLS_CERT_VERIFY_FAILED",
                details={"tracking_uri": self._uri},
            )
            logger.debug(f"[MLFLOW:GATEWAY] Connectivity check failed: {exc}")
            return False
        except FileNotFoundError as exc:
            self._last_connectivity_error = ConfigError(
                message=f"MLflow CA bundle file not found: {self._ca_bundle_path}",
                code="MLFLOW_TLS_CA_BUNDLE_INVALID",
                details={"tracking_uri": self._uri, "ca_bundle_path": self._ca_bundle_path},
            )
            logger.debug(f"[MLFLOW:GATEWAY] Connectivity check failed: {exc}")
            return False
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            self._last_connectivity_error = self._map_connectivity_error(exc)
            logger.debug(f"[MLFLOW:GATEWAY] Connectivity check failed: {exc}")
            return False

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if not self._uri.startswith("https://"):
            return None
        if self._ca_bundle_path:
            return ssl.create_default_context(cafile=self._ca_bundle_path)
        return ssl.create_default_context()

    def _map_connectivity_error(self, exc: OSError | TimeoutError | urllib.error.URLError) -> AppError:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, ssl.SSLCertVerificationError):
            return AppError(
                message=f"MLflow TLS certificate verification failed for {self._uri}: {reason}",
                code="MLFLOW_TLS_CERT_VERIFY_FAILED",
                details={"tracking_uri": self._uri},
            )
        if isinstance(reason, TimeoutError):
            return AppError(
                message=f"MLflow connectivity probe timed out for {self._uri}",
                code="MLFLOW_PREFLIGHT_TIMEOUT",
                details={"tracking_uri": self._uri},
            )
        return AppError(
            message=f"MLflow connectivity probe failed for {self._uri}: {reason}",
            code="MLFLOW_PREFLIGHT_CONNECTION_FAILED",
            details={"tracking_uri": self._uri},
        )


class NullMLflowGateway:
    """
    No-op gateway for when MLflow is disabled or not configured.

    All methods are safe to call — they return sensible defaults without
    making any network calls.
    """

    @property
    def uri(self) -> str:
        return ""

    @property
    def last_connectivity_error(self) -> AppError | None:
        return None

    def get_client(self) -> None:
        return None

    def load_prompt(self, name: str, timeout: float = 10.0) -> None:  # noqa: ARG002
        return None

    def check_connectivity(self, timeout: float = 5.0) -> bool:  # noqa: ARG002
        return False


__all__ = [
    "IMLflowGateway",
    "MLflowGateway",
    "NullMLflowGateway",
]

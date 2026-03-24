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

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Protocol
from urllib.parse import urlparse

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default port used when URI has no explicit port
_MLFLOW_PORT_DEFAULT = 5000

# Private network prefixes for localhost-optimization
_PRIVATE_PREFIXES: tuple[str, ...] = (
    "192.168.",
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.2",
    "172.30.",
    "172.31.",
)


class IMLflowGateway(Protocol):
    """
    Protocol for MLflow infrastructure layer.

    All concrete implementations must provide these methods so that
    callers can depend on the abstraction, not the implementation.
    """

    @property
    def uri(self) -> str:
        """Resolved (possibly localhost-optimised) tracking URI."""
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

    def __init__(self, tracking_uri: str, *, normalize: bool = True) -> None:
        """
        Args:
            tracking_uri: Raw tracking URI from config (may be a LAN IP).
            normalize: If True, replace private-network IPs with localhost
                       when the MLflow server runs on the same machine
                       (avoids unnecessary LAN round-trips).
        """
        if normalize:
            self._uri = self._normalize_uri(tracking_uri)
        else:
            self._uri = tracking_uri

        logger.debug(f"[MLFLOW:GATEWAY] Initialized with URI={self._uri!r} (normalize={normalize})")

    # -------------------------------------------------------------------------
    # IMLflowGateway implementation
    # -------------------------------------------------------------------------

    @property
    def uri(self) -> str:
        return self._uri

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
        import mlflow
        import mlflow.genai

        def _call() -> Any:
            mlflow.set_tracking_uri(self._uri)
            return mlflow.genai.load_prompt(name)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
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
            return True  # file:// or empty — assume available

        try:
            request = urllib.request.Request(self._uri, method="HEAD")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status < 500
        except urllib.error.HTTPError as exc:
            return exc.code < 500  # 4xx means server is reachable
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            logger.debug(f"[MLFLOW:GATEWAY] Connectivity check failed: {exc}")
            return False

    # -------------------------------------------------------------------------
    # URI normalisation
    # -------------------------------------------------------------------------

    def _normalize_uri(self, uri: str) -> str:
        """
        Replace private-network IP with localhost when on the same machine.

        Remote hosts keep the original IP (they can't reach localhost of the
        machine running this code).
        """
        if not uri or not uri.startswith("http"):
            return uri

        try:
            parsed = urlparse(uri)
            host = parsed.hostname
            port = parsed.port or _MLFLOW_PORT_DEFAULT

            if not host:
                return uri

            if not any(host.startswith(prefix) for prefix in _PRIVATE_PREFIXES):
                return uri

            # Try localhost on the same port — if it responds, we're on the same machine
            local_uri = f"http://localhost:{port}"
            if self._probe_uri(local_uri, timeout=1.0):
                logger.info(f"[MLFLOW:GATEWAY] Using localhost: {uri} → {local_uri}")
                return local_uri

            logger.debug(f"[MLFLOW:GATEWAY] Using network IP: {uri}")
            return uri

        except (ValueError, OSError) as exc:
            logger.debug(f"[MLFLOW:GATEWAY] URI normalization failed: {exc}")
            return uri

    @staticmethod
    def _probe_uri(uri: str, timeout: float) -> bool:
        """Probe a specific URI (used internally during normalization)."""
        try:
            request = urllib.request.Request(uri, method="HEAD")
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return resp.status < 500
        except urllib.error.HTTPError as exc:
            return exc.code < 500
        except (OSError, TimeoutError, urllib.error.URLError):
            return False


class NullMLflowGateway:
    """
    No-op gateway for when MLflow is disabled or not configured.

    All methods are safe to call — they return sensible defaults without
    making any network calls.
    """

    @property
    def uri(self) -> str:
        return ""

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

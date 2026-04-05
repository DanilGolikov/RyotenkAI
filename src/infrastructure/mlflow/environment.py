"""
MLflowEnvironment — single owner of MLflow process-wide state.

Manages three categories of global mutations that previously were scattered
across setup.py (os.environ, mlflow.set_tracking_uri) and main.py (atexit):

    1. os.environ["REQUESTS_CA_BUNDLE"] / os.environ["SSL_CERT_FILE"]
    2. mlflow.set_tracking_uri()
    3. atexit.unregister(mlflow._safe_end_run)

activate() and deactivate() are idempotent: safe to call multiple times.
force_unregister_atexit() is a static escape hatch for signal handlers
where full deactivation is impractical.
"""

from __future__ import annotations

import atexit
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowEnvironment:
    """
    Single owner of MLflow process-wide state.

    Lifecycle:
        env = MLflowEnvironment(tracking_uri, ca_bundle_path)
        env.activate()      # sets os.environ + mlflow.set_tracking_uri
        ...
        env.deactivate()    # restores os.environ + unregisters atexit hook
    """

    _ENV_KEYS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")

    def __init__(self, tracking_uri: str, ca_bundle_path: str | None = None) -> None:
        self._tracking_uri = tracking_uri
        self._ca_bundle_path = ca_bundle_path
        self._prev_env: dict[str, str | None] = {}
        self._active = False

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Set process-wide MLflow state. Idempotent."""
        if self._active:
            return

        if self._ca_bundle_path:
            for key in self._ENV_KEYS:
                self._prev_env[key] = os.environ.get(key)
                os.environ[key] = self._ca_bundle_path

        import mlflow

        mlflow.set_tracking_uri(self._tracking_uri)
        self._active = True
        logger.debug(
            "[MLFLOW:ENV] Activated (tracking_uri=%r, ca_bundle=%r)",
            self._tracking_uri,
            self._ca_bundle_path,
        )

    def deactivate(self) -> None:
        """Restore env vars and unregister atexit hook. Idempotent."""
        self._unregister_atexit()

        if not self._active:
            return

        for key, prev_value in self._prev_env.items():
            if prev_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev_value
        self._prev_env.clear()
        self._active = False
        logger.debug("[MLFLOW:ENV] Deactivated")

    @staticmethod
    def force_unregister_atexit() -> None:
        """
        Emergency atexit unregister without full deactivation.

        Intended for signal handlers where accessing instance state
        may be unsafe or unavailable.
        """
        MLflowEnvironment._unregister_atexit()

    @staticmethod
    def _unregister_atexit() -> None:
        try:
            import mlflow.tracking.fluent as _fluent

            atexit.unregister(_fluent._safe_end_run)
            logger.debug("[MLFLOW:ENV] atexit hook unregistered")
        except Exception:
            pass


__all__ = ["MLflowEnvironment"]

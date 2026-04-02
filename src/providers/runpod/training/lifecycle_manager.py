"""
Pod Lifecycle Manager - Manages pod lifecycle (waiting, health checking).

Handles pod state transitions and health monitoring.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from src.providers.runpod.models import PodSnapshot
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_STUCK_WARN_THRESHOLD = 120
_WAIT_TIMEOUT_DEFAULT = 300
_POD_ID_KEY = "pod_id"

if TYPE_CHECKING:
    from src.providers.runpod.training.cleanup_manager import RunPodCleanupManager


class _PodQueryControl(Protocol):
    def query_pod_snapshot(self, pod_id: str) -> Result[PodSnapshot, ProviderError]: ...


class PodLifecycleManager:
    """Manages pod lifecycle (waiting, health checking).

    Operates exclusively on typed ``PodSnapshot`` objects, with no knowledge
    of the underlying data source (GraphQL, runpodctl, etc.).
    """

    def __init__(self, api_client: _PodQueryControl, cleanup_manager: RunPodCleanupManager):
        self.api_client = api_client
        self.cleanup_manager = cleanup_manager
        logger.debug("🔄 PodLifecycleManager initialized")

    def wait_for_ready(
        self,
        pod_id: str,
        timeout: int | None = None,
        max_retries: int = 4,
    ) -> Result[PodSnapshot, ProviderError]:
        """Wait for pod to reach RUNNING state with SSH endpoint available."""
        effective_timeout = timeout if timeout is not None else _WAIT_TIMEOUT_DEFAULT
        logger.info(
            f"⏳ Waiting for pod {pod_id} to be ready (timeout: {effective_timeout}s, max retries: {max_retries})..."
        )

        for attempt in range(max_retries):
            if attempt > 0:
                logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}")

            result = self._wait_single_attempt(pod_id, effective_timeout)

            if result.is_success():
                return result

            error = result.unwrap_err()

            if attempt < max_retries - 1:
                time.sleep(5)
                continue

            return Err(
                ProviderError(
                    message=f"Pod failed to reach RUNNING state after {max_retries} attempts: {error}",
                    code="RUNPOD_POD_NOT_READY",
                    details={_POD_ID_KEY: pod_id, "attempts": max_retries},
                )
            )

        return Err(
            ProviderError(
                message=f"Pod failed after {max_retries} retries",
                code="RUNPOD_POD_NOT_READY",
                details={_POD_ID_KEY: pod_id, "attempts": max_retries},
            )
        )

    def _wait_single_attempt(self, pod_id: str, timeout: int) -> Result[PodSnapshot, ProviderError]:
        """Single attempt to wait for pod to be ready."""
        start_time = time.time()
        stuck_count = 0
        last_fingerprint: tuple[str | None, int, int, bool] | None = None
        poll_interval_seconds = 10

        while True:
            elapsed_s = int(time.time() - start_time)
            if elapsed_s >= timeout:
                break

            query_result = self.api_client.query_pod_snapshot(pod_id)

            if query_result.is_failure():
                err = query_result.unwrap_err()
                logger.warning(f"Error querying pod: {err} (elapsed: {elapsed_s}s/{timeout}s)")
                if "No pod data received" in err.message:
                    return Err(err)
                time.sleep(10)
                continue

            snapshot = query_result.unwrap()

            logger.info(
                f"📊 Pod status: {snapshot.status}, uptime: {snapshot.uptime_seconds}s, "
                f"ports: {snapshot.port_count} (elapsed: {elapsed_s}s/{timeout}s)"
            )

            if snapshot.is_ready:
                logger.info("✅ Pod is running and ready!")
                return Ok(snapshot)

            if snapshot.status == "RUNNING" and snapshot.ssh_endpoint is None:
                logger.info(
                    f"⏳ Pod running but waiting for SSH over exposed TCP... (elapsed: {elapsed_s}s/{timeout}s)"
                )
            elif snapshot.is_terminal:
                return Err(
                    ProviderError(
                        message=f"Pod entered failed state: {snapshot.status}",
                        code="RUNPOD_POD_FAILED",
                        details={_POD_ID_KEY: pod_id, "status": snapshot.status},
                    )
                )

            fingerprint = (snapshot.status, snapshot.uptime_seconds, snapshot.port_count, snapshot.ssh_endpoint is not None)
            if last_fingerprint is not None and fingerprint == last_fingerprint:
                stuck_count += 1
                no_progress_s = stuck_count * poll_interval_seconds
                if no_progress_s >= _STUCK_WARN_THRESHOLD and no_progress_s % 60 == 0:
                    logger.warning(
                        f"⚠️ Pod shows no progress for ~{no_progress_s}s "
                        f"(status={snapshot.status}, elapsed: {elapsed_s}s/{timeout}s); continuing to wait..."
                    )
            else:
                stuck_count = 0
                last_fingerprint = fingerprint

            time.sleep(poll_interval_seconds)

        return Err(
            ProviderError(
                message=f"Timeout waiting for pod to be ready ({timeout}s)",
                code="RUNPOD_POD_TIMEOUT",
                details={_POD_ID_KEY: pod_id, "timeout": timeout},
            )
        )

    def check_health(self, pod_id: str) -> Result[bool, ProviderError]:
        """Quick health check for pod."""
        result = self.api_client.query_pod_snapshot(pod_id)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        snapshot = result.unwrap()
        return Ok(snapshot.status == "RUNNING")


__all__ = ["PodLifecycleManager"]

"""
Pod Lifecycle Manager - Manages pod lifecycle (waiting, health checking).

Handles pod state transitions and health monitoring.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_STUCK_WARN_THRESHOLD = 120
_WAIT_TIMEOUT_DEFAULT = 300
_POD_ID_KEY = "pod_id"

if TYPE_CHECKING:
    from src.providers.runpod.training.api_client import RunPodAPIClient
    from src.providers.runpod.training.cleanup_manager import RunPodCleanupManager


class PodLifecycleManager:
    """
    Manages pod lifecycle (waiting, health checking).

    Responsibilities:
    - Wait for pod to reach RUNNING state
    - Health checking with retries
    - Stuck pod detection
    - Coordinate cleanup on failures
    """

    def __init__(self, api_client: RunPodAPIClient, cleanup_manager: RunPodCleanupManager):
        """
        Initialize lifecycle manager.

        Args:
            api_client: RunPodAPIClient instance for API operations
            cleanup_manager: RunPodCleanupManager for cleanup on failures
        """
        self.api_client = api_client
        self.cleanup_manager = cleanup_manager
        logger.debug("🔄 PodLifecycleManager initialized")

    def wait_for_ready(
        self,
        pod_id: str,
        timeout: int | None = None,
        max_retries: int = 4,
    ) -> Result[dict, ProviderError]:
        """
        Wait for pod to reach RUNNING state with retries.

        Args:
            pod_id: Pod ID to wait for
            timeout: Timeout in seconds for each attempt (default: 5 minutes)
            max_retries: Maximum number of retry attempts (default: 4)

        Returns:
            Result with pod data (when ready) or error message
        """
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
                # IMPORTANT:
                # `wait_for_ready` cannot create a new pod, so it must NOT terminate the current pod here.
                # Pod cleanup (terminate/unregister) is handled by the provider on overall connect() failure.
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

    def _wait_single_attempt(self, pod_id: str, timeout: int) -> Result[dict, ProviderError]:
        """Single attempt to wait for pod to be ready."""
        start_time = time.time()
        stuck_count = 0
        last_snapshot: tuple[str | None, int, int] | None = None
        poll_interval_seconds = 10

        while True:
            elapsed_s = int(time.time() - start_time)
            if elapsed_s >= timeout:
                break

            query_result = self.api_client.query_pod(pod_id)

            if query_result.is_failure():
                err = query_result.unwrap_err()
                logger.warning(f"Error querying pod: {err} (elapsed: {elapsed_s}s/{timeout}s)")
                # If the pod no longer exists, fail-fast (not a transient condition).
                if "No pod data received" in err.message:
                    return Err(err)
                time.sleep(10)
                continue

            pod_data = query_result.unwrap()

            status = pod_data.get("desiredStatus")
            runtime = pod_data.get("runtime") or {}
            uptime = runtime.get("uptimeInSeconds") or 0
            ports = runtime.get("ports") or []

            logger.info(
                f"📊 Pod status: {status}, uptime: {uptime}s, ports: {len(ports)} (elapsed: {elapsed_s}s/{timeout}s)"
            )

            # Ready when the pod is RUNNING and ports are assigned (SSH info depends on runtime.ports).
            if status == "RUNNING" and ports:
                logger.info("✅ Pod is running and ready!")
                return Ok(pod_data)

            if status == "RUNNING" and not ports:
                logger.info(f"⏳ Pod running but waiting for ports... (elapsed: {elapsed_s}s/{timeout}s)")

            elif status in ["FAILED", "TERMINATED", "EXITED"]:
                return Err(
                    ProviderError(
                        message=f"Pod entered failed state: {status}",
                        code="RUNPOD_POD_FAILED",
                        details={_POD_ID_KEY: pod_id, "status": status},
                    )
                )

            # Stuck detection: no observable progress (status/uptime/ports) for too long.
            snapshot = (status, int(uptime), len(ports))
            if last_snapshot is not None and snapshot == last_snapshot:
                stuck_count += 1
                no_progress_s = stuck_count * poll_interval_seconds
                # Do NOT fail-fast here: pods often report RUNNING before ports/uptime advance.
                # We still emit periodic warnings for observability and rely on the attempt timeout.
                if no_progress_s >= _STUCK_WARN_THRESHOLD and no_progress_s % 60 == 0:
                    logger.warning(
                        f"⚠️ Pod shows no progress for ~{no_progress_s}s (status={status}, elapsed: {elapsed_s}s/{timeout}s); "
                        "continuing to wait..."
                    )
            else:
                stuck_count = 0
                last_snapshot = snapshot

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
        result = self.api_client.query_pod(pod_id)

        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]

        pod_data = result.unwrap()
        status = pod_data.get("desiredStatus")
        runtime = pod_data.get("runtime")

        is_healthy = status == "RUNNING" and runtime is not None

        return Ok(is_healthy)


__all__ = ["PodLifecycleManager"]

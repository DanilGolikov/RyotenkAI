"""
Pod Lifecycle Manager - Manages pod lifecycle (waiting, health checking).

Handles pod state transitions and health monitoring.
"""

from __future__ import annotations

import time
from typing import Protocol

from src.providers.runpod.models import PodSnapshot
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_WAIT_TIMEOUT_DEFAULT = 300
_NO_EXPOSED_TCP_GRACE_S = 30
_POD_ID_KEY = "pod_id"

class _PodQueryControl(Protocol):
    def query_pod_snapshot(self, pod_id: str) -> Result[PodSnapshot, ProviderError]: ...


class PodLifecycleManager:
    """Manages pod lifecycle (waiting, health checking).

    Operates exclusively on typed ``PodSnapshot`` objects, with no knowledge
    of the underlying transport or API client.
    """

    def __init__(self, api_client: _PodQueryControl):
        self.api_client = api_client
        logger.debug("🔄 PodLifecycleManager initialized")

    def wait_for_ready(
        self,
        pod_id: str,
        timeout: int | None = None,
    ) -> Result[PodSnapshot, ProviderError]:
        """Wait for pod to reach RUNNING state with SSH endpoint available.

        No internal retries — the caller (provider) handles pod recreation.
        """
        effective_timeout = timeout if timeout is not None else _WAIT_TIMEOUT_DEFAULT
        logger.info(f"⏳ Waiting for pod {pod_id} to be ready (timeout: {effective_timeout}s)...")
        return self._wait_single_attempt(pod_id, effective_timeout)

    def _wait_single_attempt(self, pod_id: str, timeout: int) -> Result[PodSnapshot, ProviderError]:
        """Single attempt to wait for pod to be ready."""
        start_time = time.time()
        poll_interval_seconds = 10
        ports_without_ssh_since: float | None = None

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

            if snapshot.is_terminal:
                return Err(
                    ProviderError(
                        message=f"Pod entered failed state: {snapshot.status}",
                        code="RUNPOD_POD_FAILED",
                        details={_POD_ID_KEY: pod_id, "status": snapshot.status},
                    )
                )

            if snapshot.status == "RUNNING" and snapshot.port_count > 0 and snapshot.ssh_endpoint is None:
                if ports_without_ssh_since is None:
                    ports_without_ssh_since = time.time()
                    logger.warning(
                        f"⚠️ Pod has {snapshot.port_count} port(s) but no SSH exposed TCP endpoint. "
                        f"Waiting up to {_NO_EXPOSED_TCP_GRACE_S}s..."
                    )
                waiting_s = int(time.time() - ports_without_ssh_since)
                if waiting_s >= _NO_EXPOSED_TCP_GRACE_S:
                    return Err(
                        ProviderError(
                            message=(
                                f"Pod has {snapshot.port_count} port(s) but no SSH over exposed TCP "
                                f"after {waiting_s}s. This machine likely doesn't support "
                                f"exposed TCP ports (community cloud limitation). Pod will be recreated."
                            ),
                            code="RUNPOD_NO_EXPOSED_TCP",
                            details={_POD_ID_KEY: pod_id, "port_count": snapshot.port_count},
                        )
                    )
                logger.info(
                    f"⏳ Pod running, ports present but no SSH exposed TCP... "
                    f"(waiting: {waiting_s}s/{_NO_EXPOSED_TCP_GRACE_S}s, elapsed: {elapsed_s}s/{timeout}s)"
                )
            elif snapshot.status == "RUNNING" and snapshot.port_count == 0:
                ports_without_ssh_since = None
                logger.info(f"⏳ Pod running, waiting for ports... (elapsed: {elapsed_s}s/{timeout}s)")

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

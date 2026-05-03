"""Training-side lifecycle manager — thin shim over :class:`PodSshWaiter`.

Historically owned a self-contained 90-line poll loop. That loop has
moved to ``src.providers.runpod.lifecycle.pod_ssh_waiter`` (where it's
shared with the inference path), and this module now just adapts the
public ``wait_for_ready`` / ``check_health`` surface to the canonical
primitive.

The ``PodLifecycleManager`` name and signature stay so the training
provider's existing call sites don't churn.
"""

from __future__ import annotations

from dataclasses import replace

from ryotenkai_providers.runpod.lifecycle import (
    TRAINING_PROFILE,
    PodQuery,
    PodSshWaiter,
)
from ryotenkai_providers.runpod.models import PodSnapshot
from ryotenkai_shared.utils.logger import logger
from ryotenkai_shared.utils.result import Err, Ok, ProviderError, Result


class PodLifecycleManager:
    """Pod lifecycle adapter for the training provider.

    Operates on typed :class:`PodSnapshot` via the
    :class:`PodQuery` Protocol — no transport/API-client knowledge here.
    """

    def __init__(self, api_client: PodQuery) -> None:
        self._query = api_client
        logger.debug("🔄 PodLifecycleManager initialized")

    def wait_for_ready(
        self,
        pod_id: str,
        timeout: int | None = None,
    ) -> Result[PodSnapshot, ProviderError]:
        """Wait for pod to reach RUNNING with an SSH endpoint listening.

        Delegates to :class:`PodSshWaiter` with :data:`TRAINING_PROFILE`
        (300 s default). When ``timeout`` is supplied, it overrides the
        profile's ``total_timeout_s`` while leaving every other
        threshold at its training-tuned value.

        No internal retries — the caller (provider) decides whether to
        recreate the pod on failure.
        """
        policy = TRAINING_PROFILE if timeout is None else replace(TRAINING_PROFILE, total_timeout_s=int(timeout))
        waiter = PodSshWaiter(query=self._query, policy=policy)
        logger.info(f"⏳ Waiting for pod {pod_id} to be ready (timeout: {policy.total_timeout_s}s)...")
        return waiter.wait(pod_id)

    def check_health(self, pod_id: str) -> Result[bool, ProviderError]:
        """Quick health check — returns ``True`` iff status == ``RUNNING``."""
        result = self._query.query_pod_snapshot(pod_id)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        snapshot = result.unwrap()
        return Ok(snapshot.status == "RUNNING")


__all__ = ["PodLifecycleManager"]

"""RunPod Cleanup Manager — terminates pods with bounded retry.

Why retry lives in the cleanup manager (and not in the API client):

  * The API client is the **transport** layer — one call, one HTTP
    round-trip. Adding retry there would make every other caller
    (status polls, listing, etc.) inherit retry semantics they never
    asked for.
  * The cleanup manager is the **policy** layer — it owns the
    "make sure the pod is gone" guarantee. Retry is part of that
    policy: a transient RunPod 5xx or network blip during termination
    must not result in a leaked pod (= leaked $$$ for the user).

Retry policy:

  * **3 attempts total** (initial + 2 retries).
  * **Exponential backoff** between attempts: 2s, then 4s. Max 6s of
    sleep across the full chain — small enough that operators waiting
    on shutdown don't notice; large enough to ride out a transient
    platform issue.
  * **Retry on any failure** — we don't have a clean retryable-vs-
    permanent distinction at this layer. Network blips, transient 5xx,
    auth-renewal races, SDK transport errors all benefit from retry.
    Permanent failures (pod_id never existed) cost only 6s extra and
    surface the same error at the end.
  * **Preserve last error** — final raise carries the most recent
    failure, since "what's preventing cleanup right now" is the most
    actionable signal for the operator.

The ``sleep_fn`` parameter is injectable so unit tests don't actually
sleep (mirrors ``job_client``'s ``sleep`` parameter on its WS reconnect
loop).

Phase A2 Batch 11 (2026-05-15): migrated from
``Result[None, ProviderError]`` to raise-based contract. The underlying
api client now raises typed exceptions; on success we return ``None``,
and on exhaustion we re-raise the last typed exception.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from ryotenkai_shared.errors import RyotenkAIError
from ryotenkai_shared.utils.logger import logger

# ---------------------------------------------------------------------------
# Retry tuning — module-level constants so tests can monkey-patch and
# operators can find them via grep without spelunking through methods.
# ---------------------------------------------------------------------------

#: Maximum number of attempts (initial + retries). 3 = initial + 2 retries.
_MAX_ATTEMPTS: int = 3

#: Initial backoff in seconds. Doubled on each subsequent retry.
_INITIAL_BACKOFF_S: float = 2.0

#: Cap on individual sleep so a misconfigured ``_MAX_ATTEMPTS`` can't
#: produce hour-long sleeps. 30s is "noticeable but not catastrophic".
_MAX_BACKOFF_S: float = 30.0


class _PodTerminateControl(Protocol):
    def terminate_pod(self, pod_id: str) -> None: ...


class RunPodCleanupManager:
    """Terminate RunPod pods with bounded retry on transient failures.

    Holds a reference to a ``_PodTerminateControl`` (the API client) and
    layers retry policy on top of it. Idempotent at the call-site level:
    the underlying SDK already treats "already gone" as success, so this
    layer doesn't need to special-case it.

    Phase A2 Batch 11: raise-based contract. ``cleanup_pod`` returns
    ``None`` on success and raises the last typed exception on
    exhaustion.
    """

    def __init__(
        self,
        api_client: _PodTerminateControl,
        *,
        max_attempts: int = _MAX_ATTEMPTS,
        initial_backoff_s: float = _INITIAL_BACKOFF_S,
        max_backoff_s: float = _MAX_BACKOFF_S,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {max_attempts!r}"
            )
        if initial_backoff_s < 0:
            raise ValueError(
                f"initial_backoff_s must be >= 0, got {initial_backoff_s!r}"
            )
        if max_backoff_s < initial_backoff_s:
            raise ValueError(
                f"max_backoff_s ({max_backoff_s}) must be >= initial_backoff_s "
                f"({initial_backoff_s})"
            )
        self.api_client = api_client
        self._max_attempts = max_attempts
        self._initial_backoff_s = initial_backoff_s
        self._max_backoff_s = max_backoff_s
        self._sleep_fn = sleep_fn

    def cleanup_pod(self, pod_id: str) -> None:
        """Terminate a pod, retrying on transient failures.

        Returns ``None`` on success. Raises the LAST typed exception
        encountered when all attempts are exhausted.

        Total wall-clock cost on full failure with default tuning:
        ``initial_call + 2s + retry_call + 4s + retry_call``. The
        underlying SDK calls already have their own per-call timeouts.
        """
        logger.warning(f"🗑️ Cleaning up pod {pod_id}...")

        last_exc: RyotenkAIError | None = None
        backoff_s = self._initial_backoff_s

        for attempt in range(1, self._max_attempts + 1):
            try:
                self.api_client.terminate_pod(pod_id)
                if attempt > 1:
                    logger.info(
                        f"✅ Pod {pod_id} terminated on retry attempt {attempt}/"
                        f"{self._max_attempts}"
                    )
                return None
            except RyotenkAIError as exc:
                last_exc = exc

            # No more attempts left — surface the last error to the caller.
            if attempt >= self._max_attempts:
                code = last_exc.context.get("code", last_exc.code.value) if last_exc else "?"
                msg = last_exc.detail if last_exc and last_exc.detail else str(last_exc)
                logger.error(
                    f"❌ Pod {pod_id} cleanup failed after {self._max_attempts} "
                    f"attempts. Last error: [{code}] {msg}. "
                    f"Verify in the RunPod console — the pod may need manual cleanup."
                )
                # Re-raise the last typed exception verbatim.
                assert last_exc is not None
                raise last_exc

            # Backoff before retry. Capped so misconfiguration can't sleep
            # forever; doubled per attempt for exponential behaviour.
            sleep_s = min(backoff_s, self._max_backoff_s)
            code = last_exc.context.get("code", last_exc.code.value) if last_exc else "?"
            msg = last_exc.detail if last_exc and last_exc.detail else str(last_exc)
            logger.warning(
                f"⏳ Pod {pod_id} cleanup attempt {attempt}/{self._max_attempts} "
                f"failed: [{code}] {msg}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            self._sleep_fn(sleep_s)
            backoff_s *= 2

        # Defensive — the loop body always returns/raises when ``attempt >=
        # max_attempts``. This branch is unreachable but keeps mypy happy.
        assert last_exc is not None
        raise last_exc


def create_cleanup_manager(api_base: str, api_key: str) -> RunPodCleanupManager:
    """Create a cleanup manager instance with default retry tuning.

    Args:
        api_base: RunPod API base URL.
        api_key: RunPod API key.

    Returns:
        ``RunPodCleanupManager`` ready to terminate pods with bounded
        retry on transient failures.
    """
    from ryotenkai_providers.runpod.training.api_client import RunPodAPIClient

    api_client = RunPodAPIClient(api_base_url=api_base, api_key=api_key)
    return RunPodCleanupManager(api_client)


__all__ = ["RunPodCleanupManager", "create_cleanup_manager"]

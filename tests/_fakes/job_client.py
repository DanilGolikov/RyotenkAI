"""``FakeJobClient`` — canonical fake for :class:`IJobClient`.

In-memory HTTP-shape client toward the runner. The fake holds a
``jobs`` dict and replays the runner's REST shape (submit, get_status,
request_stop, heartbeat, health) without spinning up uvicorn.

Chaos surface:

* :meth:`inject_429` — next N calls raise :class:`JobClientRateLimitedError`
* :meth:`inject_timeout` — next N calls raise
  :class:`JobClientNetworkError`
* :meth:`inject_404_next` — next status / stop returns 404
* :meth:`inject_network_partition(duration_seconds)` — every call
  raises :class:`JobClientNetworkError` until the clock advances past
  the deadline
* :meth:`set_unhealthy` — ``health_check`` returns ``False``
* :meth:`reset_chaos`
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from ryotenkai_shared.infrastructure.job_client import (
    IJobClient,
    JobClientNetworkError,
    JobClientNotFoundError,
    JobClientRateLimitedError,
    JobSubmissionResult,
)
from tests._harness.clock import Clock, RealClock


@dataclass
class _ChaosState:
    rate_limited_remaining: int = 0
    timeout_remaining: int = 0
    not_found_next: bool = False
    network_partition_until: float | None = None
    healthy: bool = True


@dataclass
class _JobRecord:
    job_id: str
    sequence: int
    offset: int
    state: str = "preparing"
    spec: dict[str, Any] = field(default_factory=dict)
    plugins_size: int = 0
    stop_requested: bool = False
    grace_seconds: float | None = None


class FakeJobClient:
    """Deterministic in-memory fake for :class:`IJobClient`."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else RealClock()
        self._jobs: dict[str, _JobRecord] = {}
        self._chaos = _ChaosState()
        self._id_counter = itertools.count(start=1)
        self._heartbeat_count = 0
        self._call_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_429(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.rate_limited_remaining = count

    def inject_timeout(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.timeout_remaining = count

    def inject_404_next(self) -> None:
        self._chaos.not_found_next = True

    def inject_network_partition(self, duration_seconds: float) -> None:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        self._chaos.network_partition_until = self._clock.now() + duration_seconds

    def set_unhealthy(self, value: bool = True) -> None:
        self._chaos.healthy = not value

    def reset_chaos(self) -> None:
        # Preserve healthy=True default to match constructor state.
        self._chaos = _ChaosState()

    # ------------------------------------------------------------------
    # Inspection helpers (test-only)
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[str]:
        return list(self._jobs.keys())

    def get_record(self, job_id: str) -> _JobRecord:
        return self._jobs[job_id]

    def heartbeat_count(self) -> int:
        return self._heartbeat_count

    def call_history(self) -> list[dict[str, Any]]:
        return list(self._call_history)

    def force_job_state(self, job_id: str, state: str) -> None:
        self._jobs[job_id].state = state

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "jobs": {
                jid: {
                    "sequence": r.sequence,
                    "offset": r.offset,
                    "state": r.state,
                    "stop_requested": r.stop_requested,
                    "grace_seconds": r.grace_seconds,
                    "plugins_size": r.plugins_size,
                }
                for jid, r in self._jobs.items()
            },
            "heartbeat_count": self._heartbeat_count,
            "chaos": {
                "rate_limited_remaining": self._chaos.rate_limited_remaining,
                "timeout_remaining": self._chaos.timeout_remaining,
                "not_found_next": self._chaos.not_found_next,
                "network_partition_until": self._chaos.network_partition_until,
                "healthy": self._chaos.healthy,
            },
            "call_history": list(self._call_history),
        }

    # ------------------------------------------------------------------
    # IJobClient surface
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        # ``health_check`` mirrors :meth:`JobClient.health_check`: it
        # swallows transport errors and returns False so callers can
        # do simple polling loops.
        self._call_history.append({"call": "health_check"})
        if self._is_partitioned():
            return False
        if self._chaos.timeout_remaining > 0:
            self._chaos.timeout_remaining -= 1
            return False
        return self._chaos.healthy

    async def submit_job(
        self,
        job_spec: dict[str, Any],
        *,
        plugins_payload: bytes | None = None,
        timeout: float | None = None,
    ) -> JobSubmissionResult:
        self._call_history.append({"call": "submit_job", "spec": dict(job_spec)})
        self._fire_chaos(allow_not_found=False)
        job_id = job_spec.get("job_id") or f"j-{next(self._id_counter):04d}"
        record = _JobRecord(
            job_id=job_id,
            sequence=0,
            offset=0,
            state="preparing",
            spec=dict(job_spec),
            plugins_size=len(plugins_payload or b""),
        )
        self._jobs[job_id] = record
        return JobSubmissionResult(job_id=job_id, sequence=0, offset=0)

    async def get_status(self, job_id: str) -> dict[str, Any]:
        self._call_history.append({"call": "get_status", "job_id": job_id})
        self._fire_chaos(allow_not_found=True)
        if job_id not in self._jobs:
            raise JobClientNotFoundError(f"unknown job: {job_id!r}")
        rec = self._jobs[job_id]
        return {
            "job_id": rec.job_id,
            "state": rec.state,
            "sequence": rec.sequence,
            "offset": rec.offset,
            "stop_requested": rec.stop_requested,
        }

    async def request_stop(
        self,
        job_id: str,
        *,
        grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._call_history.append({"call": "request_stop", "job_id": job_id})
        self._fire_chaos(allow_not_found=True)
        if job_id not in self._jobs:
            raise JobClientNotFoundError(f"unknown job: {job_id!r}")
        rec = self._jobs[job_id]
        rec.stop_requested = True
        rec.grace_seconds = grace_seconds
        rec.state = "stopping"
        return {"job_id": job_id, "state": "stopping"}

    async def send_heartbeat(
        self, *, ttl_seconds: float | None = None,
    ) -> bool:
        self._call_history.append(
            {"call": "send_heartbeat", "ttl_seconds": ttl_seconds},
        )
        if self._is_partitioned():
            return False
        if self._chaos.timeout_remaining > 0:
            self._chaos.timeout_remaining -= 1
            return False
        if self._chaos.rate_limited_remaining > 0:
            # heartbeat-on-429 is fire-and-forget — record it but
            # mirror JobClient's "transport error → False" semantics.
            self._chaos.rate_limited_remaining -= 1
            return False
        self._heartbeat_count += 1
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fire_chaos(self, *, allow_not_found: bool) -> None:
        if self._is_partitioned():
            raise JobClientNetworkError("fake injected network partition")
        if self._chaos.timeout_remaining > 0:
            self._chaos.timeout_remaining -= 1
            raise JobClientNetworkError("fake injected timeout")
        if self._chaos.rate_limited_remaining > 0:
            self._chaos.rate_limited_remaining -= 1
            raise JobClientRateLimitedError("fake injected 429")
        if allow_not_found and self._chaos.not_found_next:
            self._chaos.not_found_next = False
            raise JobClientNotFoundError("fake injected 404")

    def _is_partitioned(self) -> bool:
        if self._chaos.network_partition_until is None:
            return False
        return self._clock.now() < self._chaos.network_partition_until


# Static guarantee.
_runtime_check: IJobClient = FakeJobClient()


__all__ = [
    "FakeJobClient",
]

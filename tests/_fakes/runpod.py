"""``FakeRunPodAPI`` — canonical fake for :class:`IRunPodAPI`.

In-memory pod registry + GraphQL stub. Determinism through injected
:class:`Clock`. Chaos surface:

* :meth:`inject_429` — raise :class:`RunPodRateLimitedError` for the
  next N calls
* :meth:`inject_5xx` — raise :class:`RunPodTransientError` for the
  next N calls
* :meth:`inject_partial_response` — next N read calls return / raise
  :class:`RunPodPartialResponseError` (sometimes the parser tolerates a
  partial; sometimes the call returns a sentinel ``RunPodInfo`` with
  most fields ``None``)
* :meth:`set_hibernation_mode` — mark a pod as hibernated; subsequent
  ``find_pod`` calls reflect the new state
* :meth:`reset_chaos`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ryotenkai_shared.infrastructure.runpod_api import (
    IRunPodAPI,
    RunPodInfo,
    RunPodLifecycleResponse,
    RunPodPartialResponseError,
    RunPodRateLimitedError,
    RunPodTransientError,
)
from tests._harness.clock import Clock, RealClock

# Custom RunPod-specific status, surfaced via ``set_hibernation_mode``.
_HIBERNATED = "HIBERNATED"


@dataclass
class _RegisteredPod:
    info: RunPodInfo
    hibernated: bool = False


@dataclass
class _ChaosState:
    rate_limit_remaining: int = 0
    transient_remaining: int = 0
    partial_remaining: int = 0


class FakeRunPodAPI:
    """In-memory deterministic fake for :class:`IRunPodAPI`."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else RealClock()
        self._pods: dict[str, _RegisteredPod] = {}
        self._chaos = _ChaosState()
        self._call_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pod registry
    # ------------------------------------------------------------------

    def register_pod(self, info: RunPodInfo) -> None:
        self._pods[info.pod_id] = _RegisteredPod(info=info)

    def upsert_pod(
        self,
        pod_id: str,
        *,
        desired_status: str = "RUNNING",
        ssh_host: str | None = "127.0.0.1",
        ssh_port: int | None = 22000,
        machine_id: str | None = None,
        cost_per_hr: float | None = 0.5,
    ) -> RunPodInfo:
        info = RunPodInfo(
            pod_id=pod_id,
            desired_status=desired_status,
            runtime_status=desired_status,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            machine_id=machine_id,
            cost_per_hr=cost_per_hr,
        )
        self.register_pod(info)
        return info

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_429(self, count: int = 5) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.rate_limit_remaining = count

    def inject_5xx(self, count: int = 3) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.transient_remaining = count

    def inject_partial_response(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.partial_remaining = count

    def set_hibernation_mode(self, pod_id: str) -> None:
        if pod_id not in self._pods:
            self._pods[pod_id] = _RegisteredPod(
                info=RunPodInfo(pod_id=pod_id, desired_status=_HIBERNATED, runtime_status=_HIBERNATED),
            )
        else:
            existing = self._pods[pod_id].info
            self._pods[pod_id].info = RunPodInfo(
                pod_id=existing.pod_id,
                desired_status=_HIBERNATED,
                runtime_status=_HIBERNATED,
                ssh_host=existing.ssh_host,
                ssh_port=existing.ssh_port,
                cost_per_hr=existing.cost_per_hr,
                machine_id=existing.machine_id,
                extras=existing.extras,
            )
            self._pods[pod_id].hibernated = True

    def reset_chaos(self) -> None:
        self._chaos = _ChaosState()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "pods": {
                pod_id: {
                    "desired_status": entry.info.desired_status,
                    "runtime_status": entry.info.runtime_status,
                    "ssh_host": entry.info.ssh_host,
                    "ssh_port": entry.info.ssh_port,
                    "cost_per_hr": entry.info.cost_per_hr,
                    "machine_id": entry.info.machine_id,
                    "hibernated": entry.hibernated,
                }
                for pod_id, entry in self._pods.items()
            },
            "chaos": {
                "rate_limit_remaining": self._chaos.rate_limit_remaining,
                "transient_remaining": self._chaos.transient_remaining,
                "partial_remaining": self._chaos.partial_remaining,
            },
            "call_history": list(self._call_history),
        }

    # ------------------------------------------------------------------
    # IRunPodAPI surface
    # ------------------------------------------------------------------

    async def find_pod(self, pod_id: str) -> RunPodInfo | None:
        self._fire_chaos(read=True)
        self._call_history.append({"call": "find_pod", "pod_id": pod_id})
        entry = self._pods.get(pod_id)
        return entry.info if entry is not None else None

    async def list_pods(self) -> list[RunPodInfo]:
        self._fire_chaos(read=True)
        self._call_history.append({"call": "list_pods"})
        return [entry.info for entry in self._pods.values()]

    async def stop_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        self._fire_chaos(read=False)
        self._call_history.append({"call": "stop_pod", "pod_id": pod_id})
        entry = self._pods.get(pod_id)
        if entry is None:
            return RunPodLifecycleResponse(outcome="not_found", message=f"unknown pod {pod_id}")
        if entry.info.desired_status in ("EXITED", "STOPPED", "TERMINATED"):
            return RunPodLifecycleResponse(outcome="already_done")
        self._pods[pod_id].info = self._with_status(entry.info, "EXITED")
        return RunPodLifecycleResponse(outcome="ok")

    async def terminate_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        self._fire_chaos(read=False)
        self._call_history.append({"call": "terminate_pod", "pod_id": pod_id})
        entry = self._pods.get(pod_id)
        if entry is None:
            return RunPodLifecycleResponse(outcome="already_done", message="not found")
        if entry.info.desired_status == "TERMINATED":
            return RunPodLifecycleResponse(outcome="already_done")
        self._pods[pod_id].info = self._with_status(entry.info, "TERMINATED")
        return RunPodLifecycleResponse(outcome="ok")

    async def resume_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        self._fire_chaos(read=False)
        self._call_history.append({"call": "resume_pod", "pod_id": pod_id})
        entry = self._pods.get(pod_id)
        if entry is None:
            return RunPodLifecycleResponse(outcome="not_found", message=f"unknown pod {pod_id}")
        if entry.info.desired_status == "TERMINATED":
            return RunPodLifecycleResponse(
                outcome="failed", message="cannot resume a terminated pod",
            )
        if entry.info.desired_status == "RUNNING":
            return RunPodLifecycleResponse(outcome="already_done")
        self._pods[pod_id].info = self._with_status(entry.info, "RUNNING")
        self._pods[pod_id].hibernated = False
        return RunPodLifecycleResponse(outcome="ok")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fire_chaos(self, *, read: bool) -> None:
        # WHY ordering: rate-limit first (cheapest path on real upstream),
        # transient second, partial third.
        if self._chaos.rate_limit_remaining > 0:
            self._chaos.rate_limit_remaining -= 1
            raise RunPodRateLimitedError("fake injected 429")
        if self._chaos.transient_remaining > 0:
            self._chaos.transient_remaining -= 1
            raise RunPodTransientError("fake injected 5xx")
        if read and self._chaos.partial_remaining > 0:
            self._chaos.partial_remaining -= 1
            raise RunPodPartialResponseError(
                "fake injected partial GraphQL payload (missing fields)",
            )

    @staticmethod
    def _with_status(info: RunPodInfo, status: str) -> RunPodInfo:
        return RunPodInfo(
            pod_id=info.pod_id,
            desired_status=status,
            runtime_status=status,
            ssh_host=info.ssh_host,
            ssh_port=info.ssh_port,
            cost_per_hr=info.cost_per_hr,
            machine_id=info.machine_id,
            extras=info.extras,
        )


# Static guarantee — fail fast at module import if the fake drifts from
# the Protocol shape.
_runtime_check: IRunPodAPI = FakeRunPodAPI()


__all__ = [
    "FakeRunPodAPI",
]

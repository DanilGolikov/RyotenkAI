"""Hibernation detector — sweep RunPod and surface hibernated pods.

Phase 1 greenfield helper. RunPod's status vocabulary includes a
``HIBERNATED`` state that operators sometimes need to detect and
either resume or terminate proactively (cf. plan
:doc:`structured-hopping-starfish` chaos scenario
``PodHibernationWhileMacAwake``).

Tolerates :class:`RunPodRateLimitedError` / :class:`RunPodTransientError`
by surfacing them in the report rather than aborting the sweep — a
single rate-limited list call shouldn't lose the partial inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ryotenkai_shared.infrastructure.runpod_api import (
    RunPodAPIError,
    RunPodInfo,
)

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.runpod_api import IRunPodAPI


_HIBERNATED_STATUSES: frozenset[str] = frozenset({"HIBERNATED"})


@dataclass(frozen=True)
class HibernatedPodInfo:
    pod_id: str
    desired_status: str


@dataclass(frozen=True)
class HibernationSweepReport:
    hibernated: tuple[HibernatedPodInfo, ...] = field(default_factory=tuple)
    api_errors: tuple[str, ...] = field(default_factory=tuple)
    inspected: int = 0


class HibernationDetector:
    """Sweep the RunPod fleet for HIBERNATED pods."""

    def __init__(self, *, api: IRunPodAPI) -> None:
        self._api = api

    async def sweep(self) -> HibernationSweepReport:
        try:
            pods = await self._api.list_pods()
        except RunPodAPIError as exc:
            return HibernationSweepReport(api_errors=(repr(exc),))
        hibernated: list[HibernatedPodInfo] = []
        for info in pods:
            if self._is_hibernated(info):
                hibernated.append(
                    HibernatedPodInfo(pod_id=info.pod_id, desired_status=info.desired_status),
                )
        return HibernationSweepReport(
            hibernated=tuple(hibernated),
            inspected=len(pods),
        )

    async def is_pod_hibernated(self, pod_id: str) -> bool | None:
        """Return ``True``/``False`` if known, ``None`` if not found.

        Distinguishes "pod is alive and not hibernated" from "pod doesn't
        exist" because the cleanup workflow handles them differently.
        """
        try:
            info = await self._api.find_pod(pod_id)
        except RunPodAPIError:
            return None
        if info is None:
            return None
        return self._is_hibernated(info)

    @staticmethod
    def _is_hibernated(info: RunPodInfo) -> bool:
        return info.desired_status in _HIBERNATED_STATUSES


__all__ = ["HibernatedPodInfo", "HibernationDetector", "HibernationSweepReport"]

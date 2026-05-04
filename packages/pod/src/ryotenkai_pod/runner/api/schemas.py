"""[DEPRECATED] Wire schemas for the runner HTTP / WebSocket surface.

Phase 0 transport-unification-v2 moved the canonical definitions to
:mod:`ryotenkai_shared.contracts.runner_api` so the Mac-side client
(in ``ryotenkai_shared.utils.clients.job_client``) and the pod-side
runner (here) consume one source of truth.

This module is kept as a thin alias re-export for **PR-0a** so
existing imports in this package keep working. **PR-0b** (next PR)
migrates all import sites to ``ryotenkai_shared.contracts.runner_api``
and removes this file. New code MUST import from
``ryotenkai_shared.contracts.runner_api``.
"""

from __future__ import annotations

from ryotenkai_shared.contracts.runner_api import (
    EventResponse,
    InternalEventRequest,
    JobSnapshotResponse,
    JobSpec,
    JobStopAcceptedResponse,
    JobSubmittedResponse,
)

__all__ = [
    "EventResponse",
    "InternalEventRequest",
    "JobSnapshotResponse",
    "JobSpec",
    "JobStopAcceptedResponse",
    "JobSubmittedResponse",
]

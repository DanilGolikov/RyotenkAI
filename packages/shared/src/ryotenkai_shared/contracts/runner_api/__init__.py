"""Wire DTOs for the in-pod uvicorn runner — single source of truth.

Both pod-side (``ryotenkai_pod.runner.api.*``) and Mac-side
(``ryotenkai_shared.utils.clients.job_client``) import from here.
Lives in the leaf ``ryotenkai_shared`` package so the
``control → pod`` import contract stays clean.

Re-exports the full surface so callers can do
``from ryotenkai_shared.contracts.runner_api import JobSpec, ...``.
"""

from .control import ControlHeartbeatRequest, ControlHeartbeatResponse
from .diagnostics import (
    DiagnosticsBlockError,
    DiagnosticsInclude,
    DiagnosticsResponse,
    DmesgReport,
    GpuReport,
    GpuRow,
    KernelSignalsReport,
)
from .events import (
    WS_CLOSE_GONE,
    WS_CLOSE_INVALID,
    WS_CLOSE_NOT_FOUND,
    EventResponse,
)
from .internal import InternalEventRequest
from .logs import LogChunkResponse, LogName, LogSizeResponse
from .resources import ResourceSnapshot
from .jobs import (
    JobSnapshotResponse,
    JobSpec,
    JobStopAcceptedResponse,
    JobSubmittedResponse,
)

__all__ = [
    "ControlHeartbeatRequest",
    "ControlHeartbeatResponse",
    "DiagnosticsBlockError",
    "DiagnosticsInclude",
    "DiagnosticsResponse",
    "DmesgReport",
    "EventResponse",
    "GpuReport",
    "GpuRow",
    "InternalEventRequest",
    "JobSnapshotResponse",
    "JobSpec",
    "JobStopAcceptedResponse",
    "JobSubmittedResponse",
    "KernelSignalsReport",
    "LogChunkResponse",
    "LogName",
    "LogSizeResponse",
    "ResourceSnapshot",
    "WS_CLOSE_GONE",
    "WS_CLOSE_INVALID",
    "WS_CLOSE_NOT_FOUND",
]

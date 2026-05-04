"""Resource snapshot DTO (Phase 2 PR-2.2).

``GET /api/v1/resources`` — instant GPU/CPU/RAM snapshot used by
the Mac orchestrator to render the live status line and feed the
load chart. Mirrors the ``health_snapshot`` event shape already
published by :class:`HealthReporter` so frontends can switch
between WS-driven and HTTP-driven sources without schema
translation.
"""

from __future__ import annotations

from pydantic import Field

from ._strict import _StrictModel


class ResourceSnapshot(_StrictModel):
    """Instant GPU + CPU + RAM telemetry snapshot.

    Field semantics mirror the existing ``health_snapshot`` event
    payload:

    * ``gpu_util_percent``   — 0..100, ``None`` when nvidia-smi missing
    * ``gpu_memory_percent`` — 0..100 of total VRAM
    * ``cpu_percent``        — 0..100 process-aggregated
    * ``ram_used_gb``        — container-aware via cgroup files
    * ``ram_total_gb``       — container-aware via cgroup files

    Why ``None`` rather than 0: distinguishes "tool unavailable" from
    "0% utilization right now". Mac status line renders ``--`` for
    ``None`` so operators don't misread an unavailable metric.
    """

    gpu_util_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_memory_percent: float | None = Field(default=None, ge=0, le=100)
    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    ram_used_gb: float | None = Field(default=None, ge=0)
    ram_total_gb: float | None = Field(default=None, ge=0)


__all__ = ["ResourceSnapshot"]

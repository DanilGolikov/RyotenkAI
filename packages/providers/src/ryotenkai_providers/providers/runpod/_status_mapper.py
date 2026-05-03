"""Phase 14.C — RunPod GraphQL ``desiredStatus`` → :class:`PodAvailability`.

Lives at the package boundary (sibling of ``__init__.py``) rather than
inside ``training/`` so importing it does NOT trigger the heavy
``training/__init__.py`` chain (which pulls api_client → sdk_adapter →
the optional ``runpod`` SDK). That chain is fine in production but
breaks slim CI venvs that only test the shared layer.

This module is the canonical RunPod-side owner of the status
vocabulary. Mac-side resume-flow callers
(:class:`PodAvailabilityProbe`, :class:`LaunchResumeService`)
consume :func:`map_runpod_desired_status_to_availability` via lazy
import; the broader RunPod provider re-exports it for symmetry
(``from src.providers.runpod.training.provider import
map_runpod_desired_status_to_availability`` keeps working).

Phase 14.C migration: replaces the pre-14.C
``_RUNPOD_STATUS_MAP`` dict that lived in
:mod:`src.pipeline.launch.pod_availability`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.pipeline.launch.pod_availability import PodAvailability


def _build_map() -> "dict[str, PodAvailability]":
    # Local import — pod_availability.py is light (no provider chain),
    # so this is cheap; deferring keeps the module-load order
    # symmetric with the rest of the pipeline.launch package.
    from src.pipeline.launch.pod_availability import PodAvailability
    return {
        "RUNNING": PodAvailability.RUNNING,
        "EXITED": PodAvailability.SLEEPING_RESUMABLE,
        "STOPPED": PodAvailability.SLEEPING_RESUMABLE,
        "PAUSED": PodAvailability.SLEEPING_RESUMABLE,
        "TERMINATED": PodAvailability.GONE,
        "DEAD": PodAvailability.GONE,
    }


def map_runpod_desired_status_to_availability(
    raw_status: str,
) -> "PodAvailability":
    """Map a RunPod GraphQL ``desiredStatus`` to
    :class:`PodAvailability`.

    Unknown statuses → :data:`PodAvailability.PROBE_FAILED`. Empty
    or whitespace input → ``PROBE_FAILED`` (callers should treat
    "no status field" before reaching this helper).

    Case-insensitive — operators sometimes get back
    lowercased / mixed-case status strings depending on the
    upstream SDK version.
    """
    from src.pipeline.launch.pod_availability import PodAvailability
    return _build_map().get(
        raw_status.upper(), PodAvailability.PROBE_FAILED,
    )


__all__: Final = ("map_runpod_desired_status_to_availability",)

"""Pod availability vocabulary.

Lived under ``ryotenkai_control.pipeline.launch.pod_availability`` until
ADR row 9 (Phase C drift fix). Provider adapters need the enum for
status-mapping (RunPod statuses → high-level availability states), but
were forbidden from importing control. Move keeps the probe / resume
logic on the control side; only the contract enum lives here.
"""

from __future__ import annotations

from enum import StrEnum


class PodAvailability(StrEnum):
    """Coarse availability states the probe maps RunPod statuses into.

    The values are stable strings — operator dashboards and Web UI
    badges grep on them, so renaming is a contract change.
    """

    RUNNING = "running"
    SLEEPING_RESUMABLE = "sleeping_resumable"
    SLEEPING_RESUME_FAILED = "sleeping_resume_failed"
    GONE = "gone"
    PROBE_FAILED = "probe_failed"

    @property
    def is_resume_needed(self) -> bool:
        """True iff the caller should call ``resume_pod_with_retry``."""
        return self == PodAvailability.SLEEPING_RESUMABLE

    @property
    def is_recoverable(self) -> bool:
        """True iff the caller can usefully proceed with the pipeline.

        RUNNING ⇒ no action needed; ModelRetriever can SSH right away.
        SLEEPING_RESUMABLE ⇒ resume first, then proceed.
        Other states ⇒ user intervention required.
        """
        return self in (
            PodAvailability.RUNNING,
            PodAvailability.SLEEPING_RESUMABLE,
        )


__all__ = ["PodAvailability"]

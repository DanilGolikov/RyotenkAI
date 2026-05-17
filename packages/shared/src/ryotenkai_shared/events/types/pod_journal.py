"""Pod-domain journal-housekeeping events.

Two events that report the state of the on-disk JSONL journal used as
the pod-side recovery store: rotation (when one file closes and the
next opens) and disk pressure (the journal footprint crossed a
percentage of its capacity). Producer: ``ryotenkai_pod.runner``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class JournalRotatedPayload(BaseModel):
    """Single rotation hop. ``oldest_remaining_seq`` is ``None`` when the
    rotation deleted the only previous file and nothing older is left.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_seq: int
    to_seq: int
    file_size_bytes: int
    oldest_remaining_seq: int | None = None


class JournalRotatedEvent(BaseEvent):
    """Journal closed file ``from_seq`` and opened ``to_seq``."""

    kind: Literal["ryotenkai.pod.journal.rotated"] = "ryotenkai.pod.journal.rotated"
    severity: Literal["info"] = "info"
    payload: JournalRotatedPayload


class JournalDiskPressurePayload(BaseModel):
    """Health-check observation that the journal is close to capacity.

    ``error_type`` is a free-form short tag (e.g. ``"footprint"``,
    ``"write_failure"``) so consumers can group sources without parsing
    a message string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str
    total_bytes: int | None = None
    file_count: int | None = None
    threshold_bytes: int | None = None
    cap_bytes: int | None = None


class JournalDiskPressureEvent(BaseEvent):
    """Journal footprint crossed the disk-pressure watermark."""

    kind: Literal["ryotenkai.pod.journal.disk_pressure"] = (
        "ryotenkai.pod.journal.disk_pressure"
    )
    severity: Literal["warning"] = "warning"
    payload: JournalDiskPressurePayload


__all__ = [
    "JournalDiskPressureEvent",
    "JournalDiskPressurePayload",
    "JournalRotatedEvent",
    "JournalRotatedPayload",
]

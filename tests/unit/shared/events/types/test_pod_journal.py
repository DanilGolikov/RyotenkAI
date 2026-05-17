"""Unit tests: pod-journal event types (Phase 2).

Two concrete event classes covering on-disk journal housekeeping:

* :class:`JournalRotatedEvent` — rotation hop with file sequence info.
* :class:`JournalDiskPressureEvent` — health-check footprint warning.

Both follow the same per-class shape as the rest of the typed taxonomy:
``kind`` and ``severity`` are pinned via ``Literal`` defaults, payload
is frozen with ``extra=forbid``, and the envelope round-trips through
the codec.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.pod_journal import (
    JournalDiskPressureEvent,
    JournalDiskPressurePayload,
    JournalRotatedEvent,
    JournalRotatedPayload,
)


def _rotated() -> JournalRotatedEvent:
    return JournalRotatedEvent(
        source="pod://runner/journal",
        run_id="r",
        offset=0,
        payload=JournalRotatedPayload(
            from_seq=1,
            to_seq=2,
            file_size_bytes=1024,
            oldest_remaining_seq=0,
        ),
    )


def _disk_pressure() -> JournalDiskPressureEvent:
    return JournalDiskPressureEvent(
        source="pod://runner/journal",
        run_id="r",
        offset=1,
        payload=JournalDiskPressurePayload(
            error_type="footprint",
            total_bytes=512 * 1024 * 1024,
            file_count=5,
            threshold_bytes=450 * 1024 * 1024,
            cap_bytes=500 * 1024 * 1024,
        ),
    )


_ALL = [_rotated, _disk_pressure]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_disk_pressure_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JournalDiskPressurePayload(error_type="x", extra="oops")  # type: ignore[call-arg]


class TestInvariants:
    def test_rotated_pins_kind_and_severity(self) -> None:
        ev = _rotated()
        assert ev.kind == "ryotenkai.pod.journal.rotated"
        assert ev.severity == "info"

    def test_disk_pressure_pins_kind_and_severity(self) -> None:
        ev = _disk_pressure()
        assert ev.kind == "ryotenkai.pod.journal.disk_pressure"
        assert ev.severity == "warning"

    def test_oldest_remaining_seq_optional(self) -> None:
        # When the journal rotation deleted the only previous file the
        # callback fires with ``oldest_remaining_seq=None``.
        ev = JournalRotatedEvent(
            source="pod://runner/journal",
            run_id="r",
            offset=2,
            payload=JournalRotatedPayload(
                from_seq=0, to_seq=1, file_size_bytes=100,
            ),
        )
        assert ev.payload.oldest_remaining_seq is None

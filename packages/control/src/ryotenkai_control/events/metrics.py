"""Phase 8 — observability snapshot of the event subsystem.

The event-subsystem counters live on the individual collaborators
(:class:`ControlEventEmitter`, :class:`InMemoryBus`,
:class:`JournalWriter`, :class:`EventDedup`). This module is the
single hand-off point between those raw counters and operator-facing
surfaces (health endpoint, optional CLI command, future audit
exports).

The aggregator is a pure read-side concern — it never mutates the
collaborators. It also defensively copies dict-valued counters so a
holder that keeps the snapshot alive cannot observe a torn dict if a
producer updates the underlying counter concurrently.

The contract is intentionally narrow: a single :func:`collect_metrics`
call snapshots all four collaborators atomically (from the caller's
point of view — under the GIL each attribute read is independent, but
the structure of the snapshot is consistent within itself).

See ``docs/plans/ethereal-tumbling-patterson.md`` — Risk Ledger R-06
("Emitter silently drops events; no alerts") for the motivating
requirement.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ryotenkai_control.events.dedup import EventDedup
    from ryotenkai_control.events.emitter import ControlEventEmitter
    from ryotenkai_control.events.in_memory_bus import InMemoryBus
    from ryotenkai_control.events.journal_writer import JournalWriter


__all__ = ["EventSubsystemMetrics", "collect_metrics"]


@dataclass(frozen=True)
class EventSubsystemMetrics:
    """Snapshot of all event-subsystem counters at one point in time.

    Frozen so a snapshot can be passed around without callers mutating
    it by accident. ``dict``-valued fields are populated via
    :func:`dict` copies so a later mutation on the source counter does
    not leak into the snapshot.
    """

    # ------------------------------------------------------------------
    # Emitter
    # ------------------------------------------------------------------
    emitter_events_emitted_total: int = 0
    emitter_events_emit_failed_total: dict[str, int] = field(default_factory=dict)
    emitter_events_remote_accepted_total: int = 0
    emitter_events_remote_dropped_total: dict[str, int] = field(default_factory=dict)
    emitter_offset_collisions_detected_total: int = 0

    # ------------------------------------------------------------------
    # Bus
    # ------------------------------------------------------------------
    bus_published_total: int = 0
    bus_dropped_total: int = 0
    bus_dropped_per_consumer: dict[str, int] = field(default_factory=dict)
    bus_current_depth: int = 0
    bus_capacity: int = 0
    bus_subscriber_count: int = 0

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------
    journal_appended_total: int = 0
    journal_fsync_total: int = 0
    journal_fsync_failed_total: int = 0
    journal_total_bytes_written: int = 0
    journal_write_failed_total: int = 0
    #: ``None`` when no fsync has happened yet (newly opened writer).
    #: Otherwise ``time.monotonic() - last_fsync_at`` at snapshot time.
    journal_last_fsync_age_seconds: float | None = None

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------
    dedup_size: int = 0
    dedup_seen_total: int = 0
    dedup_hits_total: int = 0
    dedup_evicted_total: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_degraded(self) -> bool:
        """Return ``True`` when any health-indicator counter is non-zero.

        Used by the health endpoint to flip a run from ``healthy`` to
        ``degraded``. Conservative: any single failure flips the bit —
        the operator decides whether the failure is acceptable.
        """
        return (
            bool(self.emitter_events_emit_failed_total)
            or bool(self.emitter_events_remote_dropped_total)
            or self.bus_dropped_total > 0
            or any(v > 0 for v in self.bus_dropped_per_consumer.values())
            or self.journal_fsync_failed_total > 0
            or self.journal_write_failed_total > 0
            or self.emitter_offset_collisions_detected_total > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the snapshot as a JSON-friendly dict.

        The dict shape is what the health endpoint serializes into its
        ``per_run`` mapping. Stable across releases — clients (Web UI,
        CLI ``ryotenkai events metrics``) key off these names.
        """
        return asdict(self)


def collect_metrics(
    *,
    emitter: ControlEventEmitter | None = None,
    bus: InMemoryBus | None = None,
    journal: JournalWriter | None = None,
    dedup: EventDedup | None = None,
    now: float | None = None,
) -> EventSubsystemMetrics:
    """Snapshot all four collaborators into a single dataclass.

    Each collaborator is optional — a closed / not-yet-built run might
    only have a subset attached. Missing collaborators leave their
    fields at the dataclass defaults (zero counters / empty dicts),
    which the health endpoint renders as ``"0"`` columns rather than
    omitting them.

    Parameters
    ----------
    emitter, bus, journal, dedup
        Live counter holders. Pass the same emitter four times if the
        bus/journal/dedup attributes you want are nested inside it —
        the function reads attributes defensively (``getattr`` with a
        default) so partial wiring during a teardown path doesn't
        blow up.
    now
        Override clock for tests. Defaults to ``time.monotonic()``.
        Only consulted when ``journal.last_fsync_at`` is non-None.
    """
    now_ts = now if now is not None else time.monotonic()

    # Emitter ---------------------------------------------------------
    emitter_emitted = int(getattr(emitter, "events_emitted_total", 0))
    emitter_emit_failed = dict(
        getattr(emitter, "events_emit_failed_total", {}) or {},
    )
    emitter_remote_accepted = int(
        getattr(emitter, "events_remote_accepted_total", 0),
    )
    emitter_remote_dropped = dict(
        getattr(emitter, "events_remote_dropped_total", {}) or {},
    )
    emitter_collisions = int(
        getattr(emitter, "offset_collisions_detected_total", 0),
    )

    # Bus -------------------------------------------------------------
    bus_published = int(getattr(bus, "published_total", 0))
    bus_dropped_total = int(getattr(bus, "dropped_total", 0))
    bus_dropped_per_consumer = dict(
        getattr(bus, "dropped_per_consumer", {}) or {},
    )
    bus_depth = int(getattr(bus, "current_depth", 0))
    bus_capacity = int(getattr(bus, "capacity", 0))
    bus_subscribers = int(getattr(bus, "subscriber_count", 0))

    # Journal ---------------------------------------------------------
    journal_appended = int(getattr(journal, "events_appended", 0))
    journal_fsync = int(getattr(journal, "fsyncs_total", 0))
    journal_fsync_failed = int(getattr(journal, "fsync_failed_total", 0))
    journal_bytes = int(getattr(journal, "total_bytes_written", 0))
    journal_write_failed = int(getattr(journal, "write_failures_total", 0))
    journal_last_fsync_at = getattr(journal, "last_fsync_at", None)
    journal_last_fsync_age: float | None = None
    if journal_last_fsync_at is not None:
        delta = now_ts - float(journal_last_fsync_at)
        # Clamp at zero — monotonic clocks are non-decreasing within a
        # process but tests may inject a stale ``now`` to verify
        # rendering. Negative ages would be confusing on the UI.
        journal_last_fsync_age = delta if delta >= 0 else 0.0

    # Dedup -----------------------------------------------------------
    dedup_size = int(getattr(dedup, "size", 0))
    dedup_seen = int(getattr(dedup, "seen_total", 0))
    dedup_hits = int(getattr(dedup, "dedup_hits_total", 0))
    dedup_evicted = int(getattr(dedup, "evicted_total", 0))

    return EventSubsystemMetrics(
        emitter_events_emitted_total=emitter_emitted,
        emitter_events_emit_failed_total=emitter_emit_failed,
        emitter_events_remote_accepted_total=emitter_remote_accepted,
        emitter_events_remote_dropped_total=emitter_remote_dropped,
        emitter_offset_collisions_detected_total=emitter_collisions,
        bus_published_total=bus_published,
        bus_dropped_total=bus_dropped_total,
        bus_dropped_per_consumer=bus_dropped_per_consumer,
        bus_current_depth=bus_depth,
        bus_capacity=bus_capacity,
        bus_subscriber_count=bus_subscribers,
        journal_appended_total=journal_appended,
        journal_fsync_total=journal_fsync,
        journal_fsync_failed_total=journal_fsync_failed,
        journal_total_bytes_written=journal_bytes,
        journal_write_failed_total=journal_write_failed,
        journal_last_fsync_age_seconds=journal_last_fsync_age,
        dedup_size=dedup_size,
        dedup_seen_total=dedup_seen,
        dedup_hits_total=dedup_hits,
        dedup_evicted_total=dedup_evicted,
    )


def collect_metrics_for_emitter(
    emitter: ControlEventEmitter,
    *,
    now: float | None = None,
) -> EventSubsystemMetrics:
    """Convenience wrapper that pulls bus/journal/dedup off the emitter.

    The standard wiring on the orchestrator side composes the four
    collaborators inside the emitter (see
    :meth:`ControlEventEmitter.for_run`). The health endpoint walks
    the :class:`EventEmitterRegistry`, so it always has the emitter
    and never the sub-objects in isolation — this helper avoids
    duplicating the same four ``getattr(emitter, "bus", None)`` calls
    everywhere.
    """
    return collect_metrics(
        emitter=emitter,
        bus=getattr(emitter, "bus", None),
        journal=getattr(emitter, "journal", None),
        dedup=getattr(emitter, "dedup", None),
        now=now,
    )

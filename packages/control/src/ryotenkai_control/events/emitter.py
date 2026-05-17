"""Control-side :class:`IEventEmitter` implementation (Phase 3).

Composes :class:`JournalWriter` (SSOT for emitted events) +
:class:`InMemoryBus` (live fan-out for SSE/WS) + :class:`EventDedup`
(R-13 emit_remote dedup). The class is the unique writer / publisher
seen by the orchestrator's stages and by Phase 6's HTTP catchup
endpoint.

Surface (matches :class:`IEventEmitter`):

* :meth:`emit`         — locally-produced events: assigns offset,
  fills ``stage_id`` from ContextVar, validates the envelope, writes
  journal, publishes bus. Never raises.
* :meth:`emit_remote`  — remote envelopes (e.g. pod WS forward):
  dedups by ``(run_id, source, offset)``, NEVER overwrites identity
  fields, writes journal, publishes bus. Never raises.
* :meth:`stage_scope`  — ContextVar push/pop so events emitted inside
  the scope auto-fill ``stage_id``. Async-task safe (ContextVars copy
  with the task).

Construction:

* The class accepts pre-built journal/bus/dedup — easy to test, easy
  to inject alternate implementations.
* :meth:`for_run` is the convenience builder that wires the three
  collaborators from a ``run_directory``. Production callsites use it.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ryotenkai_control.events.dedup import EventDedup
from ryotenkai_control.events.dedup_sweeper import DedupTTLSweeper
from ryotenkai_control.events.in_memory_bus import InMemoryBus
from ryotenkai_control.events.journal_reader import JournalReader
from ryotenkai_control.events.journal_writer import JournalWriter
from ryotenkai_shared.events import (
    EVENT_ADAPTER,
    UNKNOWN_OFFSET,
    BaseEvent,
)
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = get_logger(__name__)


__all__ = ["DEFAULT_JOURNAL_FILENAME", "ControlEventEmitter"]


DEFAULT_JOURNAL_FILENAME = "events.jsonl"
DEFAULT_SOURCE = "control://orchestrator"


# Module-level ContextVar — async-task isolated by construction; the
# default ``copy_context`` propagation in ``asyncio.create_task`` keeps
# nested scopes correct across awaits.
_current_stage_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ryotenkai_control_current_stage_id", default=None,
)


class ControlEventEmitter:
    """Concrete control-side emitter — see module docstring."""

    def __init__(
        self,
        *,
        run_id: str,
        source: str,
        journal: JournalWriter,
        bus: InMemoryBus,
        dedup: EventDedup,
        offset_resume: dict[str, int] | None = None,
        dedup_sweeper: DedupTTLSweeper | None = None,
    ) -> None:
        self._run_id = run_id
        self._source = source
        self._journal = journal
        self._bus = bus
        self._dedup = dedup
        # Post-Phase-10 TODO #2 — background TTL sweeper. Attached
        # eagerly by :meth:`for_run`; tests that build the emitter
        # by hand can pass ``None`` to opt out (the dedup set will not
        # grow unbounded because such tests typically last < 1 s).
        self._dedup_sweeper = dedup_sweeper

        # Per-source offset counters — each (run_id, source) pair gets
        # its own monotonic counter. Locked because multiple producer
        # threads (orchestrator main + stage callbacks on the trainer
        # pump executor) can collide otherwise (R-05).
        self._offset_counters: dict[str, int] = dict(offset_resume or {})
        self._offset_lock = threading.Lock()

        # Metric counters — Phase 8 health endpoint scrapes these. Stored
        # as plain attributes so tests can assert on them directly and the
        # metrics aggregator (:mod:`ryotenkai_control.events.metrics`) can
        # snapshot them without holding a lock. The ``_failed_reasons``
        # dict maps each reason label to a count (e.g.
        # ``{"journal_write": 2, "validation": 1}``).
        self.events_emitted_total = 0
        self.events_emit_failed_total: dict[str, int] = {}
        self.events_remote_accepted_total = 0
        self.events_remote_dropped_total: dict[str, int] = {}
        # Phase 8: bumped whenever the offset lock observes a torn-read
        # write (i.e. a value > snapshotted-current sneaks in between
        # the read and the write). With the current implementation this
        # cannot happen because the entire compute-and-store is under
        # the lock — the counter exists for future implementations that
        # might drop the lock for read-mostly paths, and to give the
        # health endpoint a stable column to display.
        self.offset_collisions_detected_total = 0

        self._closed = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_run(
        cls,
        *,
        run_id: str,
        run_directory: Path | str,
        source: str = DEFAULT_SOURCE,
        capacity: int | None = None,
    ) -> ControlEventEmitter:
        """Build journal/bus/dedup from ``run_directory`` and return the emitter.

        Side effects:

        * Ensures ``run_directory`` exists (``mkdir(parents=True,
          exist_ok=True)``).
        * Runs :meth:`JournalReader.truncate_torn_tail` to repair a
          half-written last line before the writer reopens the file.
        * Seeds per-source offset counters from
          :meth:`JournalReader.newest_persisted_offset_per_source` so a
          restart resumes without offset collisions.
        * Seeds the dedup set from the journal's tail so post-restart
          remote resends still dedup correctly (R-13).
        """
        run_dir = Path(run_directory)
        run_dir.mkdir(parents=True, exist_ok=True)
        journal_path = run_dir / DEFAULT_JOURNAL_FILENAME

        reader = JournalReader(journal_path)
        reader.truncate_torn_tail()

        # Convert "max offset on disk" → "next offset to assign" so the
        # emitter's per-source counter is internally consistent.
        offset_resume = {
            src: max_off + 1
            for src, max_off in reader.newest_persisted_offset_per_source().items()
        }

        bus_kwargs: dict[str, int] = {}
        if capacity is not None:
            bus_kwargs["capacity"] = capacity
        bus = InMemoryBus(**bus_kwargs)
        dedup = EventDedup()
        dedup.reconstruct_from_journal(reader)
        writer = JournalWriter(journal_path)

        # Background TTL sweeper — daemon thread, started below so the
        # emitter is reachable to the sweeper for its first sweep. See
        # ``dedup_sweeper.py`` module docstring for the threading vs
        # asyncio choice.
        sweeper = DedupTTLSweeper(dedup)
        sweeper.start()

        return cls(
            run_id=run_id,
            source=source,
            journal=writer,
            bus=bus,
            dedup=dedup,
            offset_resume=offset_resume,
            dedup_sweeper=sweeper,
        )

    # ------------------------------------------------------------------
    # Read-only accessors (used by Phase 4-7 wiring)
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def source(self) -> str:
        return self._source

    @property
    def journal(self) -> JournalWriter:
        return self._journal

    @property
    def bus(self) -> InMemoryBus:
        return self._bus

    @property
    def dedup(self) -> EventDedup:
        return self._dedup

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # IEventEmitter surface
    # ------------------------------------------------------------------

    def emit(self, event: BaseEvent) -> None:
        """Append a locally-produced event to the journal + bus.

        See :class:`IEventEmitter` for the full contract. Summary:
        never raises; auto-fills offset and (when scoped)
        ``stage_id``; validates envelope as a production safety net.
        """
        if self._closed:
            self._bump_failed("closed")
            return

        try:
            # Step 1: fill stage_id from active scope when None.
            updates: dict[str, object] = {}
            if event.stage_id is None:
                scope_value = _current_stage_id.get()
                if scope_value is not None:
                    updates["stage_id"] = scope_value

            # Step 2: assign offset under the per-source lock if the
            # caller passed UNKNOWN_OFFSET. We compute the next value
            # using both the locally-tracked counter and the caller-
            # supplied offset (the latter is a safety net for tests
            # that pre-number).
            if event.offset == UNKNOWN_OFFSET or event.offset < 0:
                updates["offset"] = self._assign_offset(event.source)
            else:
                # Caller pre-numbered — bump the source counter past it
                # so a later auto-assign can't collide.
                self._bump_source_counter(event.source, event.offset + 1)

            if updates:
                event = event.model_copy(update=updates)

            # Step 3: validate the envelope. The discriminator union
            # raises pydantic.ValidationError if the kind/payload
            # combination is invalid — this is a production safety
            # net since stages can't be relied upon to produce
            # well-typed envelopes when generics are involved.
            try:
                EVENT_ADAPTER.validate_python(
                    event.model_dump(mode="python"),
                )
            except ValidationError as exc:
                self._bump_failed("validation")
                logger.warning(
                    "[ControlEventEmitter] emit validation failed "
                    "(event dropped): kind=%s err=%s",
                    event.kind, exc,
                )
                return

            # Step 4: durable write — the journal IS the source of truth.
            # If it fails we DROP the bus publish too, so a slow consumer
            # never sees an event that the journal didn't persist.
            try:
                self._journal.append(event)
            except Exception as exc:
                self._bump_failed("journal_write")
                logger.warning(
                    "[ControlEventEmitter] journal append failed "
                    "(event dropped): %s: %s",
                    type(exc).__name__, exc,
                )
                return

            # Step 5: best-effort bus publish. Errors here are logged
            # but the event is already persisted, so the run isn't lost.
            try:
                self._bus.publish(event)
            except Exception as exc:
                self._bump_failed("bus_publish")
                logger.warning(
                    "[ControlEventEmitter] bus publish failed "
                    "(event persisted, live consumers will miss it): %s: %s",
                    type(exc).__name__, exc,
                )
                return

            self.events_emitted_total += 1

        except Exception as exc:
            self._bump_failed("unexpected")
            logger.exception(
                "[ControlEventEmitter] unexpected emit failure swallowed: %s",
                exc,
            )

    def emit_remote(self, event: BaseEvent) -> None:
        """Forward a pre-populated remote envelope without rewriting fields.

        Behaviour:

        * Dedup keyed on ``(run_id, source, offset)``. Duplicates are
          silent drops with ``events_remote_dropped_total["duplicate"]``
          incremented.
        * Validates the envelope; invalid envelopes are dropped with
          ``events_remote_dropped_total["validation"]`` incremented.
        * NEVER overwrites ``event_id`` / ``offset`` / ``time`` /
          ``source``.
        * Journal write before bus publish (same SSOT rule as
          :meth:`emit`).
        """
        if self._closed:
            self._bump_remote_dropped("closed")
            return

        try:
            if event.offset == UNKNOWN_OFFSET or event.offset < 0:
                self._bump_remote_dropped("invalid_offset")
                logger.warning(
                    "[ControlEventEmitter] emit_remote: invalid offset "
                    "(remote envelope must be pre-numbered): kind=%s",
                    event.kind,
                )
                return

            if self._dedup.is_duplicate(event.run_id, event.source, event.offset):
                self._bump_remote_dropped("duplicate")
                return

            # Production safety net validation. Same translation as
            # emit() — invalid envelope is dropped not raised.
            try:
                EVENT_ADAPTER.validate_python(
                    event.model_dump(mode="python"),
                )
            except ValidationError as exc:
                self._bump_remote_dropped("validation")
                logger.warning(
                    "[ControlEventEmitter] emit_remote validation failed "
                    "(event dropped): kind=%s err=%s",
                    event.kind, exc,
                )
                return

            try:
                self._journal.append(event)
            except Exception as exc:
                self._bump_remote_dropped("journal_write")
                logger.warning(
                    "[ControlEventEmitter] emit_remote journal append failed: %s: %s",
                    type(exc).__name__, exc,
                )
                return

            # Mark as seen AFTER the journal succeeds. If we marked
            # first and the journal failed, a legitimate retry from
            # the producer would be dropped as a duplicate even though
            # we never persisted the original.
            self._dedup.remember(event.run_id, event.source, event.offset)

            # Bump the local offset counter past the observed offset so
            # a future locally-emitted event on the same source doesn't
            # collide.
            self._bump_source_counter(event.source, event.offset + 1)

            try:
                self._bus.publish(event)
            except Exception as exc:
                self._bump_remote_dropped("bus_publish")
                logger.warning(
                    "[ControlEventEmitter] emit_remote bus publish failed: %s: %s",
                    type(exc).__name__, exc,
                )
                return

            self.events_remote_accepted_total += 1

        except Exception as exc:
            self._bump_remote_dropped("unexpected")
            logger.exception(
                "[ControlEventEmitter] unexpected emit_remote failure swallowed: %s",
                exc,
            )

    @contextmanager
    def stage_scope(self, stage_id: str) -> Iterator[None]:
        """ContextVar push/pop for ``stage_id`` auto-fill."""
        token = _current_stage_id.set(stage_id)
        try:
            yield
        finally:
            _current_stage_id.reset(token)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush journal, close bus, mark emitter closed. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Stop the dedup sweeper FIRST so its next tick doesn't race
        # against a half-closed dedup set. The sweeper is daemon-thread
        # backed; ``stop`` is best-effort with a 2-second join.
        if self._dedup_sweeper is not None:
            with contextlib.suppress(Exception):
                self._dedup_sweeper.stop()
        with contextlib.suppress(Exception):
            self._journal.close()
        with contextlib.suppress(Exception):
            self._bus.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assign_offset(self, source: str) -> int:
        with self._offset_lock:
            # The counter stores the NEXT offset to assign. On resume
            # from journal we seeded it via
            # ``max_persisted_offset_per_source + 1`` semantics — but
            # ``newest_persisted_offset_per_source`` returns the max
            # offset already on disk. Normalise here so the seed dict
            # always represents "next offset to hand out".
            current = self._offset_counters.get(source)
            if current is None:
                next_offset = 0
            else:
                next_offset = current
            self._offset_counters[source] = next_offset + 1
            return next_offset

    def _bump_source_counter(self, source: str, candidate: int) -> None:
        with self._offset_lock:
            current = self._offset_counters.get(source, 0)
            if candidate > current:
                self._offset_counters[source] = candidate

    def _bump_failed(self, reason: str) -> None:
        self.events_emit_failed_total[reason] = (
            self.events_emit_failed_total.get(reason, 0) + 1
        )

    # Phase 8 alias — health endpoint / aggregator call sites prefer the
    # public ``_inc_emit_failed`` name so the increment surface is
    # discoverable. Implementation reuses the private helper; both names
    # are kept (call sites that already use ``_bump_failed`` don't need
    # to change).
    def _inc_emit_failed(self, reason: str) -> None:
        self._bump_failed(reason)

    def _bump_remote_dropped(self, reason: str) -> None:
        self.events_remote_dropped_total[reason] = (
            self.events_remote_dropped_total.get(reason, 0) + 1
        )

    def _inc_remote_dropped(self, reason: str) -> None:
        self._bump_remote_dropped(reason)

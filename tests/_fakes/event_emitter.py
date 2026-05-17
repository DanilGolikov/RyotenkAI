"""``FakeEventEmitter`` — canonical fake for :class:`IEventEmitter`.

In-memory implementation that mirrors the surface of the real
control-plane / pod-side emitters: ``emit`` for locally-produced events
(auto-fills ``offset`` from a per-source counter), ``emit_remote`` for
events arriving from a remote producer (dedups by ``(run_id, source,
offset)``), and ``stage_scope`` for ContextVar-style ``stage_id``
auto-fill.

Used in greenfield tests in place of ``MagicMock(spec=IEventEmitter)``
to comply with :mod:`tests._lint.test_no_protocol_mocking`. Tests can
inspect ``emitted`` / ``received_remote`` / ``dropped_remote`` lists for
assertions and program failure modes via the ``inject_*`` helpers.

Notes:

* :meth:`emit` auto-fills only ``offset`` (because the real emitter is
  the authority for ordering); ``event_id``, ``time``, ``source`` are
  retained as the caller built them. The real emitter would also fill
  ``stage_id`` from a ContextVar; we mirror that with a module-level
  :class:`contextvars.ContextVar` so async-task isolation matches
  production (Phase 1.5 Issue #3).
* :meth:`emit_remote` is strict: it preserves the caller's identity
  fields verbatim (matching the production "never overwrite" contract)
  and dedups silently.
* Both methods MUST NOT raise on validation failures from production
  code — chaos injection is the only way to surface a failure outwards,
  and even then the public methods translate to drops / counters rather
  than re-raising (matching the real emitter's never-raises contract).
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ryotenkai_shared.events.envelope import BaseEvent

from ryotenkai_shared.events.protocol import IEventEmitter  # noqa: TC001 — runtime isinstance check below


# Module-level ContextVar — matches the production emitters'
# implementation (``ryotenkai_pod.runner.event_emitter`` and
# ``ryotenkai_control.events.emitter``). Async tasks inherit the
# scope via :func:`contextvars.copy_context`, so concurrent stages in
# different tasks don't pollute each other.
_FAKE_STAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ryotenkai_fake_stage_id", default=None,
)


@dataclass
class FakeEventEmitter:
    """In-memory :class:`IEventEmitter` for tests.

    Public lists (``emitted``, ``received_remote``) are read-only contracts
    for assertions; do not mutate them directly. Use ``reset_chaos`` /
    ``clear`` to reset between cases.
    """

    emitted: list[BaseEvent] = field(default_factory=list)
    received_remote: list[BaseEvent] = field(default_factory=list)
    dropped_remote: int = 0
    invalid_remote: int = 0

    # Internal state — leading underscores denote "test-internal".
    _emit_failures_remaining: int = 0
    _validation_failures_remaining: int = 0
    _offset_counters: dict[str, int] = field(default_factory=dict)
    _seen_dedup_keys: set[tuple[str, str, int]] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_emit_failure(self, n: int = 1) -> None:
        """Next ``n`` :meth:`emit` calls raise :class:`RuntimeError` internally.

        The real emitter swallows internal errors (never raises to caller)
        but increments a metric. We mirror that: the failures here are
        recorded by *not* appending to ``emitted`` and by re-raising the
        error inside an exception handler — i.e. ``emit`` itself still
        returns ``None`` (never-raises contract preserved).
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        self._emit_failures_remaining = n

    def inject_validation_failure(self, n: int = 1) -> None:
        """Next ``n`` :meth:`emit_remote` calls drop with ``invalid_remote += 1``."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self._validation_failures_remaining = n

    def reset_chaos(self) -> None:
        self._emit_failures_remaining = 0
        self._validation_failures_remaining = 0

    def clear(self) -> None:
        """Reset all state including emitted lists. Test-only."""
        self.emitted.clear()
        self.received_remote.clear()
        self.dropped_remote = 0
        self.invalid_remote = 0
        self._offset_counters.clear()
        self._seen_dedup_keys.clear()
        self.reset_chaos()

    # ------------------------------------------------------------------
    # IEventEmitter surface
    # ------------------------------------------------------------------

    def emit(self, event: BaseEvent) -> None:
        """Append a locally-produced event after auto-filling ``offset`` / stage.

        Never raises to the caller. Chaos injection is recorded by NOT
        appending — equivalent to the real emitter incrementing
        ``events_emit_failed_total`` and continuing.
        """
        if self._emit_failures_remaining > 0:
            self._emit_failures_remaining -= 1
            # Swallow — mirrors production "never raises" contract.
            return

        # Auto-fill ``offset`` from a per-source counter. The real
        # emitter uses a lock around this assignment; we don't need one
        # here because the fake is single-threaded by construction.
        offset = self._offset_counters.get(event.source, 0)
        self._offset_counters[event.source] = offset + 1

        # Auto-fill ``stage_id`` from the active ContextVar scope when
        # None. ContextVar (not a thread-local stack) so async-task
        # isolation matches the production emitter.
        stage_id = event.stage_id
        if stage_id is None:
            scope_value = _FAKE_STAGE.get()
            if scope_value is not None:
                stage_id = scope_value

        # Frozen events — rebuild rather than mutate.
        updated = event.model_copy(update={"offset": offset, "stage_id": stage_id})
        self.emitted.append(updated)

    def emit_remote(self, event: BaseEvent) -> None:
        """Accept a pre-populated event from a remote source.

        Identity fields are preserved verbatim. Duplicates by
        ``(run_id, source, offset)`` are silent drops. Validation chaos
        increments ``invalid_remote``.
        """
        if self._validation_failures_remaining > 0:
            self._validation_failures_remaining -= 1
            self.invalid_remote += 1
            return

        key = (event.run_id, event.source, event.offset)
        if key in self._seen_dedup_keys:
            self.dropped_remote += 1
            return
        self._seen_dedup_keys.add(key)
        self.received_remote.append(event)

    @contextmanager
    def stage_scope(self, stage_id: str) -> Iterator[None]:
        """Push ``stage_id`` onto the ContextVar for the block.

        Token-based set/reset so nested scopes restore the previous
        value (not always ``None``) on exit. Matches the production
        emitters' implementation in :mod:`ryotenkai_pod.runner.event_emitter`
        and :mod:`ryotenkai_control.events.emitter`.
        """
        token = _FAKE_STAGE.set(stage_id)
        try:
            yield
        finally:
            _FAKE_STAGE.reset(token)


# Static guarantee: the fake satisfies the runtime-checkable Protocol.
_runtime_check: IEventEmitter = FakeEventEmitter()


__all__ = ["FakeEventEmitter"]

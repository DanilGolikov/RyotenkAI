"""Pipeline-facing :class:`IEventEmitter` Protocol for the unified event system.

The Protocol describes the *surface* that orchestrator stages, callbacks
and pod-side bridges use to publish envelopes. Concrete implementations
live in two places (per the Phase 1 plan):

* ``ryotenkai_pod/runner/event_emitter.py``       — PodEventEmitter
* ``ryotenkai_control/events/emitter.py``         — ControlEventEmitter

A canonical in-memory implementation for tests is
:class:`tests._fakes.event_emitter.FakeEventEmitter`. Tests MUST use the
canonical fake rather than mocking the Protocol — see
:mod:`tests._lint.test_no_protocol_mocking` for the enforced rule.

Contracts on the methods themselves are documented inline so that
implementers do not need to track three docs (the plan, this Protocol, and
the concrete class) — the Protocol docstring IS the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from ryotenkai_shared.events.envelope import BaseEvent


@runtime_checkable
class IEventEmitter(Protocol):
    """Append-to-journal-and-publish-to-bus interface for events.

    Two emission paths exist:

    * :meth:`emit`         — locally-produced events (control orchestrator
      stages, pod-side runner). The emitter auto-fills ``offset`` and
      (when wrapped in :meth:`stage_scope`) ``stage_id``. ``source`` is
      caller-supplied and never overwritten.
    * :meth:`emit_remote`  — events that arrive from a remote producer
      already populated (e.g. a pod WebSocket frame). The envelope's
      identity fields are NEVER overwritten — instead the emitter dedups
      by ``(run_id, source, offset)`` and silently drops duplicates.

    Both methods are synchronous and thread-safe; internal publish to a
    bus and persistence are best-effort and MUST NOT raise to the caller
    on transient failures. Failure paths are metrics-driven (see plan
    section "Risk R-06").
    """

    def emit(self, event: BaseEvent) -> None:
        """Append a locally-produced event to the journal and bus.

        Contracts:

        * Synchronous; returns once the event is queued for persistence
          and publish (which may complete asynchronously inside the
          implementation).
        * Thread-safe — multiple producers may emit concurrently.
        * NEVER raises to the caller. On internal error the implementation
          logs and increments ``events_emit_failed_total{reason=...}``.
        * Auto-fills ``offset`` from a monotonic counter keyed on
          ``(run_id, source)``.
        * If ``event.event_id`` looks unset (default-generated) the
          implementation is free to keep it; the contract is that the
          emitter never invents a *different* identity.
        * If ``stage_id`` is ``None`` and a ContextVar set via
          :meth:`stage_scope` is active, the emitter fills it.
        * Caller MUST supply ``source`` (e.g. ``"pod://{run_id}/trainer"``
          or ``"control://orchestrator/{stage}"``); the emitter never
          overwrites it. Producers that want a default source should
          centralise that decision at their construction site, not
          delegate it to the emitter — keeping the URI authoritative at
          the call site eliminates sentinel-value handling here.
        """
        ...

    def emit_remote(self, event: BaseEvent) -> None:
        """Receive a pre-populated event from a remote producer.

        Contracts:

        * Same threading & never-raises semantics as :meth:`emit`.
        * Identity fields (``event_id``, ``offset``, ``time``, ``source``)
          are NEVER overwritten — they are authoritative as received.
        * Dedups by ``(run_id, source, offset)``; duplicates are silent
          drops. Producers MAY safely resend during reconnect.
        * Validation failures (e.g. payload schema mismatch) increment
          ``events_remote_invalid_total`` and drop the event silently
          rather than crashing the consumer.
        """
        ...

    def stage_scope(self, stage_id: str) -> AbstractContextManager[None]:
        """ContextVar-style scope: events emitted inside auto-fill ``stage_id``.

        Implemented as a regular method returning a context manager (rather
        than the ``@contextmanager`` decorator pattern) so Protocols can
        express the contract cleanly. Nested scopes override the outer
        scope; on exit the previous ``stage_id`` is restored via the
        ContextVar token.
        """
        ...


__all__ = ["IEventEmitter"]

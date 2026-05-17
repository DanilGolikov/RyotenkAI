"""Pod-side :class:`IEventEmitter` implementation.

Wraps :class:`~ryotenkai_pod.runner.event_bus.EventBus` and the optional
:class:`~ryotenkai_pod.runner.event_journal.EventJournal` to satisfy the
:class:`~ryotenkai_shared.events.IEventEmitter` Protocol. Concrete
contract:

* :meth:`emit` accepts a fully-built :class:`BaseEvent` envelope. If
  ``offset == UNKNOWN_OFFSET`` the bus assigns a new monotonic offset
  under its per-source lock (R-05). The journal write is implicit in
  :meth:`EventBus.publish` (it persists the envelope before waking
  subscribers).
* :meth:`emit_remote` accepts an already-numbered envelope (e.g. from
  the trainer subprocess via loopback HTTP). The bus preserves the
  caller's ``event_id`` / ``offset`` / ``time`` / ``source`` — only
  used for bookkeeping that bumps the bus' global counter past the
  observed offset.
* :meth:`stage_scope` uses a :class:`contextvars.ContextVar`. Pod-side
  events rarely carry a ``stage_id`` (the runner doesn't know the
  control-side stage taxonomy), but Protocol compliance requires a
  no-op-friendly implementation. ContextVar (not a thread-local stack)
  so async tasks remain isolated when copy_context'd.

Thread-safety: the bus' per-source offset lock makes :meth:`emit`
callable from any thread. ContextVar reads / writes are also
thread-safe by construction.

Never-raises contract: the emitter swallows internal errors and logs at
WARN level. Re-raising would break the never-raises promise documented
on :class:`IEventEmitter`.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import TYPE_CHECKING

from ryotenkai_shared.events import BaseEvent
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus


__all__ = ["PodEventEmitter"]


logger = get_logger(__name__)


# ContextVar — async-safe, copy_context-friendly. Pod-side stages are
# almost always absent (Phase 2 wires the runner to control's stages via
# WS forward, not local scope), so the default is ``None``.
_current_stage_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ryotenkai_pod_current_stage_id", default=None,
)


class PodEventEmitter:
    """Concrete pod-side emitter — see module docstring for the contract."""

    def __init__(self, bus: EventBus, *, source: str) -> None:
        """
        Args:
            bus:    The pod-side event bus. Journal (if any) is attached
                    to the bus; the emitter doesn't need a direct
                    handle. Persistence happens implicitly on publish.
            source: Authoritative URI for events emitted via this
                    emitter (e.g. ``"pod://<run_id>/runner"``). Used
                    when callers don't override ``event.source`` —
                    typically they DO, since concrete event classes
                    require it.
        """
        self._bus = bus
        self._default_source = source

    @property
    def default_source(self) -> str:
        return self._default_source

    def emit(self, event: BaseEvent) -> None:
        """Append a locally-produced event to the journal and bus.

        Never raises. Auto-fills ``stage_id`` from the active ContextVar
        scope when the event omits it.
        """
        try:
            stage_id = event.stage_id
            if stage_id is None:
                scope_value = _current_stage_id.get()
                if scope_value is not None:
                    event = event.model_copy(update={"stage_id": scope_value})
            self._bus.publish(event)
        except Exception as exc:
            logger.warning(
                "[PodEventEmitter] emit failed (event swallowed): %s: %s",
                type(exc).__name__, exc,
            )

    def emit_remote(self, event: BaseEvent) -> None:
        """Forward a pre-populated remote envelope without rewriting fields.

        The bus' :meth:`publish` keeps the caller-supplied offset
        (Phase 2 lacks a SSOT dedup table; that lands in Phase 3 with
        the control-side ``dedup.py``). We still bump the bus' global
        counter past the observed offset so future locally-emitted
        events don't collide.
        """
        try:
            # NOTE: caller-supplied offset is preserved by EventBus.publish.
            self._bus.publish(event)
        except Exception as exc:
            logger.warning(
                "[PodEventEmitter] emit_remote failed (event swallowed): %s: %s",
                type(exc).__name__, exc,
            )

    @contextlib.contextmanager
    def stage_scope(self, stage_id: str) -> Iterator[None]:
        """Push ``stage_id`` onto the ContextVar for the duration of the block.

        Nested scopes override the outer scope; on exit the previous
        value is restored via the ContextVar token. Async tasks
        launched inside the scope inherit the scope via ``copy_context``
        (the default ``asyncio.create_task`` propagation).
        """
        token = _current_stage_id.set(stage_id)
        try:
            yield
        finally:
            _current_stage_id.reset(token)

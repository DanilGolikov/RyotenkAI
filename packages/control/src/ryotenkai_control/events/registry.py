"""Process-wide registry of active :class:`ControlEventEmitter` instances (Phase 6.a).

The API events router (``/api/runs/{run_id}/events/stream``) needs the
in-memory bus for the live SSE tail. The bus is owned by the per-run
emitter the orchestrator constructs in
:meth:`PipelineOrchestrator._ensure_event_emitter`. Without a registry
the router would have to crawl the orchestrator's state or hold a
reference to every running run — both are awkward across request
boundaries.

The registry is a thin :class:`dict[str, ControlEventEmitter]` guarded by
a :class:`threading.Lock` so :meth:`register` / :meth:`deregister` /
:meth:`get` are safe from any thread (uvicorn worker, orchestrator main,
asyncio task). It does NOT own the emitter — registration is a weak
pointer in spirit: the orchestrator is responsible for matching
``register`` and ``deregister`` in a ``try/finally`` so a crashed run
doesn't leak the slot.

Usage:

.. code-block:: python

   registry = EventEmitterRegistry.instance()
   registry.register(run_id, emitter)
   try:
       ...
   finally:
       registry.deregister(run_id)
       emitter.close()

Tests reset the singleton via :meth:`reset_instance` so a fresh process
identity is observable between cases. Production never calls that —
each process has exactly one registry.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryotenkai_control.events.emitter import ControlEventEmitter


__all__ = ["EventEmitterRegistry"]


class EventEmitterRegistry:
    """Process-wide ``run_id -> ControlEventEmitter`` registry.

    Thread-safe through a single internal :class:`threading.Lock`. Each
    method snapshots the underlying mapping under the lock so callers
    never observe a torn state.

    The class exposes a :meth:`instance` singleton accessor; tests may
    call :meth:`reset_instance` to obtain a fresh registry between
    cases. The singleton is module-scoped — the first call creates it,
    subsequent calls return the same object.
    """

    _instance_lock: threading.Lock = threading.Lock()
    _singleton: EventEmitterRegistry | None = None

    def __init__(self) -> None:
        self._emitters: dict[str, ControlEventEmitter] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> EventEmitterRegistry:
        """Return the process-wide singleton, lazily constructing on first call."""
        with cls._instance_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton — tests only.

        After this call the next :meth:`instance` returns a fresh
        registry. Existing references to the previous instance remain
        usable but are no longer reachable via :meth:`instance` — kept
        intentionally so an in-flight request that captured the old
        reference still observes the same state until the request
        completes.
        """
        with cls._instance_lock:
            cls._singleton = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, run_id: str, emitter: ControlEventEmitter) -> None:
        """Insert ``emitter`` under ``run_id``.

        Idempotent on the (run_id, emitter) identity: re-registering the
        same emitter is a no-op. Registering a DIFFERENT emitter under
        an existing run_id replaces the previous entry (the orchestrator
        is the single writer for a given run, so this normally means a
        retry inside the same process; warn would be misleading).
        """
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        with self._lock:
            self._emitters[run_id] = emitter

    def deregister(self, run_id: str) -> None:
        """Remove the emitter mapped to ``run_id``. Idempotent if absent."""
        with self._lock:
            self._emitters.pop(run_id, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> ControlEventEmitter | None:
        """Return the emitter mapped to ``run_id`` or ``None`` if absent."""
        with self._lock:
            return self._emitters.get(run_id)

    def active_run_ids(self) -> list[str]:
        """Snapshot of currently-registered run ids (defensive copy)."""
        with self._lock:
            return list(self._emitters.keys())

    def __contains__(self, run_id: object) -> bool:
        if not isinstance(run_id, str):
            return False
        with self._lock:
            return run_id in self._emitters

    def __len__(self) -> int:
        with self._lock:
            return len(self._emitters)

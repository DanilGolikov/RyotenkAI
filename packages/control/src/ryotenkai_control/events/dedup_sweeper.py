"""Background TTL sweeper for :class:`EventDedup` (post-Phase-10 TODO #2).

The dedup set inside a long-lived :class:`ControlEventEmitter` only loses
entries on explicit calls to :meth:`EventDedup.evict_expired`. Production
has no such callsite — terminal events evict per-run keys, but nothing
sweeps TTL-expired entries left behind by long resume windows.

RyotenkAI runs are designed so that the operator's Mac can sleep for
hours / days while a pod trains on the provider side; on wake-up the
emitter is still the same process instance. Over the lifetime of that
process the dedup set grows unbounded unless something periodically
calls :meth:`evict_expired`.

This module provides a tiny daemon-thread sweeper. ``threading.Thread``
was chosen over ``asyncio.create_task`` because:

* :meth:`ControlEventEmitter.close` is synchronous (called from
  orchestrator finalisation, no running event loop guaranteed).
* The orchestrator process itself is largely synchronous; there is no
  long-lived event loop to attach an asyncio task to.
* The sweep call is cheap and infrequent (every 5 min by default), so
  GIL contention with the asyncio-flavoured API server is negligible.
* A daemon thread dies with the process — no shutdown coordination
  required for the common "operator Ctrl-C" path.

The sweeper is idempotent on :meth:`start` and :meth:`stop` so the
:class:`ControlEventEmitter` lifecycle stays robust against double-close
and reused-emitter scenarios.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_control.events.dedup import EventDedup


__all__ = ["DEFAULT_SWEEP_INTERVAL_SECONDS", "DedupTTLSweeper"]


logger = get_logger(__name__)


#: Default cadence of the background sweep. The TTL itself defaults to
#: 24 h (see :data:`ryotenkai_control.events.dedup.DEFAULT_TTL_SECONDS`)
#: so a 5-minute sweep is two orders of magnitude faster than the
#: eviction window — newly-aged entries vanish well within one minute
#: of their TTL.
DEFAULT_SWEEP_INTERVAL_SECONDS = 300.0


class DedupTTLSweeper:
    """Background thread that periodically calls ``dedup.evict_expired()``.

    Lifecycle:

    1. ``DedupTTLSweeper(dedup).start()`` — spawn a daemon thread.
    2. The thread waits for ``sweep_interval_seconds`` (or until
       ``stop`` is signalled), then calls ``evict_expired`` exactly
       once, logs the count, and repeats.
    3. ``DedupTTLSweeper.stop()`` — set the stop event and join with a
       short timeout. Idempotent; safe to call before ``start`` or
       multiple times.

    The thread is a *daemon* so a forgotten ``stop()`` does not block
    process exit — the worst-case outcome is the process exits with a
    sweeper still inside its ``wait()``.
    """

    def __init__(
        self,
        dedup: EventDedup,
        *,
        sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
        thread_name: str = "ryotenkai-dedup-ttl-sweeper",
    ) -> None:
        if sweep_interval_seconds <= 0:
            raise ValueError("sweep_interval_seconds must be > 0")
        self._dedup = dedup
        self._interval = sweep_interval_seconds
        self._thread_name = thread_name
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Lock guards transitions of ``_thread`` so ``start`` from two
        # callers cannot both spawn a thread.
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while a sweeper thread is alive (post-start, pre-stop)."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Spawn the background sweeper. Idempotent — no-op if already running."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Reset stop event so a stop-then-restart cycle works
            # without losing the wakeup signal from the previous run.
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Signal the sweeper to exit and wait briefly. Idempotent."""
        with self._lifecycle_lock:
            thread = self._thread
        # Signal regardless of whether a thread exists — keeps the
        # implementation tolerant of stop-before-start callers.
        self._stop_event.set()
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=timeout)
        # We do NOT clear ``self._thread`` if the join timed out;
        # the daemon flag means the OS will reap it at process exit.

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Sweep loop body. Runs on the background daemon thread."""
        # NOTE: a previous iteration of this design read ``self._dedup``
        # inside the loop so a test could swap the reference at runtime.
        # That violates the "the sweeper owns the reference passed to
        # __init__" invariant — capture once and re-use.
        dedup = self._dedup
        while not self._stop_event.is_set():
            # Block until interval elapses OR stop is signalled — the
            # ``wait()`` returns ``True`` when the event was set, which
            # we use to short-circuit the upcoming sweep on shutdown.
            stopped = self._stop_event.wait(timeout=self._interval)
            if stopped:
                break
            try:
                evicted = dedup.evict_expired()
                if evicted > 0:
                    logger.info(
                        "[dedup_ttl_sweeper] evicted %d expired entries",
                        evicted,
                    )
            except Exception:
                # The sweeper is best-effort; a bad sweep must not kill
                # the daemon. Log and continue — next tick tries again.
                logger.exception(
                    "[dedup_ttl_sweeper] sweep failed; continuing",
                )

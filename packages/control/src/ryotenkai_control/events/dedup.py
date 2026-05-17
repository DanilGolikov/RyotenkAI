"""(run_id, source, offset) dedup table for :meth:`emit_remote` (R-13).

Producers that resend during reconnect (the pod's WS bridge in Phase 5)
must not double-emit envelopes. The dedup set is the SSOT for "have I
seen this remote envelope before"; on restart it is rebuilt from the
journal so duplicates survive the gap.

Per-entry TTL eviction prevents the set from growing without bound when
the same orchestrator instance handles many runs. The default TTL of
24 hours covers any reasonable resume window — older entries are evicted
en masse on the next :meth:`evict_expired` sweep.

The dedup set is thread-safe — :meth:`is_duplicate` / :meth:`remember`
may be called from any thread that the emitter is reachable on.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ryotenkai_control.events.journal_reader import JournalReader


__all__ = ["DEFAULT_TTL_SECONDS", "EventDedup"]


DEFAULT_TTL_SECONDS = 24 * 3600  # 24 hours

# Tuple shape on the wire — (run_id, source, offset).
DedupKey = tuple[str, str, int]


class EventDedup:
    """Bounded set of ``(run_id, source, offset)`` keys with TTL eviction."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        self._ttl = ttl_seconds
        # Injectable clock — tests use ``MockClock``; production uses
        # ``time.monotonic``. The clock is read on remember/evict only,
        # never on the hot ``is_duplicate`` path.
        self._clock: Callable[[], float] = clock or time.monotonic
        self._lock = threading.Lock()
        # Map key → wall-clock timestamp of last ``remember``.
        self._entries: dict[DedupKey, float] = {}

        # Phase 8 — observability counters scraped by the health
        # endpoint. They live alongside ``size`` so callers can answer
        # "how busy is dedup right now" + "how busy has it ever been".
        # ``_seen_total`` ticks once per :meth:`remember` (NOT once per
        # is_duplicate hit) so an explicit ``remember(...)`` for the
        # same key bumps the counter again — same key, fresh
        # remembering.
        self._seen_total = 0
        # Incremented every time :meth:`is_duplicate` returns ``True``.
        # Useful for spotting a noisy producer that re-sends often.
        self._dedup_hits_total = 0
        # Cumulative count of TTL / per-run / journal-reconstruction
        # evictions. The aggregator reads this to distinguish "dedup
        # naturally aged out" from "dedup grew unbounded".
        self._evicted_total = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def seen_total(self) -> int:
        """Total :meth:`remember` calls observed (lifetime count)."""
        return self._seen_total

    @property
    def dedup_hits_total(self) -> int:
        """Total :meth:`is_duplicate` calls that returned ``True``."""
        return self._dedup_hits_total

    @property
    def evicted_total(self) -> int:
        """Cumulative entries removed via evict_run / evict_expired / reconstruct."""
        return self._evicted_total

    def is_duplicate(self, run_id: str, source: str, offset: int) -> bool:
        """Return ``True`` if the key has been remembered.

        Does NOT count as "remembering" — the caller should follow with
        :meth:`remember` after the validation/journal path completes.
        """
        key = (run_id, source, offset)
        with self._lock:
            hit = key in self._entries
            if hit:
                self._dedup_hits_total += 1
            return hit

    def remember(self, run_id: str, source: str, offset: int) -> None:
        """Record the key with the current wall-clock time."""
        key = (run_id, source, offset)
        with self._lock:
            self._entries[key] = self._clock()
            self._seen_total += 1

    def evict_run(self, run_id: str) -> int:
        """Drop every entry whose first tuple element matches ``run_id``.

        Called on terminal events (RunCompleted/Failed/Cancelled) so the
        set doesn't carry dead-run keys forever. Returns the number of
        entries evicted (useful for tests / metrics).
        """
        with self._lock:
            victims = [k for k in self._entries if k[0] == run_id]
            for k in victims:
                del self._entries[k]
            self._evicted_total += len(victims)
            return len(victims)

    def evict_expired(self) -> int:
        """Drop every entry whose age exceeds ``ttl_seconds``.

        Returns the number of entries evicted.
        """
        cutoff = self._clock() - self._ttl
        with self._lock:
            victims = [k for k, ts in self._entries.items() if ts <= cutoff]
            for k in victims:
                del self._entries[k]
            self._evicted_total += len(victims)
            return len(victims)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def reconstruct_from_journal(
        self,
        reader: JournalReader,
        *,
        max_entries_per_source: int = 10_000,
    ) -> int:
        """Replay the last ``max_entries_per_source`` envelopes into the set.

        On restart this is the only thing standing between a reconnecting
        pod and re-emitting a duplicate (R-13). Reads via
        :meth:`JournalReader.tail_per_source` to keep memory bounded
        regardless of journal size.

        Returns the number of entries inserted.
        """
        added = 0
        now = self._clock()
        bucketed = reader.tail_per_source(max_entries_per_source=max_entries_per_source)
        with self._lock:
            for source, envelopes in bucketed.items():
                for envelope in envelopes:
                    key = (envelope.run_id, source, envelope.offset)
                    self._entries[key] = now
                    added += 1
            # Reconstruction repopulates the set from durable storage —
            # bump the lifetime counter so post-restart the snapshot
            # doesn't suggest "zero remote events have ever been seen".
            self._seen_total += added
        return added

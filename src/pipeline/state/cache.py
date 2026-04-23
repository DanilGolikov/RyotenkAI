"""Process-local mtime-keyed cache for ``PipelineState``.

Sibling clients (web API, TUI) poll pipeline state frequently — every few
seconds per open client, per run. Each poll re-reads ``pipeline_state.json``,
parses JSON and re-builds dataclasses. For read-heavy workloads this is
wasteful: the file only changes when the pipeline writes a new state (stage
transitions, attempt boundaries).

This module memoizes the parsed ``PipelineState`` keyed on
``(state_path, mtime_ns)``. A ``stat()`` is cheap (inode lookup, single
syscall); re-parsing a multi-kB JSON doc and re-building frozen dataclasses
is not. On cache hit we skip the parse entirely.

Contract
--------
* The cache is **read-side only**. Writers (``PipelineOrchestrator``,
  ``AttemptController``) go through ``PipelineStateStore.save`` unchanged —
  they don't need the cache, and using it would risk returning a stale copy
  immediately after they wrote.
* Cache keys are ``(resolved state_path, mtime_ns)``. POSIX ``rename``
  (``atomic_write_json``) bumps ``mtime_ns`` on every save, so the next
  ``stat()`` invalidates naturally. There is no manual invalidation path
  except ``clear()`` for tests.
* Thread-safe via a single ``RLock``. The cached ``PipelineState`` objects
  are treated as immutable value objects by all sibling-client readers; the
  Orchestrator mutates its own in-memory copy (never a cached one).
* Bounded LRU (default 256 entries). Enough for "watch every run on the
  machine" scenarios; writers are cheap so eviction is acceptable.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.pipeline.state.store import PipelineStateLoadError, PipelineStateStore

if TYPE_CHECKING:
    from src.pipeline.state.models import PipelineState


DEFAULT_MAX_ENTRIES = 256


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable view of ``PipelineState`` paired with the source ``mtime_ns``.

    ``mtime_ns`` is the nanosecond-resolution mtime of ``pipeline_state.json``
    at the moment it was read. Callers use it both to detect "has the state
    changed?" and as the payload for HTTP caching headers (ETag /
    Last-Modified).
    """

    state: PipelineState
    mtime_ns: int


class _StateCache:
    """Bounded LRU keyed on ``(state_path, mtime_ns)``.

    Not exported — access through the module-level ``load_state_snapshot``
    / ``clear_cache`` helpers. Kept as a class (rather than bare dict) so
    tests can instantiate an isolated copy without touching the singleton.
    """

    __slots__ = ("_entries", "_hits", "_lock", "_max_entries", "_misses")

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[Path, tuple[int, PipelineState]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def load(self, run_dir: Path) -> StateSnapshot:
        """Return a snapshot, using the cache when the file's mtime matches."""
        store = PipelineStateStore(run_dir)
        state_path = store.state_path

        # Stat first — cheap and lets us short-circuit a stale cache entry
        # without holding the lock across the JSON parse.
        try:
            mtime_ns = state_path.stat().st_mtime_ns
        except FileNotFoundError as exc:
            raise PipelineStateLoadError(f"Missing pipeline state: {state_path}") from exc

        with self._lock:
            cached = self._entries.get(state_path)
            if cached is not None and cached[0] == mtime_ns:
                # LRU bump — move to most-recently-used end.
                self._entries.move_to_end(state_path)
                self._hits += 1
                return StateSnapshot(state=cached[1], mtime_ns=mtime_ns)

        # Cache miss — load outside the lock so concurrent readers of *other*
        # runs don't block on our disk I/O.
        state = store.load()

        with self._lock:
            # Another thread may have loaded the same entry meanwhile; that's
            # fine — last write wins and the object is effectively immutable.
            self._entries[state_path] = (mtime_ns, state)
            self._entries.move_to_end(state_path)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            self._misses += 1

        return StateSnapshot(state=state, mtime_ns=mtime_ns)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._entries),
                "max_entries": self._max_entries,
            }


_SINGLETON = _StateCache()


def load_state_snapshot(run_dir: Path) -> StateSnapshot:
    """Return a ``StateSnapshot`` for ``run_dir`` via the process-local cache.

    Prefer this over ``PipelineStateStore(run_dir).load()`` on the API / TUI
    read path. Writers must continue using the store directly.
    """
    return _SINGLETON.load(Path(run_dir).expanduser().resolve())


def clear_cache() -> None:
    """Drop every cached entry. Used by tests and admin tooling."""
    _SINGLETON.clear()


def cache_stats() -> dict[str, int]:
    """Observability: hit / miss / entry counts for the singleton."""
    return _SINGLETON.stats()


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "StateSnapshot",
    "cache_stats",
    "clear_cache",
    "load_state_snapshot",
]

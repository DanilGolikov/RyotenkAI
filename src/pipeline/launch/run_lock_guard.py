"""Context-manager guard around pipeline run locks.

Enforces Invariant #1 of the pipeline architecture: every acquired ``run.lock``
MUST be released on every exit path (success, exception, signal). The old
implementation used a plain ``try/finally`` with a manual ``release()`` call —
fragile, and a bug in any of the finally-steps could skip it.

Usage::

    with RunLockGuard(state_store.lock_path) as guard:
        ...                               # acquired; ``guard.lock`` available
    # guaranteed released here, including on exception

The guard swallows exceptions raised by ``release()`` in ``__exit__`` (and
logs them) so a release failure cannot mask the real exception that triggered
the exit. If the lock is already held by another process, ``__enter__`` raises
``PipelineStateLockError`` from the underlying ``acquire_run_lock``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.pipeline.state.store import PipelineRunLock, acquire_run_lock
from src.utils.logger import logger

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class RunLockGuard:
    """Context manager that guarantees ``run.lock`` is released on every exit path."""

    __slots__ = ("_lock", "_lock_path")

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._lock: PipelineRunLock | None = None

    @property
    def lock(self) -> PipelineRunLock | None:
        """The underlying ``PipelineRunLock``. ``None`` before __enter__ / after __exit__."""
        return self._lock

    @property
    def is_held(self) -> bool:
        """True while the guard owns an active lock."""
        return self._lock is not None

    def __enter__(self) -> RunLockGuard:
        # Acquire errors (e.g. already-locked) propagate; they are a legitimate
        # pre-run failure that must surface to the orchestrator.
        self._lock = acquire_run_lock(self._lock_path)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        lock = self._lock
        self._lock = None
        if lock is None:
            return
        try:
            lock.release()
        except Exception:
            # Never mask the original exception; log and swallow. ``release`` itself
            # already uses contextlib.suppress internally, but defence in depth.
            logger.exception("[RUN_LOCK_GUARD] release() failed; lock file may leak")


__all__ = ["RunLockGuard"]

"""Single owner of MLflow run finalization (atexit + signal driven).

The legacy code finalized MLflow runs from four independent places
(orchestrator ``finally``, atexit registration in ``cli/_signals``,
SIGTERM handler, and the reconciliation thread). Race conditions on
status / tag writes were observed in long-running pipelines that
combined a hard SIGTERM with a non-empty journal buffer.

:class:`RunLifecycleCoord` converges all four onto a single mutex'd
entry point:

* Used as a context manager. ``__enter__`` registers an
  :mod:`atexit` hook *and* installs signal handlers for SIGTERM /
  SIGINT (storing the previous handlers so ``__exit__`` can restore
  them). ``__exit__`` is itself idempotent so nested usage degrades
  gracefully.
* :meth:`bind_root_run` and :meth:`bind_attempt_run` register the
  handles the coord should close. They are thread-safe so the
  orchestrator can call them from the main thread while a signal
  handler is dispatched on another.
* :meth:`finalize` is the single public close path. The attempt run
  closes BEFORE the root run; both go through
  :class:`MlflowFinalizer`. A re-entrant call returns immediately
  via the :data:`_finalized` flag.

Never raises
------------
Every step delegates to :class:`MlflowFinalizer.finalize`, which is
defined to swallow exceptions; the coord itself also wraps the whole
``_safe_finalize`` body in ``try/except Exception``. This is required
because the call site is sometimes a signal handler -- exceptions
escaping a handler would corrupt the Python interpreter state.
"""

from __future__ import annotations

import atexit
import signal
import threading
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ryotenkai_control.pipeline.mlflow.lifecycle.finalizer import MlflowFinalizer
    from ryotenkai_control.pipeline.mlflow.lifecycle.opener import ParentRunOpener
    from ryotenkai_control.pipeline.mlflow.lifecycle.preflight import (
        PreflightConnectivityCheck,
    )
    from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle


logger = get_logger(__name__)


# Signals we install handlers for. SIGTERM is the canonical "graceful
# shutdown" signal sent by ``systemd``/``kubectl``; SIGINT is the
# Ctrl+C the developer sends during local debugging.
_TRAPPED_SIGNALS: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)


__all__ = ["RunLifecycleCoord"]


class RunLifecycleCoord:
    """Mutex'd, atexit/signal-bound owner of the finalization path.

    :param opener: :class:`ParentRunOpener` (currently held for
        callers that want to read it back -- the coord itself does not
        call ``opener.open`` directly; the orchestrator owns the
        open-side ordering).
    :param finalizer: :class:`MlflowFinalizer` invoked by
        :meth:`finalize` for both the attempt and root runs.
    :param preflight: :class:`PreflightConnectivityCheck` held for
        composition so callers can pull all three lifecycle helpers
        from one wiring point. Not invoked by the coord itself.
    """

    def __init__(
        self,
        opener: ParentRunOpener,
        finalizer: MlflowFinalizer,
        preflight: PreflightConnectivityCheck,
    ) -> None:
        self._opener = opener
        self._finalizer = finalizer
        self._preflight = preflight

        # Locked state.
        self._lock = threading.Lock()
        self._root_run: RunHandle | None = None
        self._attempt_run: RunHandle | None = None
        self._finalized: bool = False

        # Last finalize parameters passed through bind_finalize_payload.
        self._pending_journal_path: Path | None = None
        self._pending_journal_sha256: str | None = None

        # State for context-manager lifecycle.
        self._entered: bool = False
        self._prev_signal_handlers: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __enter__(self) -> RunLifecycleCoord:
        """Register atexit + signal handlers."""
        if self._entered:
            return self
        atexit.register(self._safe_finalize_atexit)
        for sig in _TRAPPED_SIGNALS:
            try:
                prev = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                # Setting a signal handler can fail when called outside
                # the main thread; degrade silently -- atexit still works.
                logger.debug(
                    "[MLFLOW_COORD] failed to install handler for signal=%s "
                    "(non-main-thread context); skipping",
                    sig,
                )
                continue
            self._prev_signal_handlers[sig] = prev
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Unregister handlers; idempotent."""
        if not self._entered:
            return
        try:
            atexit.unregister(self._safe_finalize_atexit)
        except Exception as e:  # noqa: BLE001
            logger.debug("[MLFLOW_COORD] atexit.unregister failed: %s", e)
        for sig, prev in self._prev_signal_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError) as e:
                logger.debug(
                    "[MLFLOW_COORD] signal restore failed sig=%s: %s",
                    sig,
                    e,
                )
        self._prev_signal_handlers.clear()
        self._entered = False

    def bind_root_run(self, run: RunHandle) -> None:
        """Register the root :class:`RunHandle` for close."""
        with self._lock:
            self._root_run = run

    def bind_attempt_run(self, run: RunHandle) -> None:
        """Register the attempt :class:`RunHandle` for close."""
        with self._lock:
            self._attempt_run = run

    def bind_finalize_payload(
        self,
        *,
        journal_path: Path | None,
        journal_sha256: str | None,
    ) -> None:
        """Bind the journal upload payload for atexit/signal-driven close.

        Stores parameters that the atexit / signal handlers will pass
        through to :meth:`MlflowFinalizer.finalize`. Explicit
        :meth:`finalize` callers may pass them inline and skip this.
        """
        with self._lock:
            self._pending_journal_path = journal_path
            self._pending_journal_sha256 = journal_sha256

    def finalize(
        self,
        *,
        status: RunStatus,
        journal_path: Path | None = None,
        journal_sha256: str | None = None,
        exit_reason: str | None = None,
    ) -> None:
        """Close the attempt run then the root run, idempotently.

        :param status: Terminal :class:`RunStatus` for both runs.
        :param journal_path: Local path to the SSOT journal file
            (forwarded to :meth:`MlflowFinalizer.finalize`).
        :param journal_sha256: Hex digest of the journal contents.
        :param exit_reason: Free-form reason for the close.
        """
        with self._lock:
            if self._finalized:
                logger.debug(
                    "[MLFLOW_COORD] already finalized; ignoring re-entrant call"
                )
                return
            self._finalized = True
            attempt = self._attempt_run
            root = self._root_run

        # Close attempt run first so the root's terminal status reflects
        # the post-attempt state visible to downstream readers.
        if attempt is not None:
            try:
                self._finalizer.finalize(
                    run=attempt,
                    status=status,
                    journal_path=journal_path,
                    journal_sha256=journal_sha256,
                    exit_reason=exit_reason,
                )
            except Exception as exc:  # noqa: BLE001 -- never raise
                logger.warning(
                    "[MLFLOW_COORD] attempt finalize raised (suppressed): %s",
                    exc,
                )
        if root is not None:
            try:
                self._finalizer.finalize(
                    run=root,
                    status=status,
                    # Journal is uploaded to the attempt run; do NOT
                    # re-upload to the root.
                    journal_path=None,
                    journal_sha256=None,
                    exit_reason=exit_reason,
                )
            except Exception as exc:  # noqa: BLE001 -- never raise
                logger.warning(
                    "[MLFLOW_COORD] root finalize raised (suppressed): %s",
                    exc,
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _safe_finalize_atexit(self) -> None:
        """atexit hook -- closes runs with :class:`RunStatus.FINISHED`.

        We assume that an interpreter shutdown reached atexit through
        the normal exit path; the orchestrator is expected to call
        :meth:`finalize` explicitly with the *correct* status before
        process exit, in which case this hook is a no-op.
        """
        try:
            self.finalize(
                status=RunStatus.FINISHED,
                journal_path=self._pending_journal_path,
                journal_sha256=self._pending_journal_sha256,
                exit_reason="atexit",
            )
        except Exception as exc:  # noqa: BLE001 -- never raise from atexit
            logger.warning("[MLFLOW_COORD] atexit finalize swallowed: %s", exc)

    def _handle_signal(
        self,
        signum: int,
        _frame: FrameType | None,
    ) -> None:
        """SIGTERM/SIGINT handler.

        Marks runs as :class:`RunStatus.KILLED`, then re-raises
        :class:`KeyboardInterrupt` so the orchestrator's existing
        ``except KeyboardInterrupt`` branch fires and the shutdown
        traceback shows the signal source.
        """
        try:
            self.finalize(
                status=RunStatus.KILLED,
                journal_path=self._pending_journal_path,
                journal_sha256=self._pending_journal_sha256,
                exit_reason=f"signal:{signum}",
            )
        except Exception as exc:  # noqa: BLE001 -- never raise from handler
            logger.warning(
                "[MLFLOW_COORD] signal finalize swallowed sig=%s: %s",
                signum,
                exc,
            )
        # Re-raise so the shutdown traceback shows the originating signal
        # and the orchestrator's KeyboardInterrupt branch fires.
        raise KeyboardInterrupt(f"signal:{signum}")

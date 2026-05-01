"""Process-wide cooperative cancellation for the pipeline worker subprocess.

The CLI parent installs ``SIG_IGN`` for SIGINT before spawning the worker
(see ``src/cli/commands/run.py``). The kernel still routes terminal-
initiated SIGINT to the whole process group, so the worker — sharing the
parent's PG — receives it directly. This module's :func:`install_handler`
hooks SIGINT/SIGTERM in the worker's main thread; the handler sets a
process-wide :class:`threading.Event` and lets pollers cooperatively
raise :class:`PipelineCancelled` instead of letting Python's default
handler raise ``KeyboardInterrupt`` from arbitrary blocking calls.

Design notes:

* Module-level globals (singleton state) — there is exactly one cancel
  event per worker process. ``contextvars`` would be overkill.
* :class:`PipelineCancelled` subclasses ``BaseException`` (matching
  stdlib ``asyncio.CancelledError`` since Python 3.8) so library code
  using generic ``except Exception:`` cannot accidentally swallow it.
  Provider cleanup hooks catch it explicitly.
* The signal handler does the absolute minimum: increment a counter,
  set the Event, optionally notify the orchestrator, arm a hard-exit
  deadline. No locks, no I/O, no logging. ``Event.set()`` is the only
  primitive officially documented as safe in a signal handler.
* Double Ctrl+C → ``os._exit(130)``. Standard kubectl/docker escalation
  pattern: first signal asks for graceful shutdown, second forces it.
* 30-second deadline timer armed on the first signal — matches k8s
  default ``terminationGracePeriodSeconds``. If cleanup wedges past
  the deadline, ``os._exit(130)`` guarantees process exit.

The CLI parent process keeps using :mod:`src.cli._signals` — that's a
separate handler for the parent's lifecycle and is intentionally not
unified with this module (different concerns, different lifetimes).
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from typing import Any


class PipelineCancelled(BaseException):
    """Raised by cooperative pollers when the cancel event is set.

    Subclasses :class:`BaseException`, NOT :class:`Exception` — matches
    the stdlib :class:`asyncio.CancelledError` model so generic
    ``except Exception:`` blocks cannot swallow cancellation.

    Provider cleanup hooks catch it explicitly with
    ``except PipelineCancelled:``, perform synchronous teardown of any
    in-flight pod / connection, and re-raise so the orchestrator's
    cleanup-in-reverse can proceed.
    """


_event: threading.Event = threading.Event()
_signal_count: int = 0
_active_orchestrator: object | None = None
_deadline_timer: threading.Timer | None = None
_DEADLINE_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# Public API — for pollers
# ---------------------------------------------------------------------------


def is_cancelled() -> bool:
    """Non-blocking probe. ``True`` once SIGINT/SIGTERM has been received."""
    return _event.is_set()


def check_cancelled() -> None:
    """Raise :class:`PipelineCancelled` if a cancel signal was received.

    Call from inside long loops where you don't want to block on
    :func:`sleep_cancellable` — e.g. between expensive synchronous
    operations.
    """
    if _event.is_set():
        raise PipelineCancelled()


def sleep_cancellable(seconds: float) -> None:
    """Block for at most ``seconds``, returning early if cancelled.

    Use this in place of ``time.sleep`` inside polling loops. If the
    cancel event is set during the wait — or was already set on entry —
    raises :class:`PipelineCancelled` immediately.
    """
    if _event.wait(timeout=seconds):
        raise PipelineCancelled()


# ---------------------------------------------------------------------------
# Public API — for the worker bootstrap
# ---------------------------------------------------------------------------


def set_active_orchestrator(orchestrator: object | None) -> None:
    """Install / clear the orchestrator reference the handler will notify.

    Pass ``None`` from a ``finally`` block to clear — leaving a stale
    reference would notify the wrong object on the next signal in tests
    that re-use the module.
    """
    global _active_orchestrator
    _active_orchestrator = orchestrator


def get_active_orchestrator() -> object | None:
    """Return the orchestrator currently registered with the handler."""
    return _active_orchestrator


def install_handler() -> None:
    """Register the SIGINT / SIGTERM handler. Idempotent.

    Must be called from the main thread of the main interpreter — the
    only place ``signal.signal`` is allowed by the runtime. The worker's
    ``main()`` calls this as its first non-import action.
    """
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------


def reset_for_tests() -> None:
    """Clear cancel state, signal counter, and any armed deadline timer.

    Test-only — production code never calls this. Pair with an autouse
    fixture in conftest so each test starts from a clean slate.
    """
    global _signal_count, _deadline_timer, _active_orchestrator
    _event.clear()
    _signal_count = 0
    if _deadline_timer is not None:
        _deadline_timer.cancel()
        _deadline_timer = None
    _active_orchestrator = None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _handler(signum: int, _frame: Any) -> None:
    """Minimal-work SIGINT/SIGTERM handler.

    Order of operations:

    1. Bump signal counter and set the cancel event so pollers see it.
    2. Best-effort notify the orchestrator (flag-only; doesn't unwind).
    3. On the second signal, hard-exit immediately — user is impatient.
    4. On the first signal only, arm a 30s deadline timer as the
       last-resort fallback if cleanup wedges past the deadline.

    Deliberately does NOT call ``sys.exit`` / raise. The cancel event is
    the canonical signal; pollers raise :class:`PipelineCancelled` at
    their own boundaries so cleanup logic stays linear.
    """
    global _signal_count, _deadline_timer
    _signal_count += 1
    _event.set()

    if _active_orchestrator is not None:
        notify = getattr(_active_orchestrator, "notify_signal", None)
        if callable(notify):
            try:
                notify(signal_name=("SIGINT" if signum == signal.SIGINT else "SIGTERM"))
            except Exception:
                pass

    if _signal_count >= 2:
        # Defensive: still drop MLflow's atexit before _exit so a stuck
        # tracking server can't hang in any registered finalizer running
        # before os._exit fires (os._exit itself bypasses atexit).
        _unregister_mlflow_atexit()
        os._exit(130)

    if _deadline_timer is None:
        _deadline_timer = threading.Timer(_DEADLINE_SECONDS, _deadline_exit)
        _deadline_timer.daemon = True
        _deadline_timer.start()


def _deadline_exit() -> None:
    """Hard-exit fallback fired by the deadline timer."""
    _unregister_mlflow_atexit()
    os._exit(130)


def _unregister_mlflow_atexit() -> None:
    """Cancel MLflow's ``_safe_end_run`` atexit handler.

    Same rationale as the CLI parent's signal handler: MLflow registers
    an atexit hook that makes a synchronous HTTP call (default retries
    ~14 minutes) — an unreachable tracking server hangs the whole
    process for those 14 minutes after ``sys.exit``. The orchestrator's
    own cleanup already terminates MLflow runs idempotently, so dropping
    the hook is safe and avoids the wait.
    """
    try:
        import mlflow.tracking.fluent as _fluent

        atexit.unregister(_fluent._safe_end_run)
    except Exception:
        pass


__all__ = [
    "PipelineCancelled",
    "check_cancelled",
    "get_active_orchestrator",
    "install_handler",
    "is_cancelled",
    "reset_for_tests",
    "set_active_orchestrator",
    "sleep_cancellable",
]

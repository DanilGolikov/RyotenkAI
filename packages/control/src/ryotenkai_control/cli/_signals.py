"""Process-wide signal handling for the CLI.

The CLI's long-running commands (``run start``, ``run resume``,
``runs logs --follow``) hold a :class:`PipelineOrchestrator` (or other
subprocess-owning object). When the user hits Ctrl-C or systemd sends
SIGTERM, two things must happen in order:

1. The orchestrator gets a chance to flush state and stop child workers
   gracefully.
2. The process actually exits — even if cleanup is wedged on a network
   call (MLflow's atexit hook is a notorious offender).

The handler in this module is registered exactly once at import time
(safe — ``signal.signal`` is idempotent on the same callable). Commands
that own an orchestrator call :func:`set_active_orchestrator` before
the long-running call and clear it in ``finally``. This keeps the
orchestrator reference out of every command's signature.

Extracted from the legacy ``src/main.py`` monolith — same behaviour,
just import-once instead of line-noise at the top of every train
function.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading

from src.utils.logger import logger

#: 30-second hard deadline. If the orchestrator's cleanup hangs longer
#: than this we fall back to ``os._exit`` to guarantee the process
#: terminates. Long enough for a clean GPU teardown; short enough that
#: a stuck MLflow request doesn't block CI.
_DEADLINE_SECONDS: float = 30.0

#: Global slot for the orchestrator currently owned by a CLI command.
#: Read by :func:`_signal_handler` to forward the signal name; set /
#: cleared by :func:`set_active_orchestrator`.
_active_orchestrator: object | None = None


def set_active_orchestrator(orchestrator: object | None) -> None:
    """Tell the signal handler which orchestrator to notify on Ctrl-C.

    Pass ``None`` from a ``finally`` block to clear the slot — leaving
    a stale reference would notify the wrong object on the next signal.
    """
    global _active_orchestrator
    _active_orchestrator = orchestrator


def _unregister_mlflow_atexit() -> None:
    """Cancel MLflow's ``_safe_end_run`` atexit handler.

    MLflow registers it from ``mlflow.start_run`` and the handler makes
    a synchronous HTTP call (``MlflowClient().set_terminated``). With
    default retries (~14 minutes) an unreachable tracking server hangs
    the whole process after ``sys.exit``. Our orchestrator's ``finally``
    block already does the same teardown idempotently, so dropping the
    atexit hook is safe and avoids the wait.
    """
    try:
        import mlflow.tracking.fluent as _fluent

        atexit.unregister(_fluent._safe_end_run)
        logger.debug("[SIGNAL] MLflow atexit hook unregistered")
    except Exception:
        pass  # mlflow not imported / API moved — neither is fatal


def _signal_handler(signum: int, _frame: object) -> None:
    """SIGINT / SIGTERM handler — see module docstring for the contract."""
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\nReceived {signal_name}, shutting down...", file=sys.stderr)
    logger.warning("Received %s, initiating graceful shutdown", signal_name)

    if _active_orchestrator is not None:
        notify = getattr(_active_orchestrator, "notify_signal", None)
        if callable(notify):
            notify(signal_name=signal_name)

    _unregister_mlflow_atexit()

    exit_code = 130 if signum == signal.SIGINT else 143

    def _deadline_exit() -> None:
        logger.warning("[SIGNAL] cleanup deadline exceeded, forcing exit")
        os._exit(exit_code)

    timer = threading.Timer(_DEADLINE_SECONDS, _deadline_exit)
    timer.daemon = True
    timer.start()
    sys.exit(exit_code)


def install() -> None:
    """Register the SIGINT / SIGTERM handler for this process.

    Called once from :mod:`src.cli.app` at import time. Idempotent —
    re-registering the same callable is a no-op.
    """
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


__all__ = [
    "install",
    "set_active_orchestrator",
]

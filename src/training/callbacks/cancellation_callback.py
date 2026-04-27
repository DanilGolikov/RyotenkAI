"""Phase 9.A — HuggingFace ``TrainerCallback`` for cooperative cancellation.

Closes the gap where SIGTERM caught by :class:`ShutdownHandler` only
sets a thread-local flag — without this callback the HF Trainer never
sees the flag and keeps stepping until the orchestrator-level
``finally`` cleanup. That:

1. Wastes the post-SIGTERM grace window (Supervisor schedules SIGKILL
   after ``--grace`` seconds; trainer needs to checkpoint + exit
   cleanly *before* that).
2. Closes nested MLflow runs as ``FAILED`` instead of ``KILLED``
   because ``run_result.is_failure()`` doesn't distinguish "trainer
   never saw the signal" from "trainer crashed".

Activation
----------

Lives in the trainer's callback list when the supervisor runs the
trainer subprocess. The supervisor sets ``RYOTENKAI_RUNNER_URL`` in
the spawn env (see ``TrainingLauncher._build_job_env``); the same
env var is the activation gate for :class:`RunnerEventCallback`.
:func:`TrainerFactory.create` inserts this callback at **index 0**
of the callback list — BEFORE HF Trainer's auto-registered MLflow
callback — so we set ``control.should_save`` and
``control.should_training_stop`` on the same step where HF will
decide whether to save and continue.

Local-mode runs (no runner attached) skip the callback entirely
because the env gate fails.

What it does on each HF hook
----------------------------

``on_step_end``
    Polls :func:`get_shutdown_handler().should_stop()`. On a fresh
    cancellation request:

    1. Sets ``control.should_save = True`` — HF saves a checkpoint
       at the next save boundary.
    2. Sets ``control.should_training_stop = True`` — HF exits the
       train loop after the save lands.
    3. Logs the transition once (idempotent across subsequent step
       calls; the underlying flag stays raised).

    Order matters: setting ``should_training_stop`` without
    ``should_save`` first would have HF skip the save and exit with
    a stale checkpoint.

``on_train_end``
    No-op in 9.A. The MLflow flush + ``KILLED`` status finalization
    layered in 9.B will hook in here. Keeping the method as an
    explicit no-op now makes the future delta a one-line change.

Signal-safety contract
----------------------

The callback never executes inside a signal handler — HF Trainer
calls these hooks from the main train loop. ``ShutdownHandler``'s
signal handler only flips a flag (no I/O); the callback reads the
flag from the regular Python thread, free to call HTTP / MLflow /
whatever in 9.B without re-entrancy concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import TrainerCallback

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = get_logger(__name__)


__all__ = ["CancellationCallback"]


class CancellationCallback(TrainerCallback):
    """Cooperative-cancellation HF callback driven by ShutdownHandler.

    See module docstring for the full contract.

    Construction is dependency-injected so unit tests can pass a fake
    handler with a deterministic ``should_stop()`` return value
    without registering real signal handlers (which pytest forbids
    in worker threads).
    """

    def __init__(
        self,
        *,
        shutdown_handler: object | None = None,
    ) -> None:
        """Build a callback.

        Args:
            shutdown_handler: object exposing ``.should_stop() -> bool``.
                When ``None`` (production default) we reach into
                :func:`src.training.orchestrator.shutdown_handler.get_shutdown_handler`
                lazily on the first poll. Tests pass a stub.
        """
        self._handler = shutdown_handler
        self._signalled = False  # idempotency: we only log once per cancel

    def _resolve_handler(self) -> object:
        """Resolve the global handler on first use.

        Lazy because importing the orchestrator package at module
        import time pulls heavy training deps into surfaces that
        only need the callback's class (e.g. unit tests of the
        TrainerFactory wiring layer).
        """
        if self._handler is None:
            from src.training.orchestrator.shutdown_handler import (
                get_shutdown_handler,
            )
            self._handler = get_shutdown_handler()
        return self._handler

    # ------------------------------------------------------------------
    # HF TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_step_end(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: object,
    ) -> None:
        """Pump the shutdown flag into HF ``TrainerControl``.

        HF inspects ``control`` after each step to decide whether to
        save / evaluate / stop. We piggy-back on that same loop —
        no extra threads, no polling cost beyond a function call per
        step.
        """
        handler = self._resolve_handler()
        try:
            should_stop = bool(handler.should_stop())  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — defensive
            # An error reading the handler's flag must never crash
            # training. Log once and behave as if no cancellation
            # was requested; the orchestrator-level shutdown path
            # remains the safety-net.
            logger.warning(
                "[CANCELLATION] failed to poll shutdown handler: %s — "
                "continuing without cooperative stop", exc,
            )
            return

        if not should_stop:
            return

        # Order matters — see module docstring.
        # ``should_save = True`` first so HF saves a checkpoint at the
        # next save boundary; ``should_training_stop = True`` after so
        # HF exits AFTER the save (not before).
        control.should_save = True
        control.should_training_stop = True

        if not self._signalled:
            self._signalled = True
            logger.info(
                "[CANCELLATION] shutdown flag observed at step=%d — "
                "set should_save+should_training_stop; trainer will "
                "exit at the next checkpoint boundary",
                getattr(state, "global_step", -1),
            )

    def on_train_end(  # type: ignore[override]
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: object,
    ) -> None:
        """No-op in Phase 9.A.

        Phase 9.B will plug MLflow buffer flush here (with a hard 5s
        timeout via ``concurrent.futures``) so any backlog from a
        flapping MLflow upstream lands before the process exits. We
        keep the empty hook now so the signature is part of the
        public callback contract from the start.
        """
        # Intentionally empty until 9.B.

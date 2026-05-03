"""
ShutdownHandler - Graceful shutdown for training phases.

Handles SIGINT (Ctrl+C) and SIGTERM signals during training,
saving checkpoint before exit to enable resume.

Single Responsibility: Signal handling and graceful shutdown.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType


class ShutdownReason(Enum):
    """Reason for shutdown."""

    SIGINT = "sigint"  # Ctrl+C
    SIGTERM = "sigterm"  # kill signal
    TIMEOUT = "timeout"  # max duration exceeded
    MANUAL = "manual"  # programmatic shutdown


@dataclass
class ShutdownState:
    """
    Current shutdown state.

    Attributes:
        requested: True if shutdown was requested
        reason: Why shutdown was requested
        timestamp: When shutdown was requested
        checkpoint_saved: True if checkpoint was saved
    """

    requested: bool = False
    reason: ShutdownReason | None = None
    timestamp: datetime | None = None
    checkpoint_saved: bool = False

    def request(self, reason: ShutdownReason) -> None:
        """Mark shutdown as requested."""
        if not self.requested:
            self.requested = True
            self.reason = reason
            self.timestamp = datetime.now()
            logger.warning(f"[SHUTDOWN:REQUESTED] reason={reason.value}")


class ShutdownHandler:
    """
    Handles graceful shutdown during training.

    Captures SIGINT/SIGTERM and allows training to save checkpoint
    before exiting.

    Thread-safe: can be accessed from signal handlers.

    Example:
        handler = ShutdownHandler()

        with handler.active():
            # Training loop
            if handler.should_stop():
                handler.save_emergency_checkpoint(trainer, output_dir)

        # Or with callback
        handler = ShutdownHandler(on_shutdown=lambda reason: save_checkpoint())
    """

    def __init__(
        self,
        on_shutdown: Callable[[ShutdownReason], None] | None = None,
    ):
        """
        Initialize ShutdownHandler.

        Args:
            on_shutdown: Optional callback called when shutdown is requested
        """
        self._state = ShutdownState()
        self._on_shutdown = on_shutdown
        self._lock = threading.Lock()
        self._original_sigint: Any = None  # signal.signal return type
        self._original_sigterm: Any = None  # signal.signal return type
        self._registered = False

        logger.debug("[SHUTDOWN:INIT] ShutdownHandler initialized")

    @property
    def state(self) -> ShutdownState:
        """Get current shutdown state (thread-safe)."""
        with self._lock:
            return self._state

    def should_stop(self) -> bool:
        """
        Check if training should stop.

        Thread-safe method to check from training loop.

        Returns:
            True if shutdown was requested
        """
        with self._lock:
            return self._state.requested

    def request_shutdown(self, reason: ShutdownReason) -> None:
        """
        Request graceful shutdown.

        Can be called programmatically or from signal handler.

        Args:
            reason: Reason for shutdown
        """
        with self._lock:
            if self._state.requested:
                return  # Already requested
            self._state.request(reason)

        logger.warning(f"⚠️ Shutdown requested: {reason.value}")

        # Call callback if provided
        if self._on_shutdown:
            try:
                self._on_shutdown(reason)
            except Exception as e:
                logger.error(f"[SHUTDOWN:CALLBACK_ERROR] {e}")

    def mark_checkpoint_saved(self) -> None:
        """Mark that emergency checkpoint was saved."""
        with self._lock:
            self._state.checkpoint_saved = True
        logger.info("✅ Emergency checkpoint saved")

    # =========================================================================
    # SIGNAL HANDLERS
    # =========================================================================

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        """Handle SIGINT (Ctrl+C)."""
        logger.warning("\n⚠️ Received SIGINT (Ctrl+C)")
        self.request_shutdown(ShutdownReason.SIGINT)

    def _handle_sigterm(self, _signum: int, _frame: FrameType | None) -> None:
        """Handle SIGTERM."""
        logger.warning("⚠️ Received SIGTERM")
        self.request_shutdown(ShutdownReason.SIGTERM)

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(self) -> None:
        """
        Register signal handlers.

        Saves original handlers for restoration.
        """
        if self._registered:
            return

        self._original_sigint = signal.signal(signal.SIGINT, self._handle_sigint)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_sigterm)
        self._registered = True

        logger.debug("[SHUTDOWN:REGISTERED] Signal handlers installed")

    def unregister(self) -> None:
        """
        Unregister signal handlers, restore originals.
        """
        if not self._registered:
            return

        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

        self._registered = False
        logger.debug("[SHUTDOWN:UNREGISTERED] Signal handlers restored")

    @contextmanager
    def active(self):
        """
        Context manager for signal handling.

        Registers handlers on enter, unregisters on exit.

        Example:
            with handler.active():
                # Training code
                pass
        """
        self.register()
        try:
            yield self
        finally:
            self.unregister()

    # =========================================================================
    # EMERGENCY CHECKPOINT
    # =========================================================================

    def save_emergency_checkpoint(
        self,
        trainer: Any,
        output_dir: str,
        phase_idx: int | None = None,
    ) -> str | None:
        """
        Save emergency checkpoint before shutdown.

        Args:
            trainer: TRL trainer with model
            output_dir: Directory to save checkpoint
            phase_idx: Optional phase index for naming

        Returns:
            Path to saved checkpoint or None if failed
        """
        from pathlib import Path

        try:
            checkpoint_name = "checkpoint-interrupted"
            if phase_idx is not None:
                checkpoint_name = f"checkpoint-interrupted-phase{phase_idx}"

            checkpoint_path = Path(output_dir) / checkpoint_name
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"💾 Saving emergency checkpoint: {checkpoint_path}")
            trainer.save_model(str(checkpoint_path))

            self.mark_checkpoint_saved()
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"[SHUTDOWN:CHECKPOINT_FAILED] {e}")
            return None

    # =========================================================================
    # TIMEOUT SUPPORT
    # =========================================================================

    def request_timeout_shutdown(self, max_duration_seconds: float) -> None:
        """
        Request shutdown due to timeout.

        Args:
            max_duration_seconds: Maximum allowed training duration
        """
        logger.warning(f"⚠️ Training exceeded max duration: {max_duration_seconds}s")
        self.request_shutdown(ShutdownReason.TIMEOUT)

    # =========================================================================
    # STATE INFO
    # =========================================================================

    def get_shutdown_info(self) -> dict[str, Any]:
        """
        Get shutdown information for logging/state.

        Returns:
            Dict with shutdown details
        """
        with self._lock:
            return {
                "requested": self._state.requested,
                "reason": self._state.reason.value if self._state.reason else None,
                "timestamp": self._state.timestamp.isoformat() if self._state.timestamp else None,
                "checkpoint_saved": self._state.checkpoint_saved,
            }

    def __repr__(self) -> str:
        return f"ShutdownHandler(requested={self._state.requested}, registered={self._registered})"


# =============================================================================
# GLOBAL HANDLER (for use in callbacks)
# =============================================================================


_global_handler: ShutdownHandler | None = None


def get_shutdown_handler() -> ShutdownHandler:
    """
    Get or create global shutdown handler.

    For use in TRL trainer callbacks.
    """
    global _global_handler
    if _global_handler is None:
        _global_handler = ShutdownHandler()
    return _global_handler


def reset_shutdown_handler() -> None:
    """Reset global shutdown handler (for testing)."""
    global _global_handler
    if _global_handler is not None:
        _global_handler.unregister()
    _global_handler = None


__all__ = [
    "ShutdownHandler",
    "ShutdownReason",
    "ShutdownState",
    "get_shutdown_handler",
    "reset_shutdown_handler",
]

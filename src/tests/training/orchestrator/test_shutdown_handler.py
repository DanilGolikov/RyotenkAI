"""
Unit tests for ShutdownHandler - Graceful Shutdown.

Tests:
- Signal registration and unregistration
- Shutdown state management
- Thread-safety
- Emergency checkpoint saving
"""

from __future__ import annotations

import signal
import threading
import time
from unittest.mock import MagicMock

from src.training.orchestrator.shutdown_handler import (
    ShutdownHandler,
    ShutdownReason,
    ShutdownState,
    get_shutdown_handler,
    reset_shutdown_handler,
)

# =============================================================================
# SHUTDOWN STATE TESTS
# =============================================================================


class TestShutdownState:
    """Test ShutdownState dataclass."""

    def test_initial_state(self):
        """Test initial state is not requested."""
        state = ShutdownState()

        assert state.requested is False
        assert state.reason is None
        assert state.timestamp is None
        assert state.checkpoint_saved is False

    def test_request_sets_all_fields(self):
        """Test request() sets all fields correctly."""
        state = ShutdownState()
        state.request(ShutdownReason.SIGINT)

        assert state.requested is True
        assert state.reason == ShutdownReason.SIGINT
        assert state.timestamp is not None

    def test_request_only_once(self):
        """Test that request only works once."""
        state = ShutdownState()
        state.request(ShutdownReason.SIGINT)
        first_timestamp = state.timestamp

        # Second request should not change anything
        state.request(ShutdownReason.SIGTERM)

        assert state.reason == ShutdownReason.SIGINT  # Still SIGINT
        assert state.timestamp == first_timestamp


# =============================================================================
# SHUTDOWN HANDLER TESTS
# =============================================================================


class TestShutdownHandler:
    """Test ShutdownHandler class."""

    def setup_method(self):
        """Reset global handler before each test."""
        reset_shutdown_handler()

    def teardown_method(self):
        """Cleanup after each test."""
        reset_shutdown_handler()

    def test_initialization(self):
        """Test ShutdownHandler initializes correctly."""
        handler = ShutdownHandler()

        assert handler.should_stop() is False
        assert handler._registered is False

    def test_register_and_unregister(self):
        """Test signal handler registration."""
        handler = ShutdownHandler()

        handler.register()
        assert handler._registered is True

        handler.unregister()
        assert handler._registered is False

    def test_should_stop_initially_false(self):
        """Test should_stop() returns False initially."""
        handler = ShutdownHandler()
        assert handler.should_stop() is False

    def test_request_shutdown_sets_should_stop(self):
        """Test request_shutdown sets should_stop to True."""
        handler = ShutdownHandler()
        handler.request_shutdown(ShutdownReason.MANUAL)

        assert handler.should_stop() is True

    def test_request_shutdown_with_callback(self):
        """Test callback is called on shutdown request."""
        callback = MagicMock()
        handler = ShutdownHandler(on_shutdown=callback)

        handler.request_shutdown(ShutdownReason.SIGINT)

        callback.assert_called_once_with(ShutdownReason.SIGINT)

    def test_mark_checkpoint_saved(self):
        """Test mark_checkpoint_saved updates state."""
        handler = ShutdownHandler()
        handler.request_shutdown(ShutdownReason.SIGINT)
        handler.mark_checkpoint_saved()

        assert handler.state.checkpoint_saved is True

    def test_get_shutdown_info(self):
        """Test get_shutdown_info returns correct dict."""
        handler = ShutdownHandler()
        handler.request_shutdown(ShutdownReason.TIMEOUT)

        info = handler.get_shutdown_info()

        assert info["requested"] is True
        assert info["reason"] == "timeout"
        assert info["timestamp"] is not None
        assert info["checkpoint_saved"] is False

    def test_context_manager(self):
        """Test active() context manager."""
        handler = ShutdownHandler()

        with handler.active():
            assert handler._registered is True

        assert handler._registered is False

    def test_double_register_is_safe(self):
        """Test that double register is idempotent."""
        handler = ShutdownHandler()
        handler.register()
        handler.register()  # Should not raise

        assert handler._registered is True
        handler.unregister()

    def test_double_unregister_is_safe(self):
        """Test that double unregister is idempotent."""
        handler = ShutdownHandler()
        handler.unregister()  # Should not raise
        handler.unregister()  # Should not raise

        assert handler._registered is False


# =============================================================================
# THREAD-SAFETY TESTS
# =============================================================================


class TestShutdownHandlerThreadSafety:
    """Test ShutdownHandler thread-safety."""

    def setup_method(self):
        reset_shutdown_handler()

    def teardown_method(self):
        reset_shutdown_handler()

    def test_concurrent_should_stop_calls(self):
        """Test should_stop() is thread-safe."""
        handler = ShutdownHandler()
        results = []

        def check_stop():
            for _ in range(100):
                results.append(handler.should_stop())
                time.sleep(0.001)

        # Start multiple threads
        threads = [threading.Thread(target=check_stop) for _ in range(5)]
        for t in threads:
            t.start()

        # Request shutdown in the middle
        time.sleep(0.05)
        handler.request_shutdown(ShutdownReason.MANUAL)

        for t in threads:
            t.join()

        # All results should be boolean and some should be True after shutdown
        assert all(isinstance(r, bool) for r in results)
        assert any(r is True for r in results)

    def test_concurrent_request_shutdown(self):
        """Test request_shutdown is thread-safe."""
        handler = ShutdownHandler()

        def request():
            handler.request_shutdown(ShutdownReason.SIGINT)

        threads = [threading.Thread(target=request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only be requested once (reason should be SIGINT)
        assert handler.should_stop() is True
        assert handler.state.reason == ShutdownReason.SIGINT


# =============================================================================
# EMERGENCY CHECKPOINT TESTS
# =============================================================================


class TestEmergencyCheckpoint:
    """Test emergency checkpoint saving."""

    def test_save_emergency_checkpoint(self, tmp_path):
        """Test save_emergency_checkpoint creates checkpoint."""
        handler = ShutdownHandler()
        handler.request_shutdown(ShutdownReason.SIGINT)

        # Create mock trainer
        mock_trainer = MagicMock()

        checkpoint_path = handler.save_emergency_checkpoint(
            trainer=mock_trainer,
            output_dir=str(tmp_path),
            phase_idx=1,
        )

        assert checkpoint_path is not None
        assert "checkpoint-interrupted-phase1" in checkpoint_path
        mock_trainer.save_model.assert_called_once()
        assert handler.state.checkpoint_saved is True

    def test_save_emergency_checkpoint_failure(self, tmp_path):
        """Test save_emergency_checkpoint handles errors."""
        handler = ShutdownHandler()
        handler.request_shutdown(ShutdownReason.SIGINT)

        # Create mock trainer that raises error
        mock_trainer = MagicMock()
        mock_trainer.save_model.side_effect = RuntimeError("Save failed")

        checkpoint_path = handler.save_emergency_checkpoint(
            trainer=mock_trainer,
            output_dir=str(tmp_path),
            phase_idx=0,
        )

        assert checkpoint_path is None
        assert handler.state.checkpoint_saved is False


# =============================================================================
# GLOBAL HANDLER TESTS
# =============================================================================


class TestGlobalHandler:
    """Test global shutdown handler functions."""

    def setup_method(self):
        reset_shutdown_handler()

    def teardown_method(self):
        reset_shutdown_handler()

    def test_get_shutdown_handler_creates_singleton(self):
        """Test get_shutdown_handler creates singleton."""
        handler1 = get_shutdown_handler()
        handler2 = get_shutdown_handler()

        assert handler1 is handler2

    def test_reset_shutdown_handler(self):
        """Test reset_shutdown_handler creates new instance."""
        handler1 = get_shutdown_handler()
        handler1.request_shutdown(ShutdownReason.MANUAL)

        reset_shutdown_handler()

        handler2 = get_shutdown_handler()
        assert handler2 is not handler1
        assert handler2.should_stop() is False


# =============================================================================
# SIGNAL HANDLER TESTS (simulated)
# =============================================================================


class TestSignalHandling:
    """Test signal handling (simulated, no actual signals sent)."""

    def setup_method(self):
        reset_shutdown_handler()

    def teardown_method(self):
        reset_shutdown_handler()

    def test_sigint_handler(self):
        """Test SIGINT handler sets shutdown state."""
        handler = ShutdownHandler()

        # Simulate SIGINT by calling internal handler
        handler._handle_sigint(signal.SIGINT, None)

        assert handler.should_stop() is True
        assert handler.state.reason == ShutdownReason.SIGINT

    def test_sigterm_handler(self):
        """Test SIGTERM handler sets shutdown state."""
        handler = ShutdownHandler()

        # Simulate SIGTERM by calling internal handler
        handler._handle_sigterm(signal.SIGTERM, None)

        assert handler.should_stop() is True
        assert handler.state.reason == ShutdownReason.SIGTERM

    def test_timeout_shutdown(self):
        """Test timeout shutdown request."""
        handler = ShutdownHandler()

        handler.request_timeout_shutdown(max_duration_seconds=3600)

        assert handler.should_stop() is True
        assert handler.state.reason == ShutdownReason.TIMEOUT


# =============================================================================
# SHUTDOWN REASON TESTS
# =============================================================================


class TestShutdownReason:
    """Test ShutdownReason enum."""

    def test_all_reasons_have_values(self):
        """Test all reasons have string values."""
        assert ShutdownReason.SIGINT.value == "sigint"
        assert ShutdownReason.SIGTERM.value == "sigterm"
        assert ShutdownReason.TIMEOUT.value == "timeout"
        assert ShutdownReason.MANUAL.value == "manual"

    def test_reason_str_representation(self):
        """Test reason string representation."""
        reason = ShutdownReason.SIGINT
        assert str(reason.value) == "sigint"

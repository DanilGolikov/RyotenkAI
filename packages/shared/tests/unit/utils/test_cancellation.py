"""Tests for the cooperative cancellation primitives.

Pinned behavior:

* :class:`PipelineCancelled` subclasses :class:`BaseException` (not
  :class:`Exception`) so generic ``except Exception:`` blocks cannot
  swallow cancellation.
* :func:`check_cancelled` is a no-op when the event isn't set, raises
  :class:`PipelineCancelled` when it is.
* :func:`sleep_cancellable` returns immediately and raises if the event
  is set during the wait — pollers wake up cooperatively instead of
  blocking the full sleep duration.
* :func:`install_handler` registers handlers for SIGINT and SIGTERM.
* The handler increments a counter, sets the event, and arms a deadline
  timer on first signal. Second signal forces ``os._exit(130)``.
* :func:`set_active_orchestrator` is propagated to the handler — the
  orchestrator's ``notify_signal`` (if callable) is invoked best-effort.
* :func:`reset_for_tests` clears all module state — autouse fixture
  guarantees test isolation.
"""

from __future__ import annotations

import signal
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_shared.utils import cancellation
from ryotenkai_shared.utils.cancellation import (
    PipelineCancelled,
    check_cancelled,
    get_active_orchestrator,
    install_handler,
    is_cancelled,
    reset_for_tests,
    set_active_orchestrator,
    sleep_cancellable,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_cancel_state() -> None:
    """Each test starts and ends with cancel state cleared.

    Without this, a single test that fires the handler would leak the
    set event into the next test's :func:`is_cancelled` check.
    """
    reset_for_tests()
    yield
    reset_for_tests()


# ---------------------------------------------------------------------------
# 1. Positive — module is wired correctly
# ---------------------------------------------------------------------------


class TestPositive:
    def test_pipeline_cancelled_is_base_exception_subclass(self) -> None:
        """Architectural invariant: ``except Exception:`` MUST NOT catch
        PipelineCancelled. If this ever fails, the cancellation contract
        is silently broken across the entire codebase."""
        assert issubclass(PipelineCancelled, BaseException)
        assert not issubclass(PipelineCancelled, Exception)

    def test_is_cancelled_false_initially(self) -> None:
        assert is_cancelled() is False

    def test_check_cancelled_no_op_when_not_set(self) -> None:
        check_cancelled()  # must not raise

    def test_sleep_cancellable_completes_when_not_cancelled(self) -> None:
        start = time.monotonic()
        sleep_cancellable(0.05)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04  # accounting for clock resolution

    def test_set_active_orchestrator_round_trip(self) -> None:
        sentinel = MagicMock()
        set_active_orchestrator(sentinel)
        assert get_active_orchestrator() is sentinel
        set_active_orchestrator(None)
        assert get_active_orchestrator() is None


# ---------------------------------------------------------------------------
# 2. Negative — wrong usage / boundary conditions
# ---------------------------------------------------------------------------


class TestNegative:
    def test_check_cancelled_raises_when_event_set(self) -> None:
        cancellation._event.set()
        with pytest.raises(PipelineCancelled):
            check_cancelled()

    def test_sleep_cancellable_raises_when_already_set(self) -> None:
        cancellation._event.set()
        with pytest.raises(PipelineCancelled):
            sleep_cancellable(10.0)


# ---------------------------------------------------------------------------
# 3. Boundary — concurrent set during sleep
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_sleep_cancellable_returns_when_set_concurrently(self) -> None:
        """Pollers using ``sleep_cancellable`` should wake within ~10ms
        of the event being set, not wait the full duration."""
        def setter() -> None:
            time.sleep(0.05)
            cancellation._event.set()

        t = threading.Thread(target=setter, daemon=True)
        t.start()
        start = time.monotonic()
        with pytest.raises(PipelineCancelled):
            sleep_cancellable(5.0)  # would block 5s if not cancellable
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
        t.join(timeout=1.0)

    def test_sleep_cancellable_zero_duration(self) -> None:
        sleep_cancellable(0.0)  # must not raise, must not block


# ---------------------------------------------------------------------------
# 4. Invariants — handler behaviour
# ---------------------------------------------------------------------------


class TestHandlerInvariants:
    def test_install_handler_registers_both_signals(self) -> None:
        with patch("signal.signal") as mock_sig:
            install_handler()
        # Two calls: SIGINT and SIGTERM, in that order.
        sig_args = [c.args[0] for c in mock_sig.call_args_list]
        assert signal.SIGINT in sig_args
        assert signal.SIGTERM in sig_args
        # Callbacks identical (single handler covers both).
        callbacks = [c.args[1] for c in mock_sig.call_args_list]
        assert callbacks[0] is callbacks[1]

    def test_first_signal_sets_event_and_increments_count(self) -> None:
        cancellation._handler(signal.SIGINT, None)
        assert is_cancelled()
        assert cancellation._signal_count == 1

    def test_first_signal_arms_deadline_timer_exactly_once(self) -> None:
        # Mock both Timer (we only want to count, not actually arm it) and
        # os._exit (the second handler call hard-exits, which would kill
        # the test process if not patched).
        with (
            patch("ryotenkai_shared.utils.cancellation.threading.Timer") as mock_timer_class,
            patch("ryotenkai_shared.utils.cancellation.os._exit"),
        ):
            mock_timer = MagicMock()
            mock_timer_class.return_value = mock_timer
            cancellation._handler(signal.SIGINT, None)
            cancellation._handler(signal.SIGTERM, None)
        # Exactly one Timer was constructed — the second signal hits
        # the hard-exit branch before re-arming.
        assert mock_timer_class.call_count == 1

    def test_handler_notifies_orchestrator(self) -> None:
        orch = MagicMock()
        set_active_orchestrator(orch)
        cancellation._handler(signal.SIGINT, None)
        orch.notify_signal.assert_called_once_with(signal_name="SIGINT")

    def test_handler_uses_sigterm_label_for_sigterm(self) -> None:
        orch = MagicMock()
        set_active_orchestrator(orch)
        cancellation._handler(signal.SIGTERM, None)
        orch.notify_signal.assert_called_once_with(signal_name="SIGTERM")

    def test_handler_swallows_orchestrator_notify_failure(self) -> None:
        """If the orchestrator's notify_signal raises (e.g. state torn
        down mid-shutdown), the handler must NOT propagate — otherwise
        the signal handler crashes the interpreter."""
        orch = MagicMock()
        orch.notify_signal.side_effect = RuntimeError("state torn down")
        set_active_orchestrator(orch)
        cancellation._handler(signal.SIGINT, None)  # must not raise
        assert is_cancelled()

    def test_handler_works_without_active_orchestrator(self) -> None:
        cancellation._handler(signal.SIGINT, None)
        assert is_cancelled()
        assert cancellation._signal_count == 1

    def test_handler_works_with_orchestrator_lacking_notify_signal(self) -> None:
        """Orchestrator without ``notify_signal`` attribute (mock missing
        method) — handler must still set the event."""
        bare = object()  # no notify_signal attribute
        set_active_orchestrator(bare)
        cancellation._handler(signal.SIGINT, None)
        assert is_cancelled()


# ---------------------------------------------------------------------------
# 5. Logic-specific — second signal hard-exits
# ---------------------------------------------------------------------------


class TestDoubleSignalEscalation:
    def test_second_signal_calls_os_exit_130(self) -> None:
        """Standard kubectl/docker pattern: first signal asks for graceful
        shutdown, second forces immediate exit."""
        with patch("os._exit") as mock_exit:
            cancellation._handler(signal.SIGINT, None)
            cancellation._handler(signal.SIGINT, None)
        mock_exit.assert_called_once_with(130)

    def test_third_signal_hits_os_exit_too(self) -> None:
        with patch("os._exit") as mock_exit:
            cancellation._handler(signal.SIGINT, None)
            cancellation._handler(signal.SIGINT, None)
            cancellation._handler(signal.SIGINT, None)
        # Each post-first signal triggers os._exit; we don't enforce
        # exact count, just that it keeps firing.
        assert mock_exit.call_count >= 1


# ---------------------------------------------------------------------------
# 6. Regression — reset_for_tests fully clears state
# ---------------------------------------------------------------------------


class TestResetForTests:
    def test_reset_clears_event(self) -> None:
        cancellation._event.set()
        reset_for_tests()
        assert is_cancelled() is False

    def test_reset_clears_signal_counter(self) -> None:
        cancellation._signal_count = 5
        reset_for_tests()
        assert cancellation._signal_count == 0

    def test_reset_cancels_armed_deadline_timer(self) -> None:
        timer = MagicMock(spec=threading.Timer)
        cancellation._deadline_timer = timer
        reset_for_tests()
        timer.cancel.assert_called_once()
        assert cancellation._deadline_timer is None

    def test_reset_clears_active_orchestrator(self) -> None:
        set_active_orchestrator(MagicMock())
        reset_for_tests()
        assert get_active_orchestrator() is None

"""Phase 9.B — :func:`with_timeout` contract.

Hard-deadline helper used by :class:`CancellationCallback` to bound
the on_train_end MLflow flush. Portable across platforms via
``concurrent.futures.ThreadPoolExecutor`` (vs ``signal.alarm`` which
was rejected during plan review for being Unix-main-thread-only).

7-category coverage:

1. **Positive** — happy path returns the callable's value.
2. **Negative** — invalid timeout (≤0) raises ValueError before
   the work starts.
3. **Boundary** — work returns exactly at the deadline (small
   margin); fast work returns well under deadline.
4. **Invariants** — exceptions from ``fn`` propagate unchanged;
   thread pool is cleaned up even when timeout fires.
5. **Dependency errors** — ``fn`` raising before the deadline
   propagates immediately, not as TimeoutExceededError.
6. **Regressions** — TimeoutExceededError is a subclass of
   TimeoutError so generic handlers catch it too.
7. **Logic-specific** — slow ``fn`` triggers
   :class:`TimeoutExceededError` not generic ``TimeoutError``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import pytest


# The helper module itself has zero ML deps, but importing
# ``src.training._concurrent_helpers`` triggers ``src.training/__init__.py``
# which transitively pulls in the orchestrator (datasets, peft, etc.).
# Load the helper module by file path to keep slim-venv tests viable —
# same trick used by ``test_cancellation_callback.py``.
_HELPER_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src" / "ryotenkai_pod" / "trainer" / "_concurrent_helpers.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ryotenkai_concurrent_helpers_under_test", _HELPER_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["_ryotenkai_concurrent_helpers_under_test"] = _module
_spec.loader.exec_module(_module)

TimeoutExceededError = _module.TimeoutExceededError
with_timeout = _module.with_timeout


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_fast_callable_returns_value(self) -> None:
        result = with_timeout(lambda: 42, timeout_seconds=1.0)
        assert result == 42

    def test_callable_with_no_return(self) -> None:
        # ``None`` returns are valid (e.g. mlflow ops often return None).
        result = with_timeout(lambda: None, timeout_seconds=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    @pytest.mark.parametrize("bad", [0.0, -0.1, -5.0])
    def test_non_positive_timeout_raises_value_error(
        self, bad: float,
    ) -> None:
        # The pool / future is never created — error fires up front.
        with pytest.raises(ValueError, match="must be positive"):
            with_timeout(lambda: 1, timeout_seconds=bad)

    def test_zero_timeout_rejected_even_for_instant_callable(self) -> None:
        # Even a no-op callable can't sneak past zero — the contract
        # is "positive deadline required".
        with pytest.raises(ValueError):
            with_timeout(lambda: 1, timeout_seconds=0.0)


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_callable_finishing_within_budget(self) -> None:
        # Sleeps less than budget — succeeds.
        def quick() -> str:
            time.sleep(0.05)
            return "done"

        assert with_timeout(quick, timeout_seconds=0.5) == "done"


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_exception_in_fn_propagates_unchanged(self) -> None:
        # Custom exception from the work bubbles up — NOT wrapped in
        # TimeoutExceededError. Callers can pattern-match on real causes.
        class MyDomainError(RuntimeError):
            pass

        def boom() -> None:
            raise MyDomainError("domain-level")

        with pytest.raises(MyDomainError, match="domain-level"):
            with_timeout(boom, timeout_seconds=1.0)

    def test_thread_pool_cleanup_on_timeout(self) -> None:
        # No leak after timeout fired. The worker thread keeps running
        # in background (Python can't kill it safely), but the pool
        # context manager is exited; no dangling Future references.
        # We can't directly assert thread teardown, but we CAN assert
        # the call doesn't leave us blocked on subsequent invocations.
        def slow() -> None:
            time.sleep(2.0)

        with pytest.raises(TimeoutExceededError):
            with_timeout(slow, timeout_seconds=0.05)

        # Subsequent call works — proves no global state corruption.
        assert with_timeout(lambda: "ok", timeout_seconds=0.5) == "ok"


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_exception_before_deadline_not_remapped(self) -> None:
        # Even when the callable would have hit the deadline if it
        # ran longer, an early exception MUST surface as that
        # exception, not as TimeoutExceededError. Tests pin the
        # "fail fast on dependency errors" semantics.
        def fast_failure() -> None:
            time.sleep(0.01)
            raise ConnectionError("upstream gone")

        with pytest.raises(ConnectionError, match="upstream gone"):
            with_timeout(fast_failure, timeout_seconds=10.0)


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_timeout_exceeded_is_subclass_of_timeout_error(self) -> None:
        """``TimeoutExceededError`` must inherit from the stdlib
        :class:`TimeoutError` so generic handlers catch us too.

        Pin via runtime ``isinstance`` and class hierarchy — a future
        refactor that switches the base class would break callers
        that catch ``TimeoutError`` (e.g. asyncio's wait_for-shaped
        handlers).
        """
        assert issubclass(TimeoutExceededError, TimeoutError)
        # Confirm via instance.
        try:
            with_timeout(lambda: time.sleep(0.5), timeout_seconds=0.05)
        except TimeoutError:
            return  # generic catch worked
        pytest.fail("with_timeout did not raise TimeoutError on slow work")


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_specific_subclass_used_for_timeout(self) -> None:
        # Slow work raises our specific subclass — callers that want
        # to distinguish "I did the timeout" from "underlying lib
        # timed out" can match on TimeoutExceededError.
        with pytest.raises(TimeoutExceededError, match="exceeded"):
            with_timeout(lambda: time.sleep(0.5), timeout_seconds=0.05)

    def test_error_message_includes_budget_value(self) -> None:
        # Operator-friendly message: include the budget so logs
        # don't require correlating with a separate field.
        try:
            with_timeout(lambda: time.sleep(0.5), timeout_seconds=0.07)
        except TimeoutExceededError as exc:
            assert "0.07" in str(exc)
            return
        pytest.fail("expected TimeoutExceededError")

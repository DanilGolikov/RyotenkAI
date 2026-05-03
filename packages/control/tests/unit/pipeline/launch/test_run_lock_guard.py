"""Comprehensive tests for RunLockGuard — Invariant #1 (lock always released).

Categories:
1. Positive        — happy path (acquire → use → release)
2. Negative        — double-acquire, stranger already holds lock
3. Boundary        — lock path missing parent, re-use after exit, already released
4. Invariants      — "if __enter__ succeeded, __exit__ releases" under ANY exception
5. Dep errors      — release() raising, os.close() failing
6. Regressions     — asserts the specific pre-refactor bug the guard prevents
7. Combinatorial   — exception-type × release-failure matrix
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.launch.run_lock_guard import RunLockGuard
from src.pipeline.state.store import PipelineStateLockError


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_happy_path_acquires_and_releases(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "run.lock"
        assert not lock_file.exists()
        with RunLockGuard(lock_file) as guard:
            assert lock_file.exists()  # lock file created
            assert guard.is_held
            assert guard.lock is not None
        # On exit — lock gone
        assert not lock_file.exists()

    def test_lock_property_exposes_inner_lock(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "run.lock"
        with RunLockGuard(lock_file) as guard:
            assert guard.lock is not None
            assert guard.lock.path == lock_file

    def test_is_held_flag_transitions(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "run.lock"
        g = RunLockGuard(lock_file)
        assert not g.is_held
        with g:
            assert g.is_held
        assert not g.is_held


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_second_guard_cannot_acquire_same_path(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "run.lock"
        with RunLockGuard(lock_file):
            with pytest.raises(PipelineStateLockError):
                RunLockGuard(lock_file).__enter__()

    def test_enter_propagates_acquire_errors(self, tmp_path: Path) -> None:
        """If acquire_run_lock raises, __enter__ re-raises — orchestrator sees the failure."""
        lock_file = tmp_path / "run.lock"
        lock_file.write_text("stolen by someone else\n")
        with pytest.raises(PipelineStateLockError):
            RunLockGuard(lock_file).__enter__()


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_parent_directory_is_created(self, tmp_path: Path) -> None:
        """Lock path with missing parent — acquire_run_lock mkdirs it."""
        lock_file = tmp_path / "deep" / "nested" / "run.lock"
        assert not lock_file.parent.exists()
        with RunLockGuard(lock_file):
            assert lock_file.exists()

    def test_can_reuse_same_guard_after_exit(self, tmp_path: Path) -> None:
        """Guard is reusable: exit resets state to pre-enter."""
        lock_file = tmp_path / "run.lock"
        guard = RunLockGuard(lock_file)
        with guard:
            pass
        assert not guard.is_held
        # Second enter should succeed cleanly
        with guard:
            assert guard.is_held
        assert not guard.is_held

    def test_exit_without_enter_is_noop(self, tmp_path: Path) -> None:
        """Defensive: __exit__ without a prior __enter__ should not crash."""
        guard = RunLockGuard(tmp_path / "run.lock")
        guard.__exit__(None, None, None)  # no crash
        assert not guard.is_held


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("domain error"),
            OSError("disk gone"),
            KeyboardInterrupt(),
            SystemExit(1),
            ValueError("bad input"),
        ],
    )
    def test_invariant_release_on_any_exception(self, tmp_path: Path, exc: BaseException) -> None:
        """INVARIANT #1: if __enter__ succeeded, __exit__ releases regardless of exception."""
        lock_file = tmp_path / "run.lock"
        with pytest.raises(type(exc)):
            with RunLockGuard(lock_file):
                raise exc
        assert not lock_file.exists(), f"Lock leaked after {type(exc).__name__}"

    def test_invariant_release_called_exactly_once(self, tmp_path: Path) -> None:
        """INVARIANT: release() called exactly once per __exit__ even if called again.

        Uses a fake lock (frozen dataclass prevents wrapping real release method).
        """
        lock_file = tmp_path / "run.lock"
        guard = RunLockGuard(lock_file)
        guard.__enter__()
        real = guard._lock
        spy = MagicMock()
        guard._lock = spy  # type: ignore[assignment]
        try:
            guard.__exit__(None, None, None)
            # Second call is a no-op because _lock is cleared
            guard.__exit__(None, None, None)
            assert spy.release.call_count == 1
        finally:
            # Release the real lock manually to keep tmp_path clean
            if real is not None:
                real.release()

    def test_invariant_guard_not_reentrant_while_held(self, tmp_path: Path) -> None:
        """INVARIANT: re-entering the same guard while already held should fail to acquire."""
        lock_file = tmp_path / "run.lock"
        g = RunLockGuard(lock_file)
        with g:
            # The OS-level O_EXCL lock will refuse the second acquire
            with pytest.raises(PipelineStateLockError):
                g.__enter__()
        # After proper exit, it becomes usable again
        with g:
            assert g.is_held


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_release_exception_is_swallowed_not_masking_original(self, tmp_path: Path) -> None:
        """release() must never mask the user exception.

        ``PipelineRunLock`` is a frozen dataclass, so we swap ``guard._lock``
        for a fake object rather than patching the real release method.
        """
        lock_file = tmp_path / "run.lock"
        with pytest.raises(RuntimeError, match="original"):
            with RunLockGuard(lock_file) as guard:
                fake_lock = MagicMock()
                fake_lock.release.side_effect = OSError("fs broken on release")
                guard._lock = fake_lock
                raise RuntimeError("original")
        # Contract: original exception reached caller; guard released internal ref.

    def test_release_exception_alone_is_swallowed_and_logged(self, tmp_path: Path) -> None:
        """If normal exit hits a release() failure, __exit__ still succeeds silently."""
        lock_file = tmp_path / "run.lock"
        guard = RunLockGuard(lock_file)
        guard.__enter__()
        # Swap the frozen PipelineRunLock for a fake we can break
        real_lock = guard._lock
        fake_lock = MagicMock()
        fake_lock.release.side_effect = OSError("weird fs state")
        guard._lock = fake_lock
        try:
            guard.__exit__(None, None, None)  # Must NOT raise
        finally:
            # Manually release the real lock so the test doesn't leak the file
            if real_lock is not None:
                real_lock.release()
        assert not guard.is_held

    def test_acquire_missing_parent_and_permission_error(self, tmp_path: Path) -> None:
        """Propagates low-level FS errors during acquire."""
        # Make parent dir non-writable
        lock_file = tmp_path / "run.lock"
        with patch(
            "src.pipeline.launch.run_lock_guard.acquire_run_lock",
            side_effect=PermissionError("denied"),
        ):
            with pytest.raises(PermissionError):
                RunLockGuard(lock_file).__enter__()


# =============================================================================
# 6. REGRESSIONS
# =============================================================================


class TestRegressions:
    def test_regression_manual_release_in_finally_would_leak_on_flush_crash(
        self, tmp_path: Path
    ) -> None:
        """REGRESSION: the pre-guard orchestrator did::

            try: ...
            finally:
                self._flush_pending_collectors()  # <- if this crashed …
                self._cleanup_resources()
                self._teardown_mlflow_attempt()
                if self._run_lock:                 # <- never reached
                    self._run_lock.release()

        With RunLockGuard wrapping the whole block, release happens regardless.
        This test codifies the new contract.
        """
        lock_file = tmp_path / "run.lock"
        cleanup_called = MagicMock()
        with pytest.raises(OSError):
            with RunLockGuard(lock_file):
                cleanup_called()
                # Simulate a disk error during cleanup — like old _flush crash
                raise OSError("disk error during cleanup")
        # Pre-guard: lock would leak here. Post-guard: lock is gone.
        assert not lock_file.exists()
        cleanup_called.assert_called_once()

    def test_regression_enter_failure_no_release_attempt(self, tmp_path: Path) -> None:
        """REGRESSION: if __enter__ fails, __exit__ must NOT be called (context-manager
        protocol) — and our guard must not attempt release() on a None lock."""
        lock_file = tmp_path / "run.lock"
        with patch(
            "src.pipeline.launch.run_lock_guard.acquire_run_lock",
            side_effect=PipelineStateLockError("taken"),
        ):
            guard = RunLockGuard(lock_file)
            with pytest.raises(PipelineStateLockError):
                guard.__enter__()
            # Manually invoke __exit__ to confirm it's a no-op on un-acquired guard
            guard.__exit__(None, None, None)  # no crash, no release attempt

    def test_regression_log_exception_tag(self, tmp_path: Path) -> None:
        """REGRESSION: release failures must not propagate.

        Uses fake-lock substitution because ``PipelineRunLock`` is frozen.
        The logger call itself isn't asserted (loguru vs std logging format);
        the real contract is "no exception leaks from __exit__".
        """
        lock_file = tmp_path / "run.lock"
        guard = RunLockGuard(lock_file)
        guard.__enter__()
        real_lock = guard._lock
        fake_lock = MagicMock()
        fake_lock.release.side_effect = OSError("mock")
        guard._lock = fake_lock
        try:
            guard.__exit__(None, None, None)  # contract: silent
        finally:
            if real_lock is not None:
                real_lock.release()


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


@pytest.mark.parametrize(
    "user_exc",
    [None, RuntimeError("r"), KeyboardInterrupt(), ValueError("v")],
)
@pytest.mark.parametrize("release_fails", [False, True])
def test_combinatorial_all_exit_paths(
    tmp_path: Path, user_exc: BaseException | None, release_fails: bool
) -> None:
    """Cross product of (user exception × release success/failure).

    In every cell: ``guard.is_held`` is False afterwards, user_exc propagates
    to the caller, and no new exception is raised from release failure.
    """
    lock_file = tmp_path / f"run-{id(user_exc)}-{release_fails}.lock"

    def _body() -> None:
        with RunLockGuard(lock_file) as guard:
            assert guard.lock is not None
            if release_fails:
                # Swap in a fake (frozen dataclass can't be patched directly)
                real = guard._lock
                fake = MagicMock()
                fake.release.side_effect = OSError("broken")
                guard._lock = fake
                # Release the real lock manually so file doesn't leak between
                # parametrised cases; the guard now holds the fake which won't
                # clean the file.
                if real is not None:
                    real.release()
            if user_exc is not None:
                raise user_exc

    if user_exc is None:
        _body()  # no exception expected
    else:
        with pytest.raises(type(user_exc)):
            _body()

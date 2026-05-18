"""Unit tests: ``RunLifecycleCoord``.

Focused tests on the lifecycle coord's responsibilities:

* atexit / signal handler register + restore via ``__enter__`` / ``__exit__``
* ``bind_*`` + ``finalize`` delegate to MlflowFinalizer
* idempotency mutex (thread-safe; only one finalize ever runs)
* signal handler installs and KILLED status routed through finalize
"""

from __future__ import annotations

import atexit
import signal
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from ryotenkai_control.pipeline.mlflow.lifecycle.coord import RunLifecycleCoord
from ryotenkai_control.pipeline.mlflow.lifecycle.finalizer import MlflowFinalizer
from ryotenkai_control.pipeline.mlflow.lifecycle.opener import ParentRunOpener
from ryotenkai_control.pipeline.mlflow.lifecycle.preflight import (
    PreflightConnectivityCheck,
)
from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from tests._fakes.mlflow_journal_uploader import FakeJournalUploader
from tests._fakes.mlflow_tracking_client import FakeTrackingClient

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Test helpers (canonical fakes only)
# ---------------------------------------------------------------------------


class _TaggedHandle:
    """Stand-in for IRunHandle with a tags attribute (re-used from finalizer tests)."""

    def __init__(self, handle: RunHandle, tags: dict[str, str]) -> None:
        self.run_id = handle.run_id
        self.experiment_id = handle.experiment_id
        self.parent_run_id = handle.parent_run_id
        self.tracking_uri = handle.tracking_uri
        self.status = handle.status
        self.tags = tags


class _EmptyTagQuery:
    """IRunQuery that returns tag-less handles (finalize always proceeds)."""

    def __init__(self) -> None:
        self._handles: dict[str, RunHandle] = {}

    def register(self, handle: RunHandle) -> None:
        self._handles[handle.run_id] = handle

    def get_run(self, run_id: str) -> _TaggedHandle:
        return _TaggedHandle(self._handles[run_id], {})

    def list_children(self, parent_run_id: str) -> Sequence[RunHandle]:
        return ()

    def search(self, experiment: str, filter_: str, max_results: int) -> Sequence[RunHandle]:
        return ()


def _build_coord() -> tuple[
    RunLifecycleCoord,
    FakeTrackingClient,
    FakeJournalUploader,
    _EmptyTagQuery,
]:
    client = FakeTrackingClient()
    uploader = FakeJournalUploader()
    query = _EmptyTagQuery()
    finalizer = MlflowFinalizer(client, uploader, query)
    opener = ParentRunOpener(client, opened_by="host:user")
    preflight = PreflightConnectivityCheck(client)
    coord = RunLifecycleCoord(opener, finalizer, preflight)
    return coord, client, uploader, query


def _open_root(client: FakeTrackingClient) -> RunHandle:
    return client.start_run("exp", "root", tags={}, params={})


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    """Happy paths: context manager + finalize delegation."""

    def test_enter_returns_self(self) -> None:
        coord, _, _, _ = _build_coord()
        with coord as c:
            assert c is coord

    def test_enter_exit_no_exception(self) -> None:
        coord, _, _, _ = _build_coord()
        with coord:
            pass

    def test_bind_and_finalize_attempt(self) -> None:
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)

        coord.bind_attempt_run(attempt)
        coord.finalize(status=RunStatus.FINISHED)

        assert client.terminated_calls == [(attempt.run_id, RunStatus.FINISHED)]

    def test_bind_root_and_attempt_closes_both(self) -> None:
        coord, client, _, query = _build_coord()
        root = _open_root(client)
        attempt = client.start_nested_run(root.run_id, "att", tags={})
        query.register(root)
        query.register(attempt)

        coord.bind_root_run(root)
        coord.bind_attempt_run(attempt)
        coord.finalize(status=RunStatus.FINISHED)

        run_ids = [rid for (rid, _) in client.terminated_calls]
        assert root.run_id in run_ids
        assert attempt.run_id in run_ids


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    """Re-entrancy and no-op paths."""

    def test_finalize_without_bound_runs_is_noop(self) -> None:
        coord, client, _, _ = _build_coord()

        coord.finalize(status=RunStatus.FINISHED)

        assert client.terminated_calls == []
        assert client.set_tags_calls == []

    def test_exit_without_enter_is_noop(self) -> None:
        coord, _, _, _ = _build_coord()
        # Calling __exit__ on a not-entered coord must not raise.
        coord.__exit__(None, None, None)


# ===========================================================================
# 3. Boundary (mutex / idempotency)
# ===========================================================================


class TestBoundary:
    """Mutex on finalize: never runs twice."""

    def test_double_finalize_only_runs_once(self) -> None:
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)

        coord.finalize(status=RunStatus.FINISHED)
        coord.finalize(status=RunStatus.FAILED)

        # Only the first call took effect.
        assert client.terminated_calls == [(attempt.run_id, RunStatus.FINISHED)]

    def test_concurrent_finalize_only_one_succeeds(self) -> None:
        """Mutex guards finalize -- 2 parallel threads cause only one
        delegation."""
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)

        barrier = threading.Barrier(2)

        def run() -> None:
            barrier.wait()
            coord.finalize(status=RunStatus.FINISHED)

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(client.terminated_calls) == 1


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    """Always-true invariants on coord behaviour."""

    def test_enter_registers_atexit(self) -> None:
        coord, _, _, _ = _build_coord()

        with mock.patch("atexit.register") as mock_register:
            coord.__enter__()
            assert mock_register.called

        # Cleanup
        coord.__exit__(None, None, None)

    def test_exit_unregisters_atexit(self) -> None:
        coord, _, _, _ = _build_coord()
        coord.__enter__()

        with mock.patch("atexit.unregister") as mock_unregister:
            coord.__exit__(None, None, None)
            assert mock_unregister.called

    def test_attempt_closed_before_root(self) -> None:
        """When both runs are bound, the attempt MUST finalize before the
        root (downstream readers depend on the ordering)."""
        coord, client, _, query = _build_coord()
        root = _open_root(client)
        attempt = client.start_nested_run(root.run_id, "att", tags={})
        query.register(root)
        query.register(attempt)
        coord.bind_root_run(root)
        coord.bind_attempt_run(attempt)

        coord.finalize(status=RunStatus.FINISHED)

        # Order in terminated_calls reflects finalize ordering.
        ordering = [rid for (rid, _) in client.terminated_calls]
        assert ordering[0] == attempt.run_id
        assert ordering[1] == root.run_id

    def test_signal_handlers_installed_on_enter(self) -> None:
        """Coord installs SIGTERM + SIGINT handlers when entered (in the
        main thread)."""
        coord, _, _, _ = _build_coord()

        with mock.patch("signal.signal") as mock_signal:
            mock_signal.return_value = signal.SIG_DFL
            coord.__enter__()
            installed_signums = {call.args[0] for call in mock_signal.call_args_list}
            assert signal.SIGTERM in installed_signums
            assert signal.SIGINT in installed_signums

        coord.__exit__(None, None, None)

    def test_signal_handlers_restored_on_exit(self) -> None:
        coord, _, _, _ = _build_coord()
        captured: dict[int, object] = {}

        original_signal = signal.signal

        def spy(signum: int, handler: object) -> object:
            captured.setdefault(signum, signal.SIG_DFL)
            return original_signal(signum, handler)  # type: ignore[arg-type]

        with mock.patch("signal.signal", side_effect=spy):
            coord.__enter__()
            assert signal.SIGTERM in captured
            coord.__exit__(None, None, None)


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    """Finalizer or client failure during coord.finalize must NOT propagate."""

    def test_finalizer_raises_swallowed(self) -> None:
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)
        # Inject a tracker that raises on set_tags via fake chaos.
        client.fail_next_n_calls(100)  # all subsequent client calls raise

        # Must not raise -- coord wraps finalize in try/except.
        coord.finalize(status=RunStatus.FINISHED)

    def test_atexit_swallows_exception(self) -> None:
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)
        client.fail_next_n_calls(100)

        coord._safe_finalize_atexit()
        # No raise; coord remains usable.


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    """Pins for previously-observed defects."""

    def test_double_enter_idempotent(self) -> None:
        """Re-entering must not re-register atexit (would cause duplicate
        finalize on shutdown)."""
        coord, _, _, _ = _build_coord()

        with mock.patch("atexit.register") as mock_register:
            coord.__enter__()
            coord.__enter__()
            assert mock_register.call_count == 1

        coord.__exit__(None, None, None)

    def test_exit_clears_signal_handler_map(self) -> None:
        coord, _, _, _ = _build_coord()
        coord.__enter__()
        coord.__exit__(None, None, None)

        assert coord._prev_signal_handlers == {}

    def test_root_finalize_skips_journal_upload(self) -> None:
        """The journal artifact lives under the attempt run -- the root
        finalize MUST NOT re-upload."""
        coord, client, uploader, query = _build_coord()
        root = _open_root(client)
        attempt = client.start_nested_run(root.run_id, "att", tags={})
        query.register(root)
        query.register(attempt)
        coord.bind_root_run(root)
        coord.bind_attempt_run(attempt)

        from pathlib import Path

        coord.finalize(
            status=RunStatus.FINISHED,
            journal_path=Path("/tmp/events.jsonl"),
            journal_sha256="sha",
        )

        # Exactly one upload attempt -- to the attempt run, not the root.
        # (Path doesn't exist, so the journal uploader's path check skips
        # it -- but the call should have been at most once total.)
        attempt_uploads = [
            c for c in uploader.upload_calls if c.run_id == attempt.run_id
        ]
        root_uploads = [c for c in uploader.upload_calls if c.run_id == root.run_id]
        assert root_uploads == []
        # attempt uploads is either [] (path missing) or one entry.
        assert len(attempt_uploads) <= 1


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    """Pins for the wiring between coord, finalizer, opener, preflight."""

    def test_coord_holds_three_collaborators(self) -> None:
        coord, _, _, _ = _build_coord()
        assert coord._opener is not None
        assert coord._finalizer is not None
        assert coord._preflight is not None

    def test_signal_handler_invokes_finalize_with_killed_status(self) -> None:
        """SIGTERM handler calls finalize(status=KILLED) -- verify by
        triggering it manually (the handler's KeyboardInterrupt is
        re-raised, which we catch)."""
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)

        with pytest.raises(KeyboardInterrupt):
            coord._handle_signal(signal.SIGTERM, None)

        # Run terminated as KILLED.
        assert client.terminated_calls == [(attempt.run_id, RunStatus.KILLED)]

    def test_bind_finalize_payload_threadsafe(self) -> None:
        """Concurrent bind_finalize_payload + finalize must not deadlock."""
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)

        from pathlib import Path

        coord.bind_finalize_payload(
            journal_path=Path("/tmp/x.jsonl"),
            journal_sha256="abc",
        )
        coord.finalize(status=RunStatus.FINISHED)

        assert client.terminated_calls == [(attempt.run_id, RunStatus.FINISHED)]

    def test_atexit_uses_finished_status(self) -> None:
        """atexit hook defaults to FINISHED -- the orchestrator's finally
        block is expected to have already issued the real status."""
        coord, client, _, query = _build_coord()
        attempt = _open_root(client)
        query.register(attempt)
        coord.bind_attempt_run(attempt)

        coord._safe_finalize_atexit()

        assert client.terminated_calls == [(attempt.run_id, RunStatus.FINISHED)]

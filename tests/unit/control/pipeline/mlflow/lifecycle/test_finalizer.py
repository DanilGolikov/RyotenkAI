"""Unit tests: ``MlflowFinalizer``.

7-class structure (positive / negative / boundary / invariants /
dependency-errors / regressions / logic-specific).

Coverage:

* Happy path - tags + terminate in the right order.
* Idempotency - second call on a finalized run is a no-op.
* Journal upload triggered only when both path + sha256 provided.
* Never raises - exception in any sub-step is swallowed.
* Skip path - journal upload skipped when payload missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ryotenkai_control.pipeline.mlflow.lifecycle.finalizer import MlflowFinalizer
from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from ryotenkai_shared.infrastructure.mlflow.taxonomy import TagKey
from tests._fakes.mlflow_journal_uploader import FakeJournalUploader
from tests._fakes.mlflow_run_query import FakeRunQuery
from tests._fakes.mlflow_tracking_client import FakeTrackingClient


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _TaggedHandle:
    """Test-only IRunHandle stand-in with a ``tags`` attribute.

    The production :class:`RunHandle` is a frozen dataclass and does not
    carry a tags dict. For idempotency tests we wrap it in this thin
    object so the finalizer's ``getattr(handle, "tags", None)`` branch
    fires correctly. The Protocol surface is implicitly Protocol-shaped
    because the finalizer only reads the ``tags`` attribute.
    """

    def __init__(self, handle: RunHandle, tags: dict[str, str]) -> None:
        self.run_id = handle.run_id
        self.experiment_id = handle.experiment_id
        self.parent_run_id = handle.parent_run_id
        self.tracking_uri = handle.tracking_uri
        self.status = handle.status
        self.tags = tags


class _TaggedRunQuery:
    """Test-only IRunQuery implementation that returns tag-bearing handles.

    Used by idempotency tests to simulate the production
    ``MlflowReadClient`` which (post-M3) returns a tag-aware handle.
    """

    def __init__(self, *, tags_by_run_id: dict[str, dict[str, str]] | None = None) -> None:
        self._tags_by_run_id: dict[str, dict[str, str]] = tags_by_run_id or {}
        self._handles: dict[str, RunHandle] = {}
        self.get_run_calls: list[str] = []
        self._fail_next = 0

    def register(self, handle: RunHandle, tags: dict[str, str] | None = None) -> None:
        self._handles[handle.run_id] = handle
        self._tags_by_run_id[handle.run_id] = tags or {}

    def set_tags(self, run_id: str, tags: dict[str, str]) -> None:
        self._tags_by_run_id.setdefault(run_id, {}).update(tags)

    def fail_next_get_run(self, n: int = 1) -> None:
        self._fail_next = n

    def get_run(self, run_id: str) -> _TaggedHandle:  # type: ignore[override]
        self.get_run_calls.append(run_id)
        if self._fail_next > 0:
            self._fail_next -= 1
            raise RuntimeError("simulated get_run failure")
        handle = self._handles[run_id]
        tags = dict(self._tags_by_run_id.get(run_id, {}))
        return _TaggedHandle(handle, tags)

    def list_children(self, parent_run_id: str) -> Sequence[RunHandle]:
        return tuple(
            h for h in self._handles.values() if h.parent_run_id == parent_run_id
        )

    def search(
        self, experiment: str, filter_: str, max_results: int
    ) -> Sequence[RunHandle]:
        return ()


def _open_root(client: FakeTrackingClient) -> RunHandle:
    """Open a real RunHandle via FakeTrackingClient and return it."""
    return client.start_run("exp", "run-1", tags={}, params={})


def _make_finalizer(
    client: FakeTrackingClient | None = None,
    uploader: FakeJournalUploader | None = None,
    query: _TaggedRunQuery | FakeRunQuery | None = None,
) -> tuple[MlflowFinalizer, FakeTrackingClient, FakeJournalUploader, _TaggedRunQuery]:
    """Construct a finalizer pre-wired with empty-tag run query."""
    client = client or FakeTrackingClient()
    uploader = uploader or FakeJournalUploader()
    query = query or _TaggedRunQuery()
    finalizer = MlflowFinalizer(client, uploader, query)
    return finalizer, client, uploader, query  # type: ignore[return-value]


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    """Happy paths: tags set, journal uploaded, run terminated."""

    def test_finalize_sets_lifecycle_tags(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        tags = client.get_tags(handle.run_id)
        assert tags[TagKey.LIFECYCLE_FINALIZED.value] == "true"
        assert tags[TagKey.LIFECYCLE_STATUS.value] == "FINISHED"

    def test_finalize_terminates_run(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        assert client.terminated_calls == [(handle.run_id, RunStatus.FINISHED)]

    def test_finalize_uploads_journal_when_provided(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        journal = tmp_path / "events.jsonl"
        journal.write_text("seed", encoding="utf-8")

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=journal,
            journal_sha256="deadbeef",
        )

        assert len(uploader.upload_calls) == 1
        assert uploader.upload_calls[0].run_id == handle.run_id
        assert uploader.upload_calls[0].sha256 == "deadbeef"

    def test_finalize_killed_status_recorded(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.KILLED,
            journal_path=None,
            journal_sha256=None,
            exit_reason="signal:15",
        )

        tags = client.get_tags(handle.run_id)
        assert tags[TagKey.LIFECYCLE_STATUS.value] == "KILLED"
        assert tags[TagKey.EXIT_REASON.value] == "signal:15"


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    """Failed status path; failures swallowed silently."""

    def test_finalize_failed_status_recorded(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FAILED,
            journal_path=None,
            journal_sha256=None,
            exit_reason="oom",
        )

        tags = client.get_tags(handle.run_id)
        assert tags[TagKey.LIFECYCLE_STATUS.value] == "FAILED"
        assert tags[TagKey.EXIT_REASON.value] == "oom"

    def test_exit_reason_none_renders_empty_string(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
            exit_reason=None,
        )

        tags = client.get_tags(handle.run_id)
        assert tags[TagKey.EXIT_REASON.value] == ""


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    """Edge cases around journal upload triggers."""

    def test_journal_skipped_when_only_path(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        journal = tmp_path / "events.jsonl"
        journal.write_text("x", encoding="utf-8")

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=journal,
            journal_sha256=None,
        )

        # No upload without sha256.
        assert uploader.upload_calls == []

    def test_journal_skipped_when_only_sha256(self) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256="sha",
        )

        assert uploader.upload_calls == []

    def test_journal_skipped_when_empty_sha256(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        journal = tmp_path / "x.jsonl"
        journal.write_text("", encoding="utf-8")

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=journal,
            journal_sha256="",
        )

        assert uploader.upload_calls == []


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    """Idempotency: never finalize the same run twice."""

    def test_idempotent_skip_when_already_finalized(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        # Pre-seed the run as already finalized in the query.
        query.register(
            handle,
            tags={TagKey.LIFECYCLE_FINALIZED.value: "true"},
        )

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        # No set_tags / terminated calls -- the run was already done.
        assert client.set_tags_calls == []
        assert client.terminated_calls == []

    def test_idempotent_does_not_re_upload_journal(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(
            handle,
            tags={TagKey.LIFECYCLE_FINALIZED.value: "true"},
        )
        journal = tmp_path / "events.jsonl"
        journal.write_text("data", encoding="utf-8")

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=journal,
            journal_sha256="abc",
        )

        assert uploader.upload_calls == []

    def test_step_order_tags_then_terminate(self) -> None:
        """Tags MUST be stamped before set_terminated -- otherwise a reader
        polling for FINISHED status would observe the run as terminated
        without lifecycle.finalized=true, and re-issue a redundant close.
        """
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        # Both lists ordered; tags was first.
        assert len(client.set_tags_calls) == 1
        assert len(client.terminated_calls) == 1

    def test_finalize_calls_get_run_for_idempotency(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        assert query.get_run_calls == [handle.run_id]


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    """Failures in any sub-step must be swallowed."""

    def test_set_tags_failure_swallowed(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        client.fail_next_n_calls(1)  # set_tags will raise

        # Must not propagate.
        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        # set_terminated still ran (best-effort).
        assert client.terminated_calls == [(handle.run_id, RunStatus.FINISHED)]

    def test_set_terminated_failure_swallowed(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        # Tags succeed (1st mutating call), terminate fails (2nd).
        client.fail_next_n_calls(2)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )
        # Nothing raised; we are still alive.

    def test_journal_failure_swallowed(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        journal = tmp_path / "events.jsonl"
        journal.write_text("x", encoding="utf-8")
        uploader.fail_next_n_calls(1)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=journal,
            journal_sha256="sha",
        )

        # Tags + terminate still applied even though journal upload failed.
        assert len(client.set_tags_calls) == 1
        assert client.terminated_calls == [(handle.run_id, RunStatus.FINISHED)]

    def test_query_get_run_failure_treated_as_not_finalized(self) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        query.fail_next_get_run(1)

        # Must not raise; behaviour: proceed with finalize.
        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        assert len(client.set_tags_calls) == 1
        assert client.terminated_calls == [(handle.run_id, RunStatus.FINISHED)]


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    """Pin behaviours that previously broke."""

    def test_double_finalize_is_safe(self) -> None:
        """A second finalize call on the same handle must be safe -- the
        legacy code re-terminated the run and corrupted the status tag."""
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )
        # Simulate the server-side tag being persisted (production
        # behaviour). The test query is not auto-synced with the client.
        query.set_tags(handle.run_id, {TagKey.LIFECYCLE_FINALIZED.value: "true"})

        finalizer.finalize(
            run=handle,
            status=RunStatus.FAILED,  # try to "downgrade" status
            journal_path=None,
            journal_sha256=None,
        )

        # Second call did not stamp FAILED status.
        assert client.terminated_calls == [(handle.run_id, RunStatus.FINISHED)]
        # set_tags called exactly once (first time only).
        assert len(client.set_tags_calls) == 1

    def test_lifecycle_finalized_tag_value_is_true_string(self) -> None:
        """MLflow tags are string-typed. The finalized marker is the
        literal string 'true' -- a tag value of 'True' (capital) would
        not match the idempotency check.
        """
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=None,
            journal_sha256=None,
        )

        captured = client.set_tags_calls[0].tags
        assert captured[TagKey.LIFECYCLE_FINALIZED.value] == "true"


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    """Pin the specific tag values written on close."""

    @pytest.mark.parametrize(
        "status,expected_str",
        [
            (RunStatus.FINISHED, "FINISHED"),
            (RunStatus.FAILED, "FAILED"),
            (RunStatus.KILLED, "KILLED"),
        ],
    )
    def test_status_stamped_as_enum_value(
        self,
        status: RunStatus,
        expected_str: str,
    ) -> None:
        finalizer, client, _, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)

        finalizer.finalize(
            run=handle,
            status=status,
            journal_path=None,
            journal_sha256=None,
        )

        captured = client.set_tags_calls[0].tags
        assert captured[TagKey.LIFECYCLE_STATUS.value] == expected_str

    def test_journal_upload_args_match_input(self, tmp_path: Path) -> None:
        finalizer, client, uploader, query = _make_finalizer()
        handle = _open_root(client)
        query.register(handle)
        journal = tmp_path / "e.jsonl"
        journal.write_text("xx", encoding="utf-8")

        finalizer.finalize(
            run=handle,
            status=RunStatus.FINISHED,
            journal_path=journal,
            journal_sha256="0123abcd",
        )

        call = uploader.upload_calls[0]
        assert call.run_id == handle.run_id
        assert call.journal_path == journal
        assert call.sha256 == "0123abcd"

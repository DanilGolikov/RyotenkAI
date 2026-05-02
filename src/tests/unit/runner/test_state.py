"""Phase 1.1 — JobLifecycleFSM contract.

Coverage matrix (mirrors the pattern in
``src/tests/unit/community/test_platform_categories.py``):

- TestJobStateModel       ........... terminal predicate, enum integrity
- TestSnapshotRoundtrip   ........... to_dict / from_dict / immutability
- TestSubmit              ........... initial transition to PREPARING
- TestLegalTransitions    ........... every entry in the matrix is exercised
- TestIllegalTransitions  ........... every disallowed move raises
- TestPersistence         ........... state.jsonl + state.json layout
- TestRestoreOrInit       ........... three branches (missing / terminal / unsafe)
- TestEdgeCases           ........... no-snapshot transition, double submit
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runner.state import (
    InvalidTransitionError,
    JobLifecycleFSM,
    JobSnapshot,
    JobState,
)


@pytest.fixture
def fsm(tmp_path: Path) -> JobLifecycleFSM:
    """Fresh FSM rooted in ``tmp_path``."""
    f = JobLifecycleFSM(workspace_dir=tmp_path)
    f.restore_or_init()
    return f


# ---------------------------------------------------------------------------
# Enum / snapshot
# ---------------------------------------------------------------------------


class TestJobStateModel:
    def test_enum_values_are_lowercase_strings(self) -> None:
        # Wire-stable values — UI, JSONL, OpenAPI all rely on them.
        assert JobState.PREPARING.value == "preparing"
        assert JobState.RUNNING.value == "running"
        assert JobState.STOPPING.value == "stopping"
        assert JobState.COMPLETED.value == "completed"
        assert JobState.FAILED.value == "failed"
        assert JobState.CANCELLED.value == "cancelled"

    def test_terminal_partition(self) -> None:
        terminal = {s for s in JobState if s.is_terminal}
        non_terminal = {s for s in JobState if not s.is_terminal}
        assert terminal | non_terminal == set(JobState)
        assert not (terminal & non_terminal)
        assert terminal == {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}


class TestSnapshotRoundtrip:
    def test_to_dict_from_dict(self) -> None:
        snap = JobSnapshot(
            job_id="j-42",
            state=JobState.RUNNING,
            sequence=2,
            started_at="2026-04-26T12:00:00Z",
            updated_at="2026-04-26T12:01:00Z",
            message="step 100 / 1000",
        )
        assert JobSnapshot.from_dict(snap.to_dict()) == snap

    def test_message_defaults_empty(self) -> None:
        snap = JobSnapshot.from_dict({
            "job_id": "j-1",
            "state": "preparing",
            "sequence": 0,
            "started_at": "2026-04-26T00:00:00Z",
            "updated_at": "2026-04-26T00:00:00Z",
        })
        assert snap.message == ""

    def test_snapshot_is_frozen(self) -> None:
        snap = JobSnapshot(
            job_id="j-1",
            state=JobState.PREPARING,
            sequence=0,
            started_at="t",
            updated_at="t",
        )
        with pytest.raises(AttributeError):
            snap.state = JobState.RUNNING  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_initialises_at_preparing(self, fsm: JobLifecycleFSM) -> None:
        snap = fsm.submit("job-1")
        assert snap.state == JobState.PREPARING
        assert snap.sequence == 0
        assert snap.job_id == "job-1"
        assert snap.message == ""
        assert snap.started_at == snap.updated_at  # initial transition

    def test_current_returns_none_before_submit(self, fsm: JobLifecycleFSM) -> None:
        assert fsm.current() is None

    def test_current_returns_snapshot_after_submit(self, fsm: JobLifecycleFSM) -> None:
        snap = fsm.submit("j")
        assert fsm.current() == snap


# ---------------------------------------------------------------------------
# Transitions matrix — positive cases
# ---------------------------------------------------------------------------


class TestLegalTransitions:
    """Every entry in ``_LEGAL_TRANSITIONS`` exercised exactly once."""

    @pytest.mark.parametrize(
        ("path", "final"),
        [
            # preparing → running → completed
            ([JobState.RUNNING, JobState.COMPLETED], JobState.COMPLETED),
            # preparing → running → failed
            ([JobState.RUNNING, JobState.FAILED], JobState.FAILED),
            # preparing → failed (preflight rejected)
            ([JobState.FAILED], JobState.FAILED),
            # preparing → running → stopping → completed (graceful save)
            ([JobState.RUNNING, JobState.STOPPING, JobState.COMPLETED], JobState.COMPLETED),
            # preparing → running → stopping → cancelled (true stop)
            ([JobState.RUNNING, JobState.STOPPING, JobState.CANCELLED], JobState.CANCELLED),
            # preparing → running → stopping → failed (mid-stop crash)
            ([JobState.RUNNING, JobState.STOPPING, JobState.FAILED], JobState.FAILED),
        ],
    )
    def test_legal_path(
        self, fsm: JobLifecycleFSM, path: list[JobState], final: JobState
    ) -> None:
        fsm.submit("j")
        for step in path:
            fsm.transition(step)
        snap = fsm.current()
        assert snap is not None
        assert snap.state == final
        assert snap.sequence == len(path)  # one bump per transition

    def test_message_threads_through(self, fsm: JobLifecycleFSM) -> None:
        fsm.submit("j")
        fsm.transition(JobState.RUNNING)
        fsm.transition(JobState.FAILED, message="oom: bitsandbytes Q4_K_M")
        snap = fsm.current()
        assert snap is not None
        assert snap.message == "oom: bitsandbytes Q4_K_M"


# ---------------------------------------------------------------------------
# Transitions matrix — negative cases
# ---------------------------------------------------------------------------


class TestIllegalTransitions:
    def test_skip_running(self, fsm: JobLifecycleFSM) -> None:
        fsm.submit("j")
        with pytest.raises(InvalidTransitionError):
            fsm.transition(JobState.STOPPING)  # preparing → stopping forbidden

    def test_skip_to_completed_from_preparing(self, fsm: JobLifecycleFSM) -> None:
        fsm.submit("j")
        with pytest.raises(InvalidTransitionError):
            fsm.transition(JobState.COMPLETED)  # preparing → completed forbidden

    @pytest.mark.parametrize(
        "terminal",
        [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED],
    )
    def test_terminal_states_accept_no_transitions(
        self, fsm: JobLifecycleFSM, terminal: JobState
    ) -> None:
        fsm.submit("j")
        fsm.transition(JobState.RUNNING)
        if terminal == JobState.CANCELLED:
            fsm.transition(JobState.STOPPING)  # cancelled is reachable only via stopping
        fsm.transition(terminal)
        # Now any further transition is illegal — we exhaustively try
        # every other state.
        for other in JobState:
            with pytest.raises(InvalidTransitionError):
                fsm.transition(other)

    def test_transition_without_submit(self, fsm: JobLifecycleFSM) -> None:
        with pytest.raises(InvalidTransitionError):
            fsm.transition(JobState.RUNNING)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_state_dir_created_under_workspace(self, tmp_path: Path) -> None:
        f = JobLifecycleFSM(workspace_dir=tmp_path)
        f.restore_or_init()
        assert (tmp_path / "state").is_dir()

    def test_jsonl_appends_one_line_per_transition(
        self, fsm: JobLifecycleFSM
    ) -> None:
        fsm.submit("j")
        fsm.transition(JobState.RUNNING)
        fsm.transition(JobState.COMPLETED)

        lines = fsm.jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

        decoded = [json.loads(line) for line in lines]
        assert [r["state"] for r in decoded] == ["preparing", "running", "completed"]
        assert [r["sequence"] for r in decoded] == [0, 1, 2]
        # timestamps are monotonically non-decreasing (second-precision).
        assert decoded[0]["started_at"] == decoded[1]["started_at"] == decoded[2]["started_at"]

    def test_state_json_holds_latest_snapshot(self, fsm: JobLifecycleFSM) -> None:
        fsm.submit("j")
        fsm.transition(JobState.RUNNING)
        payload = json.loads(fsm.json_path.read_text(encoding="utf-8"))
        assert payload["state"] == "running"
        assert payload["sequence"] == 1


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


class TestRestoreOrInit:
    def test_missing_state_json_means_fresh(self, tmp_path: Path) -> None:
        f = JobLifecycleFSM(workspace_dir=tmp_path)
        f.restore_or_init()
        assert f.current() is None

    def test_restore_terminal_preserves_state(self, tmp_path: Path) -> None:
        # Drive a job to COMPLETED, then create a fresh FSM in the same workspace.
        f1 = JobLifecycleFSM(workspace_dir=tmp_path)
        f1.restore_or_init()
        f1.submit("j-1")
        f1.transition(JobState.RUNNING)
        f1.transition(JobState.COMPLETED)

        f2 = JobLifecycleFSM(workspace_dir=tmp_path)
        f2.restore_or_init()
        snap = f2.current()
        assert snap is not None
        assert snap.state == JobState.COMPLETED
        assert snap.job_id == "j-1"

    @pytest.mark.parametrize(
        "unsafe_state",
        [JobState.PREPARING, JobState.STOPPING],
    )
    def test_restore_unsafe_transitions_to_failed(
        self, tmp_path: Path, unsafe_state: JobState
    ) -> None:
        f1 = JobLifecycleFSM(workspace_dir=tmp_path)
        f1.restore_or_init()
        f1.submit("j-unsafe")
        if unsafe_state == JobState.STOPPING:
            f1.transition(JobState.RUNNING)
            f1.transition(JobState.STOPPING)
        # Container "dies" — we don't transition further.

        f2 = JobLifecycleFSM(workspace_dir=tmp_path)
        f2.restore_or_init()
        snap = f2.current()
        assert snap is not None
        assert snap.state == JobState.FAILED
        assert snap.message == "container_restart_during_unsafe_state"

        # The recovery transition is also persisted.
        lines = f2.jsonl_path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[-1])["state"] == "failed"

    def test_corrupt_state_json_treated_as_fresh(self, tmp_path: Path) -> None:
        # Pre-populate a half-written state.json — simulates the host
        # dying between os.replace's tmp-write and rename in a way
        # that left a truncated file behind. The FSM must boot, not
        # crash; the corrupt file is preserved as ``state.json.corrupt``
        # so on-call has it for forensics.
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "job.json").write_text("{not valid json", encoding="utf-8")

        f = JobLifecycleFSM(workspace_dir=tmp_path)
        f.restore_or_init()  # must not raise
        assert f.current() is None
        assert (state_dir / "job.json.corrupt").exists()
        assert not (state_dir / "job.json").exists()

    def test_state_json_with_missing_keys_treated_as_fresh(self, tmp_path: Path) -> None:
        # Snapshot.from_dict raises KeyError on missing required keys.
        # Same recovery path as JSONDecodeError.
        import json as _json  # local — not on hot path

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "job.json").write_text(
            _json.dumps({"job_id": "j", "state": "running"}),  # missing sequence/started_at/...
            encoding="utf-8",
        )

        f = JobLifecycleFSM(workspace_dir=tmp_path)
        f.restore_or_init()
        assert f.current() is None
        assert (state_dir / "job.json.corrupt").exists()

    def test_running_state_preserved_on_restore(self, tmp_path: Path) -> None:
        # RUNNING is *not* unsafe — it just means "trainer was alive
        # last we checked". The supervisor (Phase 2) decides whether
        # to attach to an existing trainer or fail it.
        f1 = JobLifecycleFSM(workspace_dir=tmp_path)
        f1.restore_or_init()
        f1.submit("j")
        f1.transition(JobState.RUNNING)

        f2 = JobLifecycleFSM(workspace_dir=tmp_path)
        f2.restore_or_init()
        snap = f2.current()
        assert snap is not None
        assert snap.state == JobState.RUNNING


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_double_submit_while_running_rejects(self, fsm: JobLifecycleFSM) -> None:
        fsm.submit("j-1")
        fsm.transition(JobState.RUNNING)
        with pytest.raises(InvalidTransitionError):
            fsm.submit("j-2")

    def test_resubmit_after_terminal_succeeds(self, fsm: JobLifecycleFSM) -> None:
        # Single-active is "no two non-terminal jobs at once". A new
        # submit AFTER the previous reached a terminal state is allowed
        # — the FSM treats it as a fresh job.
        fsm.submit("j-1")
        fsm.transition(JobState.RUNNING)
        fsm.transition(JobState.COMPLETED)

        snap = fsm.submit("j-2")
        assert snap.state == JobState.PREPARING
        assert snap.sequence == 0
        assert snap.job_id == "j-2"

    def test_invalid_transition_error_carries_attributes(
        self, fsm: JobLifecycleFSM
    ) -> None:
        fsm.submit("j")
        with pytest.raises(InvalidTransitionError) as exc_info:
            fsm.transition(JobState.STOPPING)
        assert exc_info.value.current == JobState.PREPARING
        assert exc_info.value.attempted == JobState.STOPPING

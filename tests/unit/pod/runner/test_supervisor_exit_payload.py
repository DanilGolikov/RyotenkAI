"""Phase D — :class:`Supervisor` reap-time exit-payload handling.

Covers the helpers that read ``<workdir>/trainer-exit.json`` and the
event-builder that produces the ``trainer_exited`` body (now bumped
to ``schema_version=2``):

* :func:`_read_exit_payload` — file-absent → ``None``, valid file →
  parsed :class:`TrainerExitPayload`, malformed JSON → ``None`` with
  WARN log (no crash).
* :func:`_build_trainer_exited_event` — three branches: typed payload
  (``payload_source="trainer_file"``), SIGKILL heuristic
  (``payload_source="sigkill_heuristic"``), legacy fallback
  (``payload_source="missing"`` or ``"none"``).

We additionally pin one end-to-end behaviour via the real
:class:`Supervisor` + a fake trainer that writes a payload, to make
sure the wiring between ``_reap`` and the helpers stays connected.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.state import JobLifecycleFSM, JobState
from ryotenkai_pod.runner.supervisor import (
    TRAINER_EXITED_EVENT_SCHEMA_VERSION,
    Supervisor,
    _build_trainer_exited_event,
    _read_exit_payload,
)
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.trainer_exit import (
    TRAINER_EXIT_FILENAME,
    TrainerExitPayload,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Class 1: _read_exit_payload
# ---------------------------------------------------------------------------


class TestReadExitPayload:
    """``_read_exit_payload`` reads the trainer-exit.json safely."""

    def test_returns_none_when_workdir_is_none(self) -> None:
        assert _read_exit_payload(None) is None

    def test_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        assert _read_exit_payload(tmp_path) is None

    def test_returns_payload_on_valid_file(self, tmp_path: Path) -> None:
        original = TrainerExitPayload(
            code=ErrorCode.TRAINING_OOM,
            message="cgroup OOM",
            exit_code=137,
            wall_seconds=8.0,
        )
        original.write_to(tmp_path / TRAINER_EXIT_FILENAME)
        result = _read_exit_payload(tmp_path)
        assert result == original

    def test_returns_none_on_malformed_json(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Malformed JSON degrades gracefully (no crash, WARN log)."""
        (tmp_path / TRAINER_EXIT_FILENAME).write_text("{ broken", encoding="utf-8")
        with caplog.at_level("WARNING"):
            result = _read_exit_payload(tmp_path)
        assert result is None
        # The WARN log helps operators notice trainer<->supervisor drift.
        assert any(
            "trainer-exit.json" in rec.message
            for rec in caplog.records
            if rec.levelname == "WARNING"
        )

    def test_returns_none_on_extra_keys(self, tmp_path: Path) -> None:
        """``extra="forbid"`` — a newer schema's extra fields fail
        validation cleanly rather than half-decoding."""
        target = tmp_path / TRAINER_EXIT_FILENAME
        target.write_text(
            json.dumps({
                "code": "TRAINING_FAILED",
                "message": "x",
                "exit_code": 1,
                "wall_seconds": 0.0,
                "schema_version": 1,
                "from_the_future": True,
            }),
            encoding="utf-8",
        )
        assert _read_exit_payload(tmp_path) is None


# ---------------------------------------------------------------------------
# Class 2: _build_trainer_exited_event — clean exit
# ---------------------------------------------------------------------------


class TestBuildEventCleanExit:
    """rc==0 + no payload → legacy fields only, ``code``/``message`` null."""

    def test_natural_success_event_shape(self) -> None:
        event = _build_trainer_exited_event(
            rc=0, signal_name=None, cancellation_requested=False,
            payload=None, synthesize_if_missing=False,
        )
        assert event["exit_code"] == 0
        assert event["signal"] is None
        assert event["cancellation_requested"] is False
        assert event["code"] is None
        assert event["message"] is None
        assert event["traceback_summary"] is None
        assert event["payload_source"] == "none"
        assert event["schema_version"] == TRAINER_EXITED_EVENT_SCHEMA_VERSION

    def test_cancellation_does_not_synthesise(self) -> None:
        """A cancelled trainer (rc!=0, cancel=True) should NOT synthesise
        a TRAINING_OOM — the caller passes ``synthesize_if_missing=False``
        on cancellation."""
        event = _build_trainer_exited_event(
            rc=-15, signal_name="SIGTERM", cancellation_requested=True,
            payload=None, synthesize_if_missing=False,
        )
        assert event["code"] is None
        assert event["payload_source"] == "none"


# ---------------------------------------------------------------------------
# Class 3: _build_trainer_exited_event — typed payload branch
# ---------------------------------------------------------------------------


class TestBuildEventTypedPayload:
    """Trainer wrote a valid payload → carry its typed fields."""

    def test_typed_payload_propagates_code(self) -> None:
        payload = TrainerExitPayload(
            code=ErrorCode.TRAINING_OOM,
            message="OOM in step 100",
            traceback_summary="frame 1\nframe 2",
            exit_code=1,
            wall_seconds=42.0,
        )
        event = _build_trainer_exited_event(
            rc=1, signal_name=None, cancellation_requested=False,
            payload=payload, synthesize_if_missing=False,
        )
        assert event["code"] == ErrorCode.TRAINING_OOM.value
        assert event["message"] == "OOM in step 100"
        assert event["traceback_summary"] == "frame 1\nframe 2"
        assert event["wall_seconds"] == 42.0
        assert event["payload_source"] == "trainer_file"
        assert event["schema_version"] == TRAINER_EXITED_EVENT_SCHEMA_VERSION

    def test_typed_payload_preserves_exit_code_from_supervisor(self) -> None:
        """The supervisor's ``rc`` is the source of truth; the trainer's
        ``exit_code`` field is informational. Both land in the event
        but the top-level ``exit_code`` is the supervisor's."""
        payload = TrainerExitPayload(
            code=ErrorCode.TRAINING_FAILED,
            message="x",
            exit_code=99,  # what the trainer thought
            wall_seconds=1.0,
        )
        event = _build_trainer_exited_event(
            rc=1,  # what asyncio observed
            signal_name=None,
            cancellation_requested=False,
            payload=payload,
            synthesize_if_missing=False,
        )
        assert event["exit_code"] == 1


# ---------------------------------------------------------------------------
# Class 4: SIGKILL heuristic (R-SIGKILL)
# ---------------------------------------------------------------------------


class TestSigkillHeuristic:
    """File absent + SIGKILL exit code → synthesised TRAINING_OOM payload."""

    @pytest.mark.parametrize("rc", [137, -9])
    def test_sigkill_synthesises_training_oom(self, rc: int) -> None:
        event = _build_trainer_exited_event(
            rc=rc, signal_name="SIGKILL", cancellation_requested=False,
            payload=None, synthesize_if_missing=True,
        )
        assert event["code"] == ErrorCode.TRAINING_OOM.value
        assert "SIGKILL" in event["message"]
        assert event["payload_source"] == "sigkill_heuristic"

    def test_non_sigkill_rc_falls_back_to_internal_error(self) -> None:
        """rc=1 (or any non-SIGKILL non-zero) with no payload →
        INTERNAL_ERROR, not TRAINING_OOM. The OOM heuristic is
        narrow on purpose — only SIGKILL signatures qualify."""
        event = _build_trainer_exited_event(
            rc=1, signal_name=None, cancellation_requested=False,
            payload=None, synthesize_if_missing=True,
        )
        assert event["code"] == ErrorCode.INTERNAL_ERROR.value
        assert "exit_code=1" in event["message"]
        assert event["payload_source"] == "missing"

    @pytest.mark.parametrize("rc", [134, 139, 143])
    def test_other_signal_rcs_get_internal_error_not_oom(self, rc: int) -> None:
        """SIGABRT(134), SIGSEGV(139), SIGTERM(143) → not OOM.
        Pins that the heuristic is exclusively about the 137/-9 pair."""
        event = _build_trainer_exited_event(
            rc=rc, signal_name="SIGFOO", cancellation_requested=False,
            payload=None, synthesize_if_missing=True,
        )
        assert event["code"] == ErrorCode.INTERNAL_ERROR.value


# ---------------------------------------------------------------------------
# Class 5: schema_version pin
# ---------------------------------------------------------------------------


class TestSchemaVersionPin:
    """The ``trainer_exited`` event schema_version is pinned at 2.

    Bumping it requires updating both the supervisor (producer) and
    every consumer (training_monitor.py and any downstream code that
    pattern-matches on the event payload). This test is the canary.
    """

    def test_schema_version_is_two(self) -> None:
        assert TRAINER_EXITED_EVENT_SCHEMA_VERSION == 2

    def test_every_event_branch_stamps_schema_version(self) -> None:
        """Every code path in ``_build_trainer_exited_event`` must
        write the schema_version. Sample each branch."""
        for (rc, payload, synth) in [
            (0, None, False),
            (1, None, True),
            (137, None, True),
            (0, TrainerExitPayload(
                code=ErrorCode.TRAINING_FAILED, message="x",
                exit_code=0, wall_seconds=0.0,
            ), False),
        ]:
            event = _build_trainer_exited_event(
                rc=rc, signal_name=None, cancellation_requested=False,
                payload=payload, synthesize_if_missing=synth,
            )
            assert event["schema_version"] == TRAINER_EXITED_EVENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Class 6: End-to-end — real subprocess + supervisor reap
# ---------------------------------------------------------------------------


pytestmark_e2e = pytest.mark.asyncio


@pytest.fixture
def fsm(tmp_path: Path) -> JobLifecycleFSM:
    f = JobLifecycleFSM(workspace_dir=tmp_path)
    f.restore_or_init()
    return f


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=100)


@pytest_asyncio.fixture
async def supervisor(
    fsm: JobLifecycleFSM, bus: EventBus, tmp_path: Path,
) -> "AsyncIterator[Supervisor]":
    s = Supervisor(fsm, bus, stdio_log_path=tmp_path / "trainer.stdio.log")
    try:
        yield s
    finally:
        await s.shutdown()


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


async def _wait_terminal(fsm: JobLifecycleFSM, timeout: float = 5.0) -> JobState:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        snap = fsm.current()
        if snap is not None and snap.state in (
            JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED,
        ):
            return snap.state
        await asyncio.sleep(0.02)
    snap = fsm.current()
    raise AssertionError(
        f"FSM did not reach terminal within {timeout}s; "
        f"last={snap.state.value if snap else 'None'}",
    )


class TestSupervisorReapWithExitFile:
    """End-to-end: trainer writes trainer-exit.json; supervisor reads it."""

    @pytest.mark.asyncio
    async def test_trainer_file_payload_lands_in_event(
        self,
        supervisor: Supervisor,
        fsm: JobLifecycleFSM,
        bus: EventBus,
        tmp_path: Path,
    ) -> None:
        """A trainer that writes trainer-exit.json before exit produces
        a ``trainer_exited`` event carrying the typed code/message."""
        workdir = tmp_path / "run-1"
        workdir.mkdir()

        # Subprocess writes a Phase D payload then exits 1.
        code = (
            "import json, sys\n"
            "payload = {\n"
            "  'code': 'TRAINING_OOM',\n"
            "  'message': 'simulated OOM',\n"
            "  'traceback_summary': None,\n"
            "  'exit_code': 1,\n"
            "  'wall_seconds': 0.1,\n"
            "  'schema_version': 1,\n"
            "}\n"
            f"open({str(workdir / TRAINER_EXIT_FILENAME)!r}, 'w').write(json.dumps(payload))\n"
            "sys.exit(1)\n"
        )
        await supervisor.submit_and_spawn("j-e2e-1", _py(code), workdir=workdir)
        await _wait_terminal(fsm)

        # Find the trainer_exited event and check it carries the typed data.
        exited = [e for e in list(bus._buffer) if e.kind == "trainer_exited"]
        assert exited, "expected a trainer_exited event"
        body = exited[-1].payload
        assert body["payload_source"] == "trainer_file"
        assert body["code"] == "TRAINING_OOM"
        assert body["message"] == "simulated OOM"
        assert body["schema_version"] == TRAINER_EXITED_EVENT_SCHEMA_VERSION

    @pytest.mark.asyncio
    async def test_missing_payload_with_nonzero_exit_synthesises_internal_error(
        self,
        supervisor: Supervisor,
        fsm: JobLifecycleFSM,
        bus: EventBus,
        tmp_path: Path,
    ) -> None:
        """A trainer that exits non-zero without writing the payload
        gets INTERNAL_ERROR (not TRAINING_OOM — that's reserved for
        the SIGKILL heuristic)."""
        workdir = tmp_path / "run-2"
        workdir.mkdir()
        await supervisor.submit_and_spawn(
            "j-e2e-2",
            _py("import sys; sys.exit(7)"),
            workdir=workdir,
        )
        await _wait_terminal(fsm)
        exited = [e for e in list(bus._buffer) if e.kind == "trainer_exited"]
        assert exited
        body = exited[-1].payload
        assert body["payload_source"] == "missing"
        assert body["code"] == ErrorCode.INTERNAL_ERROR.value
        assert "exit_code=7" in body["message"]


# ---------------------------------------------------------------------------
# Class 7: Regression / invariants over the event shape
# ---------------------------------------------------------------------------


class TestEventInvariants:
    """Cross-cutting invariants the event MUST always satisfy."""

    def test_event_always_contains_legacy_keys(self) -> None:
        """Legacy ``exit_code``/``signal``/``cancellation_requested``
        are part of the contract — consumers pre-D pattern-match on
        them. Every branch must emit them so the schema bump is
        purely additive."""
        for payload, synth in [
            (None, False),
            (None, True),
            (TrainerExitPayload(
                code=ErrorCode.TRAINING_FAILED, message="x",
                exit_code=1, wall_seconds=0.0,
            ), False),
        ]:
            event = _build_trainer_exited_event(
                rc=1, signal_name="SIGKILL", cancellation_requested=False,
                payload=payload, synthesize_if_missing=synth,
            )
            assert "exit_code" in event
            assert "signal" in event
            assert "cancellation_requested" in event

    def test_event_payload_source_taxonomy_is_closed(self) -> None:
        """Only 4 well-known ``payload_source`` values across all
        branches. New branches require updating this allowlist + the
        consumer in training_monitor.py."""
        allowed = {"trainer_file", "sigkill_heuristic", "missing", "none"}
        sources = set()
        for (rc, payload, synth) in [
            (0, None, False),
            (1, None, True),
            (137, None, True),
            (1, TrainerExitPayload(
                code=ErrorCode.TRAINING_FAILED, message="x",
                exit_code=1, wall_seconds=0.0,
            ), False),
        ]:
            event = _build_trainer_exited_event(
                rc=rc, signal_name=None, cancellation_requested=False,
                payload=payload, synthesize_if_missing=synth,
            )
            sources.add(event["payload_source"])
        assert sources <= allowed

    def test_synthesized_event_message_is_non_empty(self) -> None:
        for rc in [137, -9, 1]:
            event = _build_trainer_exited_event(
                rc=rc, signal_name=None, cancellation_requested=False,
                payload=None, synthesize_if_missing=True,
            )
            assert event["message"] and len(event["message"]) > 0

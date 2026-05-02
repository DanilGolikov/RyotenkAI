"""PR-B — Supervisor's stdio_tail push in the ``trainer_exited`` payload.

Real-subprocess tests: each test spawns a small Python child that
writes to stdout/stderr in a shape relevant to the failure mode under
inspection, then asserts on:

* the ``trainer_exited`` event payload (schema_version=2, stderr_tail,
  stdout_tail, stdio_log_path),
* the ``_read_stdio_tail`` helper directly for boundary cases that are
  awkward to provoke through a real subprocess (file missing, partial
  decode, secret redaction).

These cover the seven test categories from the plan
(``docs/plans/2026-05-02-fail-fast-prevention-and-log-visibility.md``
§7) for PR-B.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from src.runner.event_bus import EventBus
from src.runner.state import JobLifecycleFSM, JobState
from src.runner.supervisor import (
    STDIO_TAIL_MAX_BYTES,
    STDIO_TAIL_MAX_LINES,
    TRAINER_EXITED_SCHEMA_VERSION,
    Supervisor,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Async tests get the marker individually below (TestPayloadShape /
# TestTailContent / TestEmptyAndMissing / TestReapResilience). The
# TestBoundary and TestRedaction classes are pure sync — no event loop
# needed — so they live outside the asyncio marker to avoid noisy
# pytest-asyncio warnings.


# ---------------------------------------------------------------------------
# Fixtures (reused from test_supervisor.py shape — keep behaviour aligned)
# ---------------------------------------------------------------------------


@pytest.fixture
def fsm(tmp_path: Path) -> JobLifecycleFSM:
    f = JobLifecycleFSM(workspace_dir=tmp_path)
    f.restore_or_init()
    return f


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=100)


@pytest.fixture
def stdio_log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "trainer.stdio.log"


@pytest_asyncio.fixture
async def supervisor(
    fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
) -> AsyncIterator[Supervisor]:
    s = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
    try:
        yield s
    finally:
        await s.shutdown()


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


async def _wait_terminal(fsm: JobLifecycleFSM, *, timeout: float = 5.0) -> JobState:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        snap = fsm.current()
        if snap is not None and snap.state in (
            JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED,
        ):
            return snap.state
        await asyncio.sleep(0.02)
    raise AssertionError(f"FSM did not reach terminal state in {timeout}s")


def _last_trainer_exited(bus: EventBus) -> dict[str, Any]:
    """Return the most recent ``trainer_exited`` payload from the bus.

    Uses ``bus._buffer`` directly — the same approach the existing
    Supervisor tests take (no public history API yet, that's an
    intentional choice in EventBus to keep history streaming-only).
    """
    events = list(bus._buffer)
    for ev in reversed(events):
        if ev.kind == "trainer_exited":
            return dict(ev.payload)
    raise AssertionError(f"no trainer_exited event in bus snapshot: {events!r}")


# ---------------------------------------------------------------------------
# Positive — happy path with both streams
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPayloadShape:
    async def test_payload_carries_schema_version_2(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        await supervisor.submit_and_spawn("j-v2", _py("print('hi')"))
        await _wait_terminal(fsm)
        payload = _last_trainer_exited(bus)
        assert payload["schema_version"] == TRAINER_EXITED_SCHEMA_VERSION

    async def test_payload_includes_stdio_log_path_when_configured(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
        bus: EventBus, stdio_log_path: Path,
    ) -> None:
        await supervisor.submit_and_spawn("j-path", _py("print('hi')"))
        await _wait_terminal(fsm)
        payload = _last_trainer_exited(bus)
        assert payload["stdio_log_path"] == str(stdio_log_path)

    async def test_payload_keeps_legacy_fields_v1_compat(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        """Schema bump must remain *additive*: every v1 field still
        present so legacy v1 consumers (Mac WS subscribers built before
        PR-B landed) continue to work — any new field they don't know
        about is just ignored."""
        await supervisor.submit_and_spawn(
            "j-v1compat", _py("import sys; sys.exit(3)"),
        )
        await _wait_terminal(fsm)
        payload = _last_trainer_exited(bus)
        # v1 invariant
        assert payload["exit_code"] == 3
        assert payload["signal"] is None
        assert payload["cancellation_requested"] is False
        # v2 additions
        assert "stderr_tail" in payload
        assert "stdout_tail" in payload


@pytest.mark.asyncio
class TestTailContent:
    async def test_stderr_tail_carries_traceback_text(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        """Regression for the 15-crash incident: trainer prints a
        ModuleNotFoundError to stderr and exits non-zero; Mac must see
        the traceback in the trainer_exited payload even if the pod is
        immediately platform-evicted."""
        await supervisor.submit_and_spawn(
            "j-traceback",
            _py("import sys; sys.stderr.write('ModuleNotFoundError: src.providers\\n'); sys.exit(1)"),
        )
        await _wait_terminal(fsm)
        payload = _last_trainer_exited(bus)
        assert "ModuleNotFoundError" in payload["stderr_tail"]
        assert "src.providers" in payload["stderr_tail"]

    async def test_stdout_tail_split_from_stderr(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        """Stdout and stderr must be reported in separate fields — the
        ``[OUT] /[ERR] `` prefix in the capture file is the source of
        the split."""
        await supervisor.submit_and_spawn(
            "j-split",
            _py(
                "import sys;"
                " print('out-line');"
                " sys.stderr.write('err-line\\n');"
                " sys.exit(0)"
            ),
        )
        await _wait_terminal(fsm)
        payload = _last_trainer_exited(bus)
        assert "out-line" in payload["stdout_tail"]
        assert "err-line" in payload["stderr_tail"]
        # Cross-contamination guard
        assert "out-line" not in payload["stderr_tail"]
        assert "err-line" not in payload["stdout_tail"]


# ---------------------------------------------------------------------------
# Negative / Boundary — empty, missing, large
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEmptyAndMissing:
    async def test_silent_exit_yields_empty_tails(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        """Trainer exits without producing any output → tails are empty
        but the payload is still well-formed (schema_version present)."""
        await supervisor.submit_and_spawn(
            "j-silent", _py("import sys; sys.exit(2)"),
        )
        await _wait_terminal(fsm)
        payload = _last_trainer_exited(bus)
        assert payload["stderr_tail"] == ""
        assert payload["stdout_tail"] == ""
        assert payload["schema_version"] == TRAINER_EXITED_SCHEMA_VERSION

    async def test_no_stdio_path_configured_yields_empty(
        self, fsm: JobLifecycleFSM, bus: EventBus,
    ) -> None:
        """If a Supervisor is created without ``stdio_log_path``, the
        tail reader must return ``("", "")`` — no exceptions, no AttributeError."""
        sup = Supervisor(fsm, bus, stdio_log_path=None)
        try:
            await sup.submit_and_spawn("j-no-path", _py("print('x')"))
            await _wait_terminal(fsm)
            payload = _last_trainer_exited(bus)
            assert payload["stderr_tail"] == ""
            assert payload["stdout_tail"] == ""
            assert payload["stdio_log_path"] is None
        finally:
            await sup.shutdown()


class TestBoundary:
    def test_read_stdio_tail_bounds_payload_to_max_bytes(
        self, fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
    ) -> None:
        """A 1 MB log file must not produce a 1 MB payload — we read
        only the last STDIO_TAIL_MAX_BYTES from the end of the file."""
        stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
        big = b"[OUT] " + (b"x" * 100) + b"\n"
        # Write ~50 KB of synthetic [OUT] lines, then 5 distinctive trailing lines.
        with stdio_log_path.open("wb") as fh:
            for _ in range(500):
                fh.write(big)
            for tag in (b"AAAA", b"BBBB", b"CCCC", b"DDDD", b"EEEE"):
                fh.write(b"[OUT] tail-" + tag + b"\n")

        sup = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
        # Mimic the spawn path: open the same handle the pumps would use.
        sup._stdio_log_file = stdio_log_path.open("ab", buffering=0)  # type: ignore[assignment]
        try:
            stderr_tail, stdout_tail = sup._read_stdio_tail()
        finally:
            sup._stdio_log_file.close()

        # Total payload bytes ≤ STDIO_TAIL_MAX_BYTES.
        assert len(stderr_tail) + len(stdout_tail) <= STDIO_TAIL_MAX_BYTES
        # Most recent lines preserved (the explicit tail tags).
        assert "tail-EEEE" in stdout_tail
        # Lines budget honoured.
        assert stdout_tail.count("\n") <= STDIO_TAIL_MAX_LINES

    def test_read_stdio_tail_drops_partial_first_line_after_seek(
        self, fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
    ) -> None:
        """Seek-from-end usually lands mid-line; the first ``readline``
        consumes that partial fragment so subsequent lines are whole."""
        stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Build a payload > STDIO_TAIL_MAX_BYTES so the seek fires.
        with stdio_log_path.open("wb") as fh:
            fh.write(b"[OUT] " + (b"a" * STDIO_TAIL_MAX_BYTES) + b"\n")
            fh.write(b"[OUT] complete-final-line\n")

        sup = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
        sup._stdio_log_file = stdio_log_path.open("ab", buffering=0)  # type: ignore[assignment]
        try:
            _, stdout_tail = sup._read_stdio_tail()
        finally:
            sup._stdio_log_file.close()

        # Partial-decode artefact (no [OUT] prefix on the truncated
        # first row) MUST be filtered out — and the complete line stays.
        assert "complete-final-line" in stdout_tail
        # No fragment of the gigantic 'a'-line should leak (it had no
        # [OUT] prefix after truncation).
        assert "a" * 100 not in stdout_tail

    def test_read_stdio_tail_handles_missing_file(
        self, fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
    ) -> None:
        """File deleted out from under us → return ``("", "")``, never raise."""
        sup = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
        # No spawn → no file handle → method must short-circuit safely.
        assert sup._read_stdio_tail() == ("", "")

    def test_read_stdio_tail_handles_zero_byte_file(
        self, fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
    ) -> None:
        """Zero-byte capture file (trainer crashed before the first byte
        was flushed) → empty tails, no exception."""
        stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdio_log_path.touch()
        sup = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
        sup._stdio_log_file = stdio_log_path.open("ab", buffering=0)  # type: ignore[assignment]
        try:
            assert sup._read_stdio_tail() == ("", "")
        finally:
            sup._stdio_log_file.close()


# ---------------------------------------------------------------------------
# Invariants — secret redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_stderr_tail_redacts_hf_token(
        self, fsm: JobLifecycleFSM, bus: EventBus, stdio_log_path: Path,
    ) -> None:
        """RP4: trainer prints HF_TOKEN to stderr (e.g. via env dump) →
        the tail returned to the WS bridge must NOT contain the token
        body. Centralized via ``redact_secrets`` so the policy lives in
        one place."""
        stdio_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdio_log_path.write_bytes(
            b"[ERR] crashed at HF_TOKEN=hf_realsecretXYZ123 in env\n"
        )
        sup = Supervisor(fsm, bus, stdio_log_path=stdio_log_path)
        sup._stdio_log_file = stdio_log_path.open("ab", buffering=0)  # type: ignore[assignment]
        try:
            stderr_tail, _ = sup._read_stdio_tail()
        finally:
            sup._stdio_log_file.close()

        assert "hf_realsecretXYZ123" not in stderr_tail
        # KEY name preserved for operator triage
        assert "HF_TOKEN" in stderr_tail


# ---------------------------------------------------------------------------
# Dependency-error — unreadable file MUST not raise from reap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapResilience:
    async def test_reap_proceeds_when_tail_read_raises(
        self, supervisor: Supervisor, fsm: JobLifecycleFSM,
        bus: EventBus, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``_read_stdio_tail`` somehow raises, the reap path must
        still publish ``trainer_exited`` and transition the FSM to a
        terminal state. Without this guard a stuck pump or a corrupted
        file could keep a job in ``running`` forever."""
        def _boom(self_: Any) -> tuple[str, str]:
            raise RuntimeError("intentional test failure")

        monkeypatch.setattr(Supervisor, "_read_stdio_tail", _boom)

        await supervisor.submit_and_spawn(
            "j-resilient", _py("import sys; sys.exit(4)"),
        )
        terminal = await _wait_terminal(fsm)
        assert terminal == JobState.FAILED
        # And payload still emitted with empty tails as fallback.
        payload = _last_trainer_exited(bus)
        assert payload["stderr_tail"] == ""
        assert payload["stdout_tail"] == ""
        assert payload["schema_version"] == TRAINER_EXITED_SCHEMA_VERSION

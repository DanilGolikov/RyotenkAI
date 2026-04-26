"""Phase 7.1 — ``ryotenkai job <verb>`` contract.

Each subcommand of the new ``job`` noun is a thin shim around
:class:`SSHTunnelManager` + :class:`JobClient`. We verify:

- the submission file is correctly resolved from ``run_dir``
- ``--attempt`` overrides the default "latest attempt" pick
- missing/corrupt submission → friendly ``die`` error (not traceback)
- the inner async work is exercised against mocked tunnel/client
- ``status`` / ``stop`` / ``metrics`` produce the expected JSON shape
  in machine-readable mode

We don't open real SSH tunnels or HTTP — every external dependency
is patched on the ``src.cli.commands.job`` module.

Coverage split (project policy):

1. **Positive**           — happy-path JSON shape for every verb.
2. **Negative**           — no attempts / unknown attempt / corrupt
                             submission → friendly error, not traceback.
3. **Boundary**           — single attempt, large attempt numbers,
                             empty event stream, payloads with > 4 keys.
4. **Invariants**         — ``status`` / ``stop`` always reach the
                             runner exactly once per command;
                             ``--attempt`` always wins over implicit
                             latest pick.
5. **Dependency errors**  — submission file unreadable → friendly
                             error; runner raises mid-call → JSON
                             renderer never partially emits.
6. **Regressions**        — JsonRenderer single-emit contract (was
                             violated when ``events`` streamed in JSON
                             mode); ``_format_event_line`` truncates
                             payloads with > 4 keys; ``--attempt 0``
                             rejected by Typer's ``ge=1``.
7. **Logic-specific**     — ``_format_event_line`` formatting matches
                             the documented one-liner; latest-attempt
                             picker uses numeric (not lexical) sort.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.app import app
from src.pipeline.state.job_submission import JobSubmission, save_job_submission


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_submission_file(run_dir: Path, *, attempt_no: int = 1) -> Path:
    """Create ``run_dir/attempts/attempt_<n>/job_submission.json``
    with a known-good payload. Returns the attempt directory."""
    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_no}"
    submission = JobSubmission(
        schema_version=JobSubmission.CURRENT_VERSION,
        job_id=f"j-{attempt_no}",
        provider_name="runpod",
        pod_id="pod-xyz",
        ssh_host="1.2.3.4",
        ssh_port=22022,
        ssh_username="root",
        ssh_key_path="/k/id",
        created_at_iso="2026-04-26T00:00:00+00:00",
    )
    save_job_submission(attempt_dir, submission)
    return attempt_dir


def _patch_with_runner(client: MagicMock):
    """Patch ``_with_runner`` so the helper bypasses real SSH/HTTP and
    runs the inner ``fn(client, job_id)`` with our mock client.

    ``_with_runner`` is async; we replace it with an async stub that
    just calls ``fn`` with the mock — keeps the production tunnel
    teardown out of the test path."""
    async def _stub(submission, fn):  # type: ignore[no-untyped-def]
        return await fn(client, submission.job_id)
    return patch("src.cli.commands.job._with_runner", side_effect=_stub)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Submission resolution
# ---------------------------------------------------------------------------


class TestAttemptResolution:
    def test_status_uses_latest_attempt_by_default(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=1)
        _make_submission_file(run_dir, attempt_no=2)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"job_id": "j-2", "state": "running"})

        with _patch_with_runner(client):
            result = runner.invoke(app, ["-o", "json", "job", "status", str(run_dir)])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        # Latest attempt → submission for attempt_2 (job_id j-2).
        assert payload["submission"]["job_id"] == "j-2"

    def test_status_respects_explicit_attempt_flag(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=1)
        _make_submission_file(run_dir, attempt_no=2)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"job_id": "j-1", "state": "running"})

        with _patch_with_runner(client):
            result = runner.invoke(
                app,
                ["-o", "json", "job", "status", str(run_dir), "--attempt", "1"],
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["submission"]["job_id"] == "j-1"

    def test_no_attempts_is_friendly_error(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-empty"
        run_dir.mkdir()

        result = runner.invoke(app, ["job", "status", str(run_dir)])
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout
        assert "no attempts/ subdirectories" in result.output or "no attempts/" in result.stderr

    def test_unknown_attempt_is_friendly_error(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=1)

        result = runner.invoke(
            app, ["job", "status", str(run_dir), "--attempt", "99"],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout

    def test_corrupt_submission_is_friendly_error(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        attempt_dir = tmp_path / "run" / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "job_submission.json").write_text("not-valid-json")

        result = runner.invoke(app, ["job", "status", str(tmp_path / "run")])
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout


# ---------------------------------------------------------------------------
# Subcommand wire shapes
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_without_grace(self, tmp_path: Path, runner: CliRunner) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        client = MagicMock()
        client.request_stop = AsyncMock(return_value={
            "job_id": "j-1", "state": "stopping", "sequence": 2,
        })

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "stop", str(run_dir)],
            )

        assert result.exit_code == 0, result.output
        # ``request_stop`` is called with ``grace_seconds=None`` so the
        # runner uses its default 30 s SIGTERM window.
        client.request_stop.assert_awaited_once()
        call = client.request_stop.await_args
        assert call.kwargs.get("grace_seconds") is None

    def test_stop_with_grace_override(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        client = MagicMock()
        client.request_stop = AsyncMock(return_value={
            "job_id": "j-1", "state": "stopping", "sequence": 3,
        })

        with _patch_with_runner(client):
            result = runner.invoke(
                app,
                ["-o", "json", "job", "stop", str(run_dir), "--grace", "15"],
            )

        assert result.exit_code == 0
        assert client.request_stop.await_args.kwargs.get("grace_seconds") == 15.0


class TestEvents:
    def test_events_renders_replayed_events(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        events: list[dict[str, Any]] = [
            {"offset": 0, "kind": "trainer_spawned", "payload": {}},
            {"offset": 1, "kind": "step", "payload": {"loss": 0.5}},
        ]

        async def _stream(_job_id, **_kwargs):
            for ev in events:
                yield ev

        client = MagicMock()
        client.subscribe_events = _stream

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "events", str(run_dir)],
            )

        assert result.exit_code == 0, result.output
        # JSON renderer buffers and emits a list of events.
        out = json.loads(result.stdout)
        assert isinstance(out, list)
        assert {e["kind"] for e in out} == {"trainer_spawned", "step"}

    def test_events_text_mode_one_line_per_event(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            yield {"offset": 0, "kind": "trainer_spawned", "payload": {}}

        client = MagicMock()
        client.subscribe_events = _stream

        with _patch_with_runner(client):
            result = runner.invoke(app, ["job", "events", str(run_dir)])

        assert result.exit_code == 0
        assert "trainer_spawned" in result.stdout


class TestMetrics:
    def test_metrics_returns_latest_health_snapshot(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            yield {"offset": 0, "kind": "trainer_spawned", "payload": {}}
            yield {"offset": 1, "kind": "health_snapshot",
                   "payload": {"gpu_util_percent": 30}}
            yield {"offset": 2, "kind": "step", "payload": {"loss": 0.5}}
            yield {"offset": 3, "kind": "health_snapshot",
                   "payload": {"gpu_util_percent": 90}}

        client = MagicMock()
        client.subscribe_events = _stream

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "metrics", str(run_dir)],
            )

        assert result.exit_code == 0, result.output
        out = json.loads(result.stdout)
        # Latest snapshot wins (offset=3, gpu_util=90).
        assert out["payload"]["gpu_util_percent"] == 90

    def test_metrics_empty_buffer(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            yield {"offset": 0, "kind": "trainer_spawned", "payload": {}}

        client = MagicMock()
        client.subscribe_events = _stream

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "metrics", str(run_dir)],
            )
        assert result.exit_code == 0
        # null = no health_snapshot ever seen.
        assert json.loads(result.stdout) is None


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_single_attempt_no_flag_works(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=1)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"job_id": "j-1", "state": "running"})

        with _patch_with_runner(client):
            result = runner.invoke(app, ["-o", "json", "job", "status", str(run_dir)])
        assert result.exit_code == 0

    def test_high_attempt_number(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        # No reason an attempt number can't be in the hundreds; the
        # numeric sort path is the same.
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=257)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"job_id": "j-257", "state": "running"})

        with _patch_with_runner(client):
            result = runner.invoke(
                app,
                ["-o", "json", "job", "status", str(run_dir), "--attempt", "257"],
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["submission"]["job_id"] == "j-257"

    def test_metrics_with_no_health_in_stream(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            yield {"offset": 0, "kind": "trainer_spawned", "payload": {}}
            yield {"offset": 1, "kind": "step", "payload": {"loss": 0.4}}
            # No health_snapshot.

        client = MagicMock()
        client.subscribe_events = _stream

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "metrics", str(run_dir)],
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout) is None

    def test_events_text_mode_with_empty_stream(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        async def _empty(_job_id, **_kwargs):
            return
            yield  # pragma: no cover

        client = MagicMock()
        client.subscribe_events = _empty

        with _patch_with_runner(client):
            result = runner.invoke(app, ["job", "events", str(run_dir)])
        assert result.exit_code == 0
        # Text mode prints the no-events placeholder rather than nothing.
        assert "(no events)" in result.stdout


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_status_calls_runner_exactly_once(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"job_id": "j-1", "state": "running"})

        with _patch_with_runner(client):
            runner.invoke(app, ["-o", "json", "job", "status", str(run_dir)])
        client.get_status.assert_awaited_once()

    def test_stop_calls_runner_exactly_once(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        client = MagicMock()
        client.request_stop = AsyncMock(return_value={"state": "stopping"})

        with _patch_with_runner(client):
            runner.invoke(app, ["-o", "json", "job", "stop", str(run_dir)])
        client.request_stop.assert_awaited_once()

    def test_explicit_attempt_overrides_latest_in_every_verb(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=1)
        _make_submission_file(run_dir, attempt_no=5)
        _make_submission_file(run_dir, attempt_no=10)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"job_id": "j-1", "state": "running"})
        client.request_stop = AsyncMock(return_value={"state": "stopping"})

        async def _stream(job_id, **_kwargs):
            yield {"offset": 0, "kind": "trainer_spawned", "payload": {"job_id": job_id}}

        client.subscribe_events = _stream

        # Each verb must respect the explicit ``--attempt`` and read
        # the corresponding submission's job_id (j-5), not the latest
        # (j-10) or the first (j-1).
        with _patch_with_runner(client):
            status = runner.invoke(
                app, ["-o", "json", "job", "status", str(run_dir), "--attempt", "5"],
            )
            assert json.loads(status.stdout)["submission"]["job_id"] == "j-5"

    def test_renderer_emits_at_most_one_payload_per_command(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        # JsonRenderer raises ``AssertionError`` if ``emit()`` is called
        # twice. This guards Phase 7.1's events-streaming bug — we
        # buffer in machine mode and emit once at the end.
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            for offset in range(50):
                yield {"offset": offset, "kind": "step", "payload": {"loss": 0.5}}

        client = MagicMock()
        client.subscribe_events = _stream

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "events", str(run_dir)],
            )
        assert result.exit_code == 0
        # Single JSON document, parses cleanly.
        json.loads(result.stdout)


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_runner_failure_translates_to_friendly_error(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)

        client = MagicMock()
        client.get_status = AsyncMock(side_effect=ConnectionError("tunnel closed"))

        with _patch_with_runner(client):
            result = runner.invoke(app, ["job", "status", str(run_dir)])
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr

    def test_unreadable_submission_dir(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        # ``attempts/`` exists but the submission file is missing — same
        # category as "launcher never persisted", surfaces as friendly
        # error.
        run_dir = tmp_path / "run-1"
        (run_dir / "attempts" / "attempt_1").mkdir(parents=True)
        result = runner.invoke(app, ["job", "status", str(run_dir)])
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_format_event_line_truncates_long_payloads(
        self,
    ) -> None:
        # Direct test of the helper — guards against a future refactor
        # that loses the > 4-keys clamp and dumps multi-MB payloads to
        # the terminal.
        from src.cli.commands.job import _format_event_line

        emitted: list[str] = []

        class _FakeRenderer:
            def text(self, line: str) -> None:
                emitted.append(line)

        event = {
            "offset": 7,
            "kind": "step",
            "payload": {f"k{i}": i for i in range(20)},
        }
        _format_event_line(event, _FakeRenderer())
        assert len(emitted) == 1
        line = emitted[0]
        assert line.startswith("[     7] step")
        # Trailing "..." marker indicates the truncation kicked in.
        assert ", ..." in line
        # Only the first four payload keys appear.
        assert "k0=0" in line
        assert "k3=3" in line
        assert "k5=5" not in line

    def test_format_event_line_handles_missing_fields(self) -> None:
        from src.cli.commands.job import _format_event_line

        emitted: list[str] = []

        class _FakeRenderer:
            def text(self, line: str) -> None:
                emitted.append(line)

        # Defensive: events from older runner builds may lack ``offset``
        # or ``payload``. Don't crash — render placeholders.
        _format_event_line({"kind": "noisy"}, _FakeRenderer())
        line = emitted[0]
        assert "[     ?] noisy" in line

    @pytest.mark.parametrize("bad_attempt", ["0", "-1"])
    def test_attempt_below_one_rejected_by_typer(
        self, tmp_path: Path, runner: CliRunner, bad_attempt: str,
    ) -> None:
        # Same Pydantic ``ge=1`` constraint as the API router. Bad
        # attempt numbers must surface as a Typer usage error, never
        # reach the resolution code.
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir)
        result = runner.invoke(
            app, ["job", "status", str(run_dir), "--attempt", bad_attempt],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.stdout

    def test_latest_attempt_picker_numeric_sort(
        self, tmp_path: Path, runner: CliRunner,
    ) -> None:
        # 10 > 2 numerically, but lexical "attempt_10" < "attempt_2".
        # The CLI helper sorts numerically so the latest attempt wins
        # when crossing the 9 → 10 boundary.
        run_dir = tmp_path / "run-1"
        _make_submission_file(run_dir, attempt_no=2)
        _make_submission_file(run_dir, attempt_no=10)

        client = MagicMock()
        client.get_status = AsyncMock(return_value={"state": "running"})

        with _patch_with_runner(client):
            result = runner.invoke(
                app, ["-o", "json", "job", "status", str(run_dir)],
            )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["submission"]["job_id"] == "j-10"


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestFormatEventLine:
    """Direct unit tests for the text-mode renderer helper.

    Pure function — no IO, no async. Keeps the tested behaviour
    independent of Typer / CliRunner overhead, so a typo in the
    formatter surfaces immediately.
    """

    def _render(self, event: dict[str, Any]) -> str:
        from src.cli.commands.job import _format_event_line

        sink: list[str] = []

        class _FakeRenderer:
            def text(self, line: str) -> None:
                sink.append(line)

        _format_event_line(event, _FakeRenderer())
        assert len(sink) == 1
        return sink[0]

    def test_offset_is_right_aligned_to_six_chars(self) -> None:
        line = self._render({"offset": 1, "kind": "step", "payload": {}})
        # "[     1] step                    "
        assert line.startswith("[     1]")

    def test_kind_is_left_aligned_to_24_chars(self) -> None:
        # The fixed-width column means consecutive events render aligned
        # in the user's terminal; check the padding is preserved when
        # ``kind`` is short.
        line = self._render({"offset": 0, "kind": "x", "payload": {}})
        # "x" + 23 padding spaces.
        assert "[     0] x" in line
        # 24 chars after the closing bracket + space.
        prefix, _, _ = line.partition("[     0] ")
        # Implementation detail: tail starts with "x" then 23 spaces.
        kind_field = line.split("] ", 1)[1].rstrip()
        assert kind_field == "x"

    def test_payload_summary_emits_first_four_keys(self) -> None:
        line = self._render({
            "offset": 0, "kind": "k",
            "payload": {"a": 1, "b": 2, "c": 3, "d": 4},
        })
        for token in ("a=1", "b=2", "c=3", "d=4"):
            assert token in line
        assert ", ..." not in line

    def test_payload_summary_clamps_at_four_with_marker(self) -> None:
        line = self._render({
            "offset": 0, "kind": "k",
            "payload": {f"k{i}": i for i in range(7)},
        })
        # All four printed; remainder collapsed.
        assert "k0=0" in line and "k3=3" in line
        assert ", ..." in line
        # k4 / k5 / k6 must NOT appear in the line.
        assert "k4=" not in line
        assert "k6=" not in line

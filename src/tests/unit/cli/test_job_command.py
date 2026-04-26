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

"""Phase 6.3b — :class:`TrainingMonitor` (post-rewrite) contract.

Covers the new JobClient/WebSocket-based monitor. The old SSH-poll
flow is exercised in :file:`test_stages_monitor.py` (skipped post
6.3b).

Coverage:
- TestExecuteContract     missing job_client/job_id → MONITOR_LAUNCHER_NOT_WIRED
- TestMockMode            mock provider short-circuits to Ok
- TestEventDispatch       trainer_exited(0) → on_training_completed +
                          Ok; non-zero → on_training_failed + Err;
                          cancelled → TRAINING_CANCELLED;
                          health_snapshot → on_resource_check
- TestStreamErrors        JobNotFoundError / JobClientError →
                          mapped error codes
- TestReplayTruncated     ReplayTruncatedError → falls back to
                          get_status; terminal state respected,
                          non-terminal raises MONITOR_REPLAY_TRUNCATED
- TestTunnelTeardown      tunnel.close + client.aclose called even
                          on success path

Tests bypass :mod:`src.pipeline.stages.__init__` heavy package init
by loading the monitor module directly via importlib (same pattern
as :file:`test_plugin_packer.py` / :file:`test_training_launcher_v2.py`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def _load_monitor():
    """Load the monitor module directly so we don't drag the whole
    pipeline.stages package into the test process (it eager-imports
    heavy deps not present in the dev venv)."""
    if "ryotenkai_monitor_test" in sys.modules:
        return sys.modules["ryotenkai_monitor_test"]
    repo_root = Path(__file__).resolve().parents[4]
    src_path = repo_root / "src" / "pipeline" / "stages" / "training_monitor.py"
    spec = importlib.util.spec_from_file_location(
        "ryotenkai_monitor_test", str(src_path),
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ryotenkai_monitor_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_monitor_mod = _load_monitor()
TrainingMonitor = _monitor_mod.TrainingMonitor
TrainingMonitorEventCallbacks = _monitor_mod.TrainingMonitorEventCallbacks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stub_config() -> Any:
    """Minimal config the monitor's __init__ accepts without touching
    PipelineStage internals."""
    cfg = SimpleNamespace()
    # PipelineStage.__init__ reads self.config.training and a few
    # other attrs; SimpleNamespace returns AttributeError for missing
    # which is fine because the monitor's __init__ doesn't dereference
    # them. Provide an empty namespace just in case.
    cfg.training = SimpleNamespace()
    return cfg


def _make_monitor(callbacks=None) -> TrainingMonitor:
    # Bypass PipelineStage.__init__ — it expects a fully-built config
    # object that we don't need for these tests. We instantiate the
    # subclass without calling its parent and set the bare attributes
    # the tests touch.
    monitor = TrainingMonitor.__new__(TrainingMonitor)
    monitor._secrets = None
    monitor._callbacks = callbacks or TrainingMonitorEventCallbacks()
    monitor._training_start_time = 0.0
    monitor._last_offset = 0
    return monitor


def _ctx_with_handles(client, *, tunnel=None, job_id="j-1") -> dict[str, Any]:
    return {
        "job_client": client,
        "ssh_tunnel": tunnel,
        "job_id": job_id,
    }


def _make_client(events: list[dict[str, Any]] | None = None):
    """Build a fake JobClient with subscribe_events as an async
    generator yielding ``events``."""
    client = MagicMock()

    async def _stream(_job_id, *, since=0, **_kwargs):
        for ev in events or []:
            yield ev

    client.subscribe_events = _stream
    client.aclose = AsyncMock(return_value=None)
    client.get_status = AsyncMock(return_value={"state": "completed"})
    return client


# ---------------------------------------------------------------------------
# execute() contract
# ---------------------------------------------------------------------------


class TestExecuteContract:
    def test_missing_handles_returns_err(self) -> None:
        monitor = _make_monitor()
        result = monitor.execute({})
        assert result.is_err()
        err = result.unwrap_err()
        assert err.code == "MONITOR_LAUNCHER_NOT_WIRED"


# ---------------------------------------------------------------------------
# Mock-mode short circuit
# ---------------------------------------------------------------------------


class TestMockMode:
    def test_mock_returns_ok(self) -> None:
        started = []
        completed = []
        cb = TrainingMonitorEventCallbacks(
            on_training_started=lambda: started.append(True),
            on_training_completed=lambda d: completed.append(d),
        )
        monitor = _make_monitor(cb)
        # The mock-mode branch reads context[StageNames.GPU_DEPLOYER]
        # which is the human-readable label, not a snake-case key.
        ctx = {
            "GPU Deployer": {"provider_info": {"mock": True}},
        }
        result = monitor.execute(ctx)
        assert result.is_ok()
        assert result.unwrap()["mock"] is True
        assert started and completed


# ---------------------------------------------------------------------------
# Event dispatch
# ---------------------------------------------------------------------------


class TestEventDispatch:
    def test_trainer_exited_zero_returns_ok(self) -> None:
        completions: list[float] = []
        cb = TrainingMonitorEventCallbacks(
            on_training_completed=lambda d: completions.append(d),
        )
        monitor = _make_monitor(cb)
        client = _make_client(events=[
            {"offset": 0, "kind": "trainer_spawned", "payload": {}},
            {
                "offset": 1, "kind": "trainer_exited",
                "payload": {"exit_code": 0, "signal": None,
                            "cancellation_requested": False},
            },
        ])
        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_ok()
        assert result.unwrap()["status"] == "completed"
        assert len(completions) == 1
        # Phase 11.E — teardown moved to cleanup() so the SSH tunnel
        # + JobClient stay alive through ModelRetriever.
        client.aclose.assert_not_awaited()
        monitor.cleanup()
        client.aclose.assert_awaited()

    def test_trainer_exited_nonzero_returns_err(self) -> None:
        failures: list[tuple[str, float]] = []
        cb = TrainingMonitorEventCallbacks(
            on_training_failed=lambda msg, d: failures.append((msg, d)),
        )
        monitor = _make_monitor(cb)
        client = _make_client(events=[
            {
                "offset": 0, "kind": "trainer_exited",
                "payload": {"exit_code": 137, "signal": "SIGKILL",
                            "cancellation_requested": False},
            },
        ])
        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_err()
        assert result.unwrap_err().code == "TRAINING_FAILED"
        assert failures and "exit_code=137" in failures[0][0]

    def test_trainer_exited_cancelled(self) -> None:
        monitor = _make_monitor()
        client = _make_client(events=[
            {
                "offset": 0, "kind": "trainer_exited",
                "payload": {"exit_code": 130, "signal": "SIGINT",
                            "cancellation_requested": True},
            },
        ])
        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_err()
        assert result.unwrap_err().code == "TRAINING_CANCELLED"

    def test_health_snapshot_fires_resource_check(self) -> None:
        seen: list[dict] = []
        cb = TrainingMonitorEventCallbacks(
            on_resource_check=lambda d: seen.append(d),
        )
        monitor = _make_monitor(cb)
        client = _make_client(events=[
            {"offset": 0, "kind": "health_snapshot",
             "payload": {"gpu_util_percent": 80.0}},
            {
                "offset": 1, "kind": "trainer_exited",
                "payload": {"exit_code": 0, "signal": None,
                            "cancellation_requested": False},
            },
        ])
        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_ok()
        assert seen == [{"gpu_util_percent": 80.0}]


# ---------------------------------------------------------------------------
# Stream errors
# ---------------------------------------------------------------------------


class TestStreamErrors:
    def test_job_not_found_maps_to_monitor_code(self) -> None:
        monitor = _make_monitor()

        async def _raise(_job_id, *, since=0, **_kwargs):
            raise _monitor_mod.JobNotFoundError("nope")
            yield  # pragma: no cover  (make this an async generator)

        client = MagicMock()
        client.subscribe_events = _raise
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_JOB_NOT_FOUND"

    def test_generic_client_error_propagates(self) -> None:
        monitor = _make_monitor()

        async def _raise(_job_id, *, since=0, **_kwargs):
            raise _monitor_mod.JobClientError("transport flaky")
            yield  # pragma: no cover

        client = MagicMock()
        client.subscribe_events = _raise
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_CLIENT_ERROR"


# ---------------------------------------------------------------------------
# Replay truncation fallback
# ---------------------------------------------------------------------------


class TestReplayTruncated:
    def test_truncated_with_terminal_status_returns_terminal(self) -> None:
        monitor = _make_monitor()

        async def _raise(_job_id, *, since=0, **_kwargs):
            raise _monitor_mod.ReplayTruncatedError("buffer rolled")
            yield  # pragma: no cover

        client = MagicMock()
        client.subscribe_events = _raise
        client.get_status = AsyncMock(return_value={"state": "completed"})
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_ok()
        client.get_status.assert_awaited()

    def test_truncated_with_non_terminal_returns_err(self) -> None:
        monitor = _make_monitor()

        async def _raise(_job_id, *, since=0, **_kwargs):
            raise _monitor_mod.ReplayTruncatedError("buffer rolled")
            yield  # pragma: no cover

        client = MagicMock()
        client.subscribe_events = _raise
        client.get_status = AsyncMock(return_value={"state": "running"})
        client.aclose = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_REPLAY_TRUNCATED"


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


class TestTunnelTeardown:
    def test_tunnel_and_client_closed_on_cleanup(self) -> None:
        # Phase 11.E — teardown happens in cleanup() (orchestrator
        # finalize) instead of execute().finally so the SSH tunnel +
        # JobClient stay alive across stages including ModelRetriever.
        monitor = _make_monitor()
        client = _make_client(events=[
            {"offset": 0, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ])
        tunnel = MagicMock()
        tunnel.close = AsyncMock(return_value=None)

        result = monitor.execute(_ctx_with_handles(client, tunnel=tunnel))
        assert result.is_ok()
        # Execute does NOT close — kept alive for downstream stages.
        tunnel.close.assert_not_awaited()
        client.aclose.assert_not_awaited()
        # Cleanup tears them down.
        monitor.cleanup()
        tunnel.close.assert_awaited()
        client.aclose.assert_awaited()

    def test_tunnel_close_failure_does_not_mask_error(self) -> None:
        monitor = _make_monitor()

        async def _raise(_job_id, *, since=0, **_kwargs):
            raise _monitor_mod.JobNotFoundError("nope")
            yield  # pragma: no cover

        client = MagicMock()
        client.subscribe_events = _raise
        client.aclose = AsyncMock(side_effect=RuntimeError("close boom"))

        tunnel = MagicMock()
        tunnel.close = AsyncMock(side_effect=RuntimeError("tunnel boom"))

        result = monitor.execute(_ctx_with_handles(client, tunnel=tunnel))
        # Original JobNotFoundError → MONITOR_JOB_NOT_FOUND, NOT
        # masked by tunnel-close noise.
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_JOB_NOT_FOUND"


# ---------------------------------------------------------------------------
# PR2 — Event mirror integration
# ---------------------------------------------------------------------------


def _ctx_with_attempt_dir(
    client, attempt_dir: Path, *, tunnel=None, job_id="j-1",
) -> dict[str, Any]:
    return {
        "job_client": client,
        "ssh_tunnel": tunnel,
        "job_id": job_id,
        # Key the monitor reads to spin up the mirror writer. Must
        # match PipelineContextKeys.ATTEMPT_DIRECTORY.
        "attempt_directory": str(attempt_dir),
    }


class TestEventMirrorIntegration:
    """PR2: monitor writes every received event to
    ``<attempt>/events/events_mirror.jsonl`` for cold-replay."""

    def test_mirror_writes_each_event_to_jsonl(self, tmp_path: Path) -> None:
        import json as _json

        attempt_dir = tmp_path / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)

        events = [
            {"v": 1, "offset": 0, "ts": "t0", "kind": "trainer_log",
             "payload": {"kind": "stdout", "line": "hello"}},
            {"v": 1, "offset": 1, "ts": "t1", "kind": "health_snapshot",
             "payload": {"gpu_util_percent": 50.0}},
            {"v": 1, "offset": 2, "ts": "t2", "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ]
        monitor = _make_monitor()
        client = _make_client(events=events)

        result = monitor.execute(_ctx_with_attempt_dir(client, attempt_dir))
        assert result.is_ok()

        mirror_path = attempt_dir / "events" / "events_mirror.jsonl"
        assert mirror_path.exists()
        lines = mirror_path.read_text(encoding="utf-8").splitlines()
        # All three events should be in the mirror.
        assert len(lines) == 3
        offsets = [_json.loads(line)["offset"] for line in lines]
        assert offsets == [0, 1, 2]

    def test_mirror_failure_does_not_break_monitor(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A broken mirror writer must NOT bring down the monitor —
        observability never blocks training."""
        attempt_dir = tmp_path / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)

        # Patch EventMirrorWriter on the monitor module to a mock
        # whose write() raises.
        bad_mirror = MagicMock()
        bad_mirror.write = MagicMock(side_effect=RuntimeError("disk full"))
        bad_mirror.close = MagicMock()

        monkeypatch.setattr(
            _monitor_mod, "EventMirrorWriter", lambda *_a, **_k: bad_mirror,
        )

        client = _make_client(events=[
            {"offset": 0, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ])
        monitor = _make_monitor()
        result = monitor.execute(_ctx_with_attempt_dir(client, attempt_dir))
        # mirror.write raises, but the monitor swallows it at debug
        # level — the run still ends Ok via trainer_exited dispatch.
        assert result.is_ok(), result

    def test_no_attempt_dir_skips_mirror(self, tmp_path: Path) -> None:
        """Without ``attempt_directory`` in context (e.g. in older
        callers / unit tests) the monitor must still work — no mirror
        is written, and that's OK."""
        monitor = _make_monitor()
        client = _make_client(events=[
            {"offset": 0, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ])
        # _ctx_with_handles deliberately omits attempt_directory.
        result = monitor.execute(_ctx_with_handles(client))
        assert result.is_ok()


# ---------------------------------------------------------------------------
# PR2 — trainer_log → logger.info routing
# ---------------------------------------------------------------------------


class TestTrainerLogRouting:
    """PR2: trainer-stdout/stderr events surface as INFO-level log
    entries so they land in ``training_monitor.log`` and
    ``pipeline.log``."""

    def test_trainer_log_events_emit_logger_info(
        self, tmp_path: Path, caplog,
    ) -> None:
        import logging

        attempt_dir = tmp_path / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)

        events = [
            {"v": 1, "offset": 0, "kind": "trainer_log",
             "payload": {"kind": "stdout", "line": "epoch 1 loss=0.5"}},
            {"v": 1, "offset": 1, "kind": "trainer_log",
             "payload": {"kind": "stderr", "line": "warning: deprecated"}},
            {"v": 1, "offset": 2, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ]
        monitor = _make_monitor()
        client = _make_client(events=events)

        with caplog.at_level(logging.INFO, logger="ryotenkai"):
            result = monitor.execute(_ctx_with_attempt_dir(client, attempt_dir))

        assert result.is_ok()
        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("[TRAINER:stdout] epoch 1 loss=0.5" in m for m in info_messages)
        assert any("[TRAINER:stderr] warning: deprecated" in m for m in info_messages)

    def test_trainer_log_with_empty_line_is_skipped(
        self, tmp_path: Path, caplog,
    ) -> None:
        """An empty ``line`` shouldn't produce ``[TRAINER:stdout] `` —
        empty banner lines from trainer would just spam the log."""
        import logging

        attempt_dir = tmp_path / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)

        events = [
            {"v": 1, "offset": 0, "kind": "trainer_log",
             "payload": {"kind": "stdout", "line": ""}},
            {"v": 1, "offset": 1, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ]
        monitor = _make_monitor()
        client = _make_client(events=events)

        with caplog.at_level(logging.INFO, logger="ryotenkai"):
            monitor.execute(_ctx_with_attempt_dir(client, attempt_dir))

        # No "[TRAINER:..." line should appear with empty content.
        for record in caplog.records:
            if "[TRAINER:" in record.getMessage():
                assert record.getMessage().split("] ", 1)[1] != ""

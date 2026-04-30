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

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


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
    monitor._last_status_log_time = 0.0
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
# Status reporting — `[MONITOR] ALIVE | …` lines from health_snapshot
# ---------------------------------------------------------------------------


class TestStatusReporting:
    def _capture_log(self, monkeypatch, monitor):
        """Patch the monitor's logger.info and return the captured calls."""
        captured: list[tuple[str, tuple[Any, ...]]] = []

        def _info(msg, *args, **_kwargs):
            captured.append((msg, args))

        monkeypatch.setattr(_monitor_mod.logger, "info", _info)
        return captured

    def test_status_line_format_includes_all_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = self._capture_log(monkeypatch, monitor)

        monitor._maybe_log_status({
            "gpu_util_percent": 87.5,
            "gpu_memory_percent": 62.0,
            "cpu_percent": 41.0,
            "ram_used_gb": 12.4,
            "ram_total_gb": 64.0,
        })

        assert len(captured) == 1
        msg, args = captured[0]
        assert "ALIVE" in msg
        assert "GPU" in msg and "VRAM" in msg and "CPU" in msg and "RAM" in msg
        # Format strings substituted; verify rendered values
        rendered = msg % args
        assert "GPU: 88%" in rendered or "GPU: 87%" in rendered
        assert "VRAM: 62%" in rendered
        assert "CPU: 41%" in rendered
        assert "12.4/64 GB" in rendered

    def test_status_line_renders_dash_for_missing_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = self._capture_log(monkeypatch, monitor)

        monitor._maybe_log_status({
            "gpu_util_percent": None,
            "gpu_memory_percent": None,
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
        })

        assert len(captured) == 1
        msg, args = captured[0]
        rendered = msg % args
        # Three em-dashes for GPU/VRAM/CPU; one for RAM.
        assert rendered.count("—") >= 4

    def test_rate_limit_suppresses_second_call_within_window(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = self._capture_log(monkeypatch, monitor)

        # Pin time so the rate-limit window is deterministic.
        fake_now = [1000.0]
        monkeypatch.setattr(_monitor_mod.time, "time", lambda: fake_now[0])

        payload = {"gpu_util_percent": 50.0}
        monitor._maybe_log_status(payload)
        # Same instant → suppressed.
        monitor._maybe_log_status(payload)
        # Five seconds later — still inside the 15-s window.
        fake_now[0] += 5
        monitor._maybe_log_status(payload)

        assert len(captured) == 1, "rate-limit window should suppress repeats"

        # Past the window → fires again.
        fake_now[0] += _monitor_mod.TRAINING_MONITOR_LOG_STATUS_INTERVAL + 1
        monitor._maybe_log_status(payload)
        assert len(captured) == 2

    def test_health_snapshot_calls_both_status_log_and_callback(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Status reporting must NOT replace ``on_resource_check`` —
        # MLflow integrations rely on the callback for system metrics.
        seen: list[dict] = []
        cb = TrainingMonitorEventCallbacks(
            on_resource_check=lambda d: seen.append(d),
        )
        monitor = _make_monitor(cb)
        captured = self._capture_log(monkeypatch, monitor)

        monitor._dispatch_event({
            "kind": "health_snapshot",
            "payload": {"gpu_util_percent": 50.0},
        })

        assert seen == [{"gpu_util_percent": 50.0}]
        # Status line emitted at least once.
        assert any("ALIVE" in msg for msg, _ in captured)

    def test_status_line_duration_renders_from_training_start(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = self._capture_log(monkeypatch, monitor)

        # Training started 1h 23m 45s ago.
        fake_now = 5000.0
        monitor._training_start_time = fake_now - (3600 + 23 * 60 + 45)
        monkeypatch.setattr(_monitor_mod.time, "time", lambda: fake_now)

        monitor._maybe_log_status({})

        rendered = captured[0][0] % captured[0][1]
        assert "1:23:45" in rendered


# ---------------------------------------------------------------------------
# Periodic log download — log_manager.download(silent=True) every 30 s
# plus a final ``silent=False`` flush on cleanup
# ---------------------------------------------------------------------------


class TestPeriodicLogDownload:
    def test_loop_invokes_download_each_tick_until_cancelled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio as _asyncio

        monitor = _make_monitor()
        log_manager = MagicMock()
        log_manager.download = MagicMock(return_value=True)

        # Drive the loop fast — tick interval ~1 ms — so the test
        # exercises ≥1 real iteration without a real 30-s sleep.
        monkeypatch.setattr(
            _monitor_mod, "TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL", 0.001,
        )

        async def _drive() -> None:
            task = _asyncio.create_task(
                monitor._log_downloader_loop(log_manager),
            )
            # Yield enough times for the loop to tick a couple of
            # times. Each iteration: sleep(0.001) → asyncio.to_thread.
            await _asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(_asyncio.CancelledError):
                await task

        _asyncio.run(_drive())
        assert log_manager.download.call_count >= 1
        # Periodic ticks request silent=True so transient failures
        # don't spam the operator's terminal.
        for call in log_manager.download.call_args_list:
            assert call.kwargs.get("silent") is True or call.args == (True,)

    def test_loop_swallows_download_exception(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio as _asyncio

        monitor = _make_monitor()
        log_manager = MagicMock()
        log_manager.download = MagicMock(side_effect=RuntimeError("scp boom"))

        monkeypatch.setattr(
            _monitor_mod, "TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL", 0.001,
        )

        async def _drive() -> None:
            task = _asyncio.create_task(
                monitor._log_downloader_loop(log_manager),
            )
            await _asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(_asyncio.CancelledError):
                await task

        # Should NOT raise even though every download() raises.
        _asyncio.run(_drive())
        assert log_manager.download.call_count >= 1

    def test_watch_and_download_runs_final_flush_on_finally(self) -> None:
        import asyncio as _asyncio

        monitor = _make_monitor()
        log_manager = MagicMock()
        # First call (final flush) succeeds.
        log_manager.download = MagicMock(return_value=True)

        client = _make_client(events=[
            {"offset": 0, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ])

        result = _asyncio.run(
            monitor._watch_and_download(client, "j-1", log_manager),
        )
        assert result.is_ok()
        # The downloader loop may not have fired (sleep elapsed before
        # cancel), but the FINAL flush is mandatory.
        final_calls = [
            c for c in log_manager.download.call_args_list
            if c.kwargs.get("silent") is False
            or (c.args and c.args[0] is False)
        ]
        assert final_calls, "final silent=False flush must fire on cleanup"

    def test_watch_and_download_skips_loop_when_no_log_manager(self) -> None:
        # Single-node / mock flows: no LogManager, so we should fall
        # straight through to ``_watch`` with no extra task.
        import asyncio as _asyncio

        monitor = _make_monitor()
        client = _make_client(events=[
            {"offset": 0, "kind": "trainer_exited",
             "payload": {"exit_code": 0, "signal": None,
                         "cancellation_requested": False}},
        ])

        result = _asyncio.run(
            monitor._watch_and_download(client, "j-1", None),
        )
        assert result.is_ok()


# ---------------------------------------------------------------------------
# LogManager construction from gpu_deployer context
# ---------------------------------------------------------------------------


class TestLogManagerFromContext:
    def test_returns_none_on_local_provider(self) -> None:
        monitor = _make_monitor()
        log_manager, ssh = monitor._build_log_manager_from_context({
            "provider_type": "local",
            "ssh_host": "127.0.0.1",
        })
        assert log_manager is None
        assert ssh is None

    def test_returns_none_when_ssh_host_missing(self) -> None:
        monitor = _make_monitor()
        log_manager, ssh = monitor._build_log_manager_from_context({
            "provider_type": "cloud",
        })
        assert log_manager is None
        assert ssh is None

    def test_constructs_for_cloud_provider(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Stub SSHClient so the test doesn't actually open a control
        # socket. We only care that the helper plumbs the deployer
        # context through to ``LogManager``.
        constructed: dict[str, Any] = {}

        class _FakeSSH:
            def __init__(self, **kwargs):
                constructed.update(kwargs)
            def close_master(self) -> None:
                pass

        class _FakeLM:
            def __init__(self, ssh, remote_path=None):
                constructed["remote_path"] = remote_path
                constructed["ssh"] = ssh

        monkeypatch.setattr(_monitor_mod, "SSHClient", _FakeSSH)
        monkeypatch.setattr(_monitor_mod, "LogManager", _FakeLM)

        monitor = _make_monitor()
        lm, ssh = monitor._build_log_manager_from_context({
            "provider_type": "cloud",
            "ssh_host": "pod.example.com",
            "ssh_port": 2222,
            "ssh_user": "root",
            "ssh_key_path": "/tmp/key",
            "is_alias_mode": False,
            "workspace_path": "/workspace",
        })
        assert lm is not None
        assert ssh is not None
        assert constructed["host"] == "pod.example.com"
        assert constructed["port"] == 2222
        assert constructed["username"] == "root"
        assert constructed["key_path"] == "/tmp/key"
        assert constructed["remote_path"] == "/workspace/training.log"

    def test_alias_mode_forces_username_and_key_to_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        constructed: dict[str, Any] = {}

        class _FakeSSH:
            def __init__(self, **kwargs):
                constructed.update(kwargs)
            def close_master(self) -> None:
                pass

        class _FakeLM:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(_monitor_mod, "SSHClient", _FakeSSH)
        monkeypatch.setattr(_monitor_mod, "LogManager", _FakeLM)

        monitor = _make_monitor()
        monitor._build_log_manager_from_context({
            "provider_type": "cloud",
            "ssh_host": "my-pod-alias",
            "ssh_user": "root",
            "ssh_key_path": "/should-be-ignored",
            "is_alias_mode": True,
        })
        # Alias mode tells SSHClient to read ~/.ssh/config — username
        # and key_path must NOT be passed through.
        assert constructed["username"] is None
        assert constructed["key_path"] is None

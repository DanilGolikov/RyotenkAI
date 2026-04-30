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
    monitor._ssh_client = None
    monitor._log_manager = None
    monitor._provider_name = None
    monitor._resource_id = None
    monitor._recovery_attempts = 0
    monitor._first_event_logged = False
    monitor._trainer_started_logged = False
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


# ---------------------------------------------------------------------------
# Post-mortem diagnostics — pod-side probes on non-zero exit
# ---------------------------------------------------------------------------


class TestPostMortemDiagnostics:
    def _fake_ssh(self, *, output_per_label: dict | None = None):
        outputs = output_per_label or {}
        executed: list[tuple[str, str]] = []

        def _exec(*, command, silent=False, timeout=30):
            # Match by substring — the probes embed the label-specific
            # path in the command, so we can pick up which probe fired.
            label = "unknown"
            for marker, lbl in [
                ("TRAINING_EXIT_CODE", "exit_code"),
                ("TRAINING_*", "workspace_markers"),
                ("training.faulthandler.log", "faulthandler"),
                ("training.log", "training_log_tail"),
                ("oom|kill|memory", "dmesg_oom"),
                ("nvrm|xid|nvidia", "dmesg_nvidia"),
                ("dmesg", "dmesg_tail"),
                ("nvidia-smi", "nvidia_smi"),
            ]:
                if marker in command:
                    label = lbl
                    break
            executed.append((label, command))
            stdout = outputs.get(label, "")
            return True, stdout, ""

        ssh = MagicMock()
        ssh.exec_command = MagicMock(side_effect=_exec)
        ssh.executed = executed
        return ssh

    def test_zero_exit_skips_postmortem(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        monitor._ssh_client = self._fake_ssh()
        monitor._handle_trainer_exited({
            "exit_code": 0, "signal": None, "cancellation_requested": False,
        })
        assert monitor._ssh_client.exec_command.call_count == 0

    def test_cancellation_skips_postmortem_even_with_nonzero_code(self) -> None:
        # Operator pressed stop — no crash to investigate.
        monitor = _make_monitor()
        monitor._ssh_client = self._fake_ssh()
        monitor._handle_trainer_exited({
            "exit_code": 137, "signal": "SIGKILL",
            "cancellation_requested": True,
        })
        assert monitor._ssh_client.exec_command.call_count == 0

    def test_nonzero_exit_runs_full_probe_set(self) -> None:
        monitor = _make_monitor()
        monitor._ssh_client = self._fake_ssh()
        monitor._handle_trainer_exited({
            "exit_code": 1, "signal": None, "cancellation_requested": False,
        })
        labels = [lbl for lbl, _ in monitor._ssh_client.executed]
        for required in [
            "exit_code", "workspace_markers", "faulthandler",
            "training_log_tail", "dmesg_tail", "dmesg_oom",
            "dmesg_nvidia", "nvidia_smi",
        ]:
            assert required in labels, (
                f"missing probe {required!r} in {labels!r}"
            )

    def test_signal_kill_with_zero_exit_still_triggers_postmortem(self) -> None:
        # SIGTERM-killed trainer that returns 0 (process raced before
        # exit) is still a crash from the operator's POV.
        monitor = _make_monitor()
        monitor._ssh_client = self._fake_ssh()
        monitor._handle_trainer_exited({
            "exit_code": 0, "signal": "SIGTERM",
            "cancellation_requested": False,
        })
        assert monitor._ssh_client.exec_command.called

    def test_probe_failure_does_not_abort_remaining_probes(self) -> None:
        monitor = _make_monitor()
        ssh = MagicMock()
        # First two probes raise, the rest succeed — the loop must
        # keep going so the operator gets the dmesg signals.
        call_count = {"n": 0}
        def _exec(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("ssh boom")
            return True, "ok-output", ""
        ssh.exec_command = MagicMock(side_effect=_exec)
        monitor._ssh_client = ssh
        monitor._handle_trainer_exited({
            "exit_code": 1, "signal": None, "cancellation_requested": False,
        })
        # All 8 probes attempted.
        assert ssh.exec_command.call_count == 8

    def test_no_ssh_client_makes_postmortem_a_noop(self) -> None:
        # Single-node / mock flow — _ssh_client is None.
        monitor = _make_monitor()
        # Should not raise.
        monitor._handle_trainer_exited({
            "exit_code": 1, "signal": None, "cancellation_requested": False,
        })

    def test_log_lines_carry_postmortem_prefix(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[tuple[str, tuple[Any, ...]]] = []

        def _info(msg, *args, **_kwargs):
            captured.append((msg, args))

        monkeypatch.setattr(_monitor_mod.logger, "info", _info)

        monitor = _make_monitor()
        monitor._ssh_client = self._fake_ssh(output_per_label={
            "exit_code": "137",
            "nvidia_smi": "RTX 5090, 99 %, 24000 MiB, 32000 MiB",
        })
        monitor._handle_trainer_exited({
            "exit_code": 137, "signal": None,
            "cancellation_requested": False,
        })
        # logger.info is called both with an already-formatted f-string
        # (no args) AND with %-format strings; render conditionally.
        def _render(msg, args):
            if not args:
                return msg
            try:
                return msg % args
            except TypeError:
                return msg
        rendered = [_render(m, a) for m, a in captured]
        # Banner + at least one probe with rendered content.
        assert any(
            "[MONITOR:POSTMORTEM] non-zero exit detected" in line
            for line in rendered
        )
        assert any(
            "[MONITOR:POSTMORTEM] exit_code: 137" in line for line in rendered
        )


# ---------------------------------------------------------------------------
# Pod resilience — RunPod SDK wake-up on JobClientError
# ---------------------------------------------------------------------------


class _FakeOk:
    def __init__(self, value):
        self._value = value

    def is_err(self):
        return False

    def is_ok(self):
        return True

    def unwrap(self):
        return self._value

    def unwrap_err(self):
        raise AssertionError("called unwrap_err on Ok")


class _FakeErr:
    def __init__(self, error):
        self._error = error

    def is_err(self):
        return True

    def is_ok(self):
        return False

    def unwrap(self):
        raise AssertionError("called unwrap on Err")

    def unwrap_err(self):
        return self._error


class TestPodResilience:
    def test_no_op_for_local_provider(self) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "single_node"
        monitor._resource_id = "ignored"
        result = monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        )
        assert result is None

    def test_no_op_when_runpod_but_no_resource_id(self) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = None
        assert monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        ) is None

    def test_no_op_when_no_api_key(self) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = "pod-1"
        monitor._secrets = SimpleNamespace(runpod_api_key=None)
        assert monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        ) is None

    def test_terminal_pod_returns_terminated_err(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = "pod-1"
        monitor._secrets = SimpleNamespace(runpod_api_key="rk-1")

        terminal_pod = {
            "id": "pod-1",
            "desiredStatus": "TERMINATED",
            "runtime": {"uptimeInSeconds": 0, "ports": []},
        }

        sdk = MagicMock()
        sdk.get_pod = MagicMock(return_value=_FakeOk(terminal_pod))
        sdk.start_pod = MagicMock()

        monkeypatch.setattr(
            "src.providers.runpod.sdk_adapter.RunPodSDKClient",
            lambda *, api_key: sdk,
        )

        result = monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        )
        assert result is not None
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_POD_TERMINATED"
        sdk.start_pod.assert_not_called()

    def test_running_pod_returns_none_for_caller_to_propagate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = "pod-1"
        monitor._secrets = SimpleNamespace(runpod_api_key="rk-1")

        running_pod = {
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 120,
                "ports": [
                    {"isIpPublic": True, "privatePort": 22,
                     "ip": "1.2.3.4", "publicPort": 2222, "type": "tcp"},
                ],
            },
        }

        sdk = MagicMock()
        sdk.get_pod = MagicMock(return_value=_FakeOk(running_pod))
        monkeypatch.setattr(
            "src.providers.runpod.sdk_adapter.RunPodSDKClient",
            lambda *, api_key: sdk,
        )

        result = monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        )
        assert result is None

    def test_stopped_pod_triggers_wake_up(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = "pod-1"
        monitor._secrets = SimpleNamespace(runpod_api_key="rk-1")

        stopped_pod = {
            "id": "pod-1",
            "desiredStatus": "EXITED" if False else "STOPPED",
            "runtime": {"uptimeInSeconds": 0, "ports": []},
        }
        # Use a status that is non-terminal AND non-ready (not in
        # FAILED/TERMINATED/EXITED; not RUNNING).
        stopped_pod["desiredStatus"] = "STOPPED"

        sdk = MagicMock()
        sdk.get_pod = MagicMock(return_value=_FakeOk(stopped_pod))
        sdk.start_pod = MagicMock(return_value=_FakeOk(None))
        monkeypatch.setattr(
            "src.providers.runpod.sdk_adapter.RunPodSDKClient",
            lambda *, api_key: sdk,
        )

        result = monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        )
        assert result is not None
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_POD_RECOVERED"
        sdk.start_pod.assert_called_once_with(pod_id="pod-1")

    def test_wake_up_failure_surfaces_specific_code(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = "pod-1"
        monitor._secrets = SimpleNamespace(runpod_api_key="rk-1")

        stopped_pod = {
            "id": "pod-1",
            "desiredStatus": "STOPPED",
            "runtime": {"uptimeInSeconds": 0, "ports": []},
        }

        sdk = MagicMock()
        sdk.get_pod = MagicMock(return_value=_FakeOk(stopped_pod))
        sdk.start_pod = MagicMock(return_value=_FakeErr("rate limit"))
        monkeypatch.setattr(
            "src.providers.runpod.sdk_adapter.RunPodSDKClient",
            lambda *, api_key: sdk,
        )

        result = monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        )
        assert result is not None
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_POD_WAKE_FAILED"

    def test_attempt_cap_returns_exhausted_err(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        monitor._provider_name = "runpod"
        monitor._resource_id = "pod-1"
        monitor._secrets = SimpleNamespace(runpod_api_key="rk-1")
        monitor._recovery_attempts = monitor._RECOVERY_ATTEMPT_CAP

        # SDK should never be touched once the cap is hit.
        sdk = MagicMock()
        monkeypatch.setattr(
            "src.providers.runpod.sdk_adapter.RunPodSDKClient",
            lambda *, api_key: sdk,
        )

        result = monitor._recover_pod_if_needed(
            _monitor_mod.JobClientError("boom"),
        )
        assert result is not None
        assert result.is_err()
        assert result.unwrap_err().code == "MONITOR_RECOVERY_EXHAUSTED"
        sdk.get_pod.assert_not_called()


# ---------------------------------------------------------------------------
# Milestone & metric surfacing — give the operator visible signals on a
# short run without flooding monitor.log with full trainer stdout
# ---------------------------------------------------------------------------


def _capture_logger(monkeypatch) -> list[tuple[str, tuple[Any, ...]]]:
    captured: list[tuple[str, tuple[Any, ...]]] = []

    def _info(msg, *args, **_kwargs):
        captured.append((msg, args))

    monkeypatch.setattr(_monitor_mod.logger, "info", _info)
    return captured


def _render(captured: list[tuple[str, tuple[Any, ...]]]) -> list[str]:
    out = []
    for msg, args in captured:
        if not args:
            out.append(msg)
            continue
        try:
            out.append(msg % args)
        except TypeError:
            out.append(msg)
    return out


class TestMilestoneSurfacing:
    def test_first_event_logs_ws_stream_open(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = _capture_logger(monkeypatch)

        monitor._dispatch_event({"kind": "anything", "payload": {}})

        rendered = _render(captured)
        assert any("WS event stream open" in line for line in rendered)

    def test_first_event_logs_only_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = _capture_logger(monkeypatch)

        monitor._dispatch_event({"kind": "x", "payload": {}})
        monitor._dispatch_event({"kind": "y", "payload": {}})
        monitor._dispatch_event({"kind": "z", "payload": {}})

        rendered = _render(captured)
        ws_open_count = sum(1 for line in rendered if "WS event stream open" in line)
        assert ws_open_count == 1

    def test_trainer_spawned_event_logs_pid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = _capture_logger(monkeypatch)

        monitor._dispatch_event({"kind": "trainer_spawned", "payload": {"pid": 4242}})

        rendered = _render(captured)
        assert any("Trainer process started" in line and "4242" in line for line in rendered)

    def test_trainer_spawned_logged_only_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        captured = _capture_logger(monkeypatch)

        monitor._dispatch_event({"kind": "trainer_spawned", "payload": {"pid": 1}})
        monitor._dispatch_event({"kind": "trainer_spawned", "payload": {"pid": 2}})

        rendered = _render(captured)
        spawn_count = sum(1 for line in rendered if "Trainer process started" in line)
        assert spawn_count == 1


class TestTrainerLogIsSilent:
    """Operator policy: training metrics — including HF Trainer's
    canonical dict literals — DO NOT belong in monitor.log. The full
    trainer stdout is captured via the trainer's FileHandler into
    ``training.log`` and reachable through the LogDock + delta-fetch
    REST endpoint. The monitor stream is reserved for control-plane
    signals (ALIVE, milestones, postmortem)."""

    @pytest.mark.parametrize("trainer_line", [
        "{'loss': 2.271, 'grad_norm': 2.6, 'learning_rate': 3.4e-05, 'epoch': 2.62}",
        "{'train_runtime': 30.86, 'train_loss': 2.10, 'epoch': 3.0}",
        "[MC:EXTRACTED] loss=2.10, steps=12, epoch=3.0",
        "Epochs: default, LR: default",
        "Loading checkpoint shards: 100%|##########| 4/4",
        "",
    ])
    def test_no_trainer_line_is_surfaced(
        self, monkeypatch: pytest.MonkeyPatch, trainer_line: str,
    ) -> None:
        monitor = _make_monitor()
        captured = _capture_logger(monkeypatch)

        monitor._dispatch_event({
            "kind": "trainer_log",
            "payload": {"line": trainer_line},
        })

        rendered = _render(captured)
        # No legacy [MONITOR:METRIC] tag, no echo of trainer content.
        assert not any("[MONITOR:METRIC]" in line for line in rendered)
        assert not any(trainer_line and trainer_line in line for line in rendered)

    def test_trainer_log_does_not_block_terminal_event(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # trainer_log dispatch must return None so the watcher loop
        # keeps consuming the next event.
        monitor = _make_monitor()
        _capture_logger(monkeypatch)

        result = monitor._dispatch_event({
            "kind": "trainer_log",
            "payload": {"line": "{'loss': 1.0, 'epoch': 1.0}"},
        })
        assert result is None

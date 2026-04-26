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
    def test_tunnel_and_client_closed_on_success(self) -> None:
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

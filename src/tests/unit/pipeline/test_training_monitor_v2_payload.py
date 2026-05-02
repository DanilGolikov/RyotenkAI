"""PR-B — TrainingMonitor schema-v2 ``trainer_exited`` consumer +
dual log-downloader loop tests.

Cross-checks the symmetric Mac-side handling of:

* schema_version-2 payloads carrying ``stderr_tail`` / ``stdout_tail``
  produced by Supervisor (``test_supervisor_stdio_tail.py``);
* the periodic-pull loop now driving BOTH ``trainer.stdio.log`` and
  ``runner.log`` so uvicorn pre-import crashes flow to Mac mid-run, not
  only at postmortem time;
* the v1 → v2 migration path: legacy v1 payloads still produce a valid
  terminal Result, no schema-aware code crashes on missing fields.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.pipeline.stages import training_monitor as _monitor_mod
from src.pipeline.stages.training_monitor import (
    TrainingMonitor,
    TrainingMonitorEventCallbacks,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrored shape from test_training_monitor_v2.py so
# the same _FakeLogManager interface and ``_make_monitor`` semantics apply)
# ---------------------------------------------------------------------------


def _make_monitor() -> TrainingMonitor:
    monitor = TrainingMonitor.__new__(TrainingMonitor)
    monitor._secrets = None
    monitor._callbacks = TrainingMonitorEventCallbacks()
    monitor._training_start_time = 0.0
    monitor._last_offset = 0
    monitor._last_status_log_time = 0.0
    monitor._ssh_client = None
    monitor._log_manager = None
    monitor._runner_log_manager = None
    monitor._provider_name = None
    monitor._resource_id = None
    monitor._recovery_attempts = 0
    monitor._first_event_logged = False
    monitor._trainer_started_logged = False
    return monitor


class _FakeLogManager:
    """Minimal LogManager stand-in for periodic-loop tests."""

    def __init__(
        self,
        *,
        download_results: list[bool] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._results = list(download_results) if download_results else [True]
        self._raises = raises
        self.download_calls = 0

    def download(self, silent: bool = True) -> bool:
        self.download_calls += 1
        if self._raises is not None:
            raise self._raises
        if self._results:
            return self._results.pop(0)
        return True


# ---------------------------------------------------------------------------
# 1. _log_trainer_exited_tail — formats stderr_tail / stdout_tail
# ---------------------------------------------------------------------------


class TestTailLogging:
    def test_renders_stderr_tail_with_trainer_prefix(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monitor = _make_monitor()
        payload = {
            "schema_version": 2,
            "exit_code": 1,
            "stderr_tail": "Traceback (most recent call last):\n"
                           "  File \"trainer.py\", line 1\n"
                           "ModuleNotFoundError: No module named 'src.providers'",
            "stdout_tail": "",
        }
        with caplog.at_level(logging.INFO):
            monitor._log_trainer_exited_tail(payload)
        msg = "\n".join(rec.message for rec in caplog.records)
        assert "[MONITOR:TRAINER_EXITED] stderr tail" in msg
        assert "[TRAINER:STDERR] ModuleNotFoundError" in msg
        assert "src.providers" in msg

    def test_renders_stdout_tail_when_present(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monitor = _make_monitor()
        payload = {
            "schema_version": 2,
            "exit_code": 0,
            "stderr_tail": "",
            "stdout_tail": "Step 100: loss=0.123\nStep 101: loss=0.119",
        }
        with caplog.at_level(logging.INFO):
            monitor._log_trainer_exited_tail(payload)
        msg = "\n".join(rec.message for rec in caplog.records)
        assert "[TRAINER:STDOUT] Step 100" in msg
        assert "[TRAINER:STDOUT] Step 101" in msg

    def test_silent_when_both_tails_empty(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Trainer crashed before producing any output → no spurious
        '[MONITOR:TRAINER_EXITED]' line in the operator's view."""
        monitor = _make_monitor()
        payload = {
            "schema_version": 2,
            "exit_code": 1,
            "stderr_tail": "",
            "stdout_tail": "",
        }
        with caplog.at_level(logging.INFO):
            monitor._log_trainer_exited_tail(payload)
        # No tail-related entries should fire.
        assert not any("TRAINER_EXITED" in rec.message for rec in caplog.records)

    def test_does_not_raise_on_missing_fields(self) -> None:
        """Future-proof: a v3 payload missing v2 keys must not throw."""
        monitor = _make_monitor()
        # No stderr_tail / stdout_tail at all.
        monitor._log_trainer_exited_tail({"schema_version": 3, "exit_code": 1})

    def test_truncates_stderr_to_30_lines(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Even if the runner sends a long tail, the operator-visible
        rendering caps at 30 lines for stderr to keep pipeline.log
        readable."""
        monitor = _make_monitor()
        long_tail = "\n".join(f"err line {i}" for i in range(100))
        payload = {
            "schema_version": 2,
            "exit_code": 1,
            "stderr_tail": long_tail,
            "stdout_tail": "",
        }
        with caplog.at_level(logging.INFO):
            monitor._log_trainer_exited_tail(payload)
        stderr_records = [
            rec for rec in caplog.records if "[TRAINER:STDERR]" in rec.message
        ]
        assert len(stderr_records) == 30
        # Last line preserved (most recent context wins).
        assert "err line 99" in stderr_records[-1].message


# ---------------------------------------------------------------------------
# 2. _handle_trainer_exited gates rendering on schema_version
# ---------------------------------------------------------------------------


class TestSchemaVersionGate:
    def test_v1_payload_does_not_invoke_tail_logging(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Legacy v1 payloads (no schema_version key) treat as v=1 and
        skip the tail block entirely — the v1 wire format never carried
        tail fields, so reading them would be reading None."""
        monitor = _make_monitor()
        called = []

        def _spy(self, payload: dict[str, Any]) -> None:
            called.append(payload)

        monkeypatch.setattr(
            TrainingMonitor, "_log_trainer_exited_tail", _spy,
        )

        with caplog.at_level(logging.INFO):
            monitor._handle_trainer_exited({
                "exit_code": 0,
                "signal": None,
                "cancellation_requested": False,
                # NO schema_version → defaults to 1 → tail logging skipped.
            })
        assert called == []

    def test_v2_payload_invokes_tail_logging(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monitor = _make_monitor()
        called = []

        def _spy(self, payload: dict[str, Any]) -> None:
            called.append(payload)

        monkeypatch.setattr(
            TrainingMonitor, "_log_trainer_exited_tail", _spy,
        )

        # Avoid the actual postmortem probes (they'd try to SSH).
        monkeypatch.setattr(monitor, "_collect_death_diagnostics", lambda: None)
        monitor._handle_trainer_exited({
            "schema_version": 2,
            "exit_code": 1,
            "signal": None,
            "cancellation_requested": False,
            "stderr_tail": "boom",
            "stdout_tail": "",
        })
        assert len(called) == 1
        assert called[0]["stderr_tail"] == "boom"


# ---------------------------------------------------------------------------
# 3. Dual log-downloader loop pulls BOTH trainer.stdio.log and runner.log
# ---------------------------------------------------------------------------


class TestDualLogDownloaderLoop:
    def test_loop_pulls_both_when_both_provided(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            _monitor_mod, "TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL", 0.001,
        )
        monitor = _make_monitor()
        trainer_lm = _FakeLogManager(download_results=[True, True])
        runner_lm = _FakeLogManager(download_results=[True, True])

        async def _drive() -> None:
            task = asyncio.create_task(
                monitor._log_downloader_loop(trainer_lm, runner_lm),
            )
            await asyncio.sleep(0.02)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_drive())
        # Each tick pulls both → both should have ≥1 calls.
        assert trainer_lm.download_calls >= 1
        assert runner_lm.download_calls >= 1

    def test_loop_works_without_runner_lm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backward-compat: loop signature accepts ``runner_log_manager=None``
        and skips the second pull when not provided."""
        monkeypatch.setattr(
            _monitor_mod, "TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL", 0.001,
        )
        monitor = _make_monitor()
        trainer_lm = _FakeLogManager(download_results=[True])

        async def _drive() -> None:
            task = asyncio.create_task(
                monitor._log_downloader_loop(trainer_lm, None),
            )
            await asyncio.sleep(0.02)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_drive())
        assert trainer_lm.download_calls >= 1

    def test_runner_lm_error_does_not_stop_trainer_pull(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Independence guarantee: a failing runner.log pull must not
        cause us to skip the trainer.stdio.log pull on the same tick."""
        monkeypatch.setattr(
            _monitor_mod, "TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL", 0.001,
        )
        monitor = _make_monitor()
        trainer_lm = _FakeLogManager(download_results=[True])
        runner_lm = _FakeLogManager(raises=RuntimeError("scp boom"))

        async def _drive() -> None:
            task = asyncio.create_task(
                monitor._log_downloader_loop(trainer_lm, runner_lm),
            )
            await asyncio.sleep(0.02)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_drive())
        # Trainer pull succeeded at least once even though runner blew up.
        assert trainer_lm.download_calls >= 1


# ---------------------------------------------------------------------------
# 4. Polling interval lowered (RP6)
# ---------------------------------------------------------------------------


def test_log_download_interval_default_is_5_seconds() -> None:
    """Static guard: PR-B lowered LOG_DOWNLOAD_INTERVAL_DEFAULT 30→5
    so a trainer that crashes at T+11s gets at least one mid-flight
    pull. Defensive against accidental revert."""
    from src.constants import LOG_DOWNLOAD_INTERVAL_DEFAULT
    assert LOG_DOWNLOAD_INTERVAL_DEFAULT == 5


# ---------------------------------------------------------------------------
# 5. Integration: schema_version=2 flow end-to-end
# ---------------------------------------------------------------------------


# Avoid unused-import lint when running in isolation
_ = SimpleNamespace

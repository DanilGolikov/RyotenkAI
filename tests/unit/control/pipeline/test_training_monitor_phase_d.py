"""Phase D — typed-code dispatch in :class:`TrainingMonitor`.

When the runner publishes a ``trainer_exited`` event with
``schema_version=2`` + ``payload_source=trainer_file`` /
``sigkill_heuristic``, the monitor maps ``code`` to a typed exception
(``TrainingOOMError`` / ``InternalError`` / ``TrainingFailedError``)
instead of falling through the legacy exit-code heuristic.

Tests pin the dispatch map and verify the legacy path still works for
pre-D payload shapes (forward-compat: pre-D producers continue to
function while D rolls out across the fleet).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from ryotenkai_control.pipeline.stages.training_monitor import (
    TrainingMonitor,
    TrainingMonitorEventCallbacks,
)
from ryotenkai_shared.errors import (
    InternalError,
    TrainingFailedError,
    TrainingOOMError,
)


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
    monitor._provider = None
    monitor._client = None
    return monitor


def _phase_d_payload(
    code: str,
    message: str,
    *,
    payload_source: str = "trainer_file",
    exit_code: int = 1,
    signal_name: str | None = None,
) -> dict:
    return {
        "exit_code": exit_code,
        "signal": signal_name,
        "cancellation_requested": False,
        "schema_version": 2,
        "code": code,
        "message": message,
        "traceback_summary": None,
        "wall_seconds": 1.5,
        "payload_source": payload_source,
    }


class TestPhaseDDispatch:
    """The monitor maps ``code`` to the right typed exception."""

    def test_training_oom_raises_training_oom_error(self) -> None:
        monitor = _make_monitor()
        # Disable death diagnostics — they require an SSH client; tests
        # care about the dispatch, not the postmortem side-effects.
        monitor._collect_death_diagnostics = lambda: None
        payload = _phase_d_payload("TRAINING_OOM", "VRAM exhausted at step 50")
        with pytest.raises(TrainingOOMError) as exc:
            monitor._handle_trainer_exited(payload)
        assert exc.value.detail == "VRAM exhausted at step 50"
        assert exc.value.context.get("phase_d_typed") is True

    def test_internal_error_raises_internal_error(self) -> None:
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        payload = _phase_d_payload("INTERNAL_ERROR", "trainer crashed in setup")
        with pytest.raises(InternalError) as exc:
            monitor._handle_trainer_exited(payload)
        assert exc.value.detail == "trainer crashed in setup"

    def test_unknown_code_falls_back_to_training_failed(self) -> None:
        """Forward-compat: a producer with an unknown code degrades to
        the generic ``TrainingFailedError`` rather than crashing the
        consumer."""
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        payload = _phase_d_payload(
            "FUTURE_NEW_CODE_NOT_IN_THIS_VERSION",
            "from the future",
        )
        with pytest.raises(TrainingFailedError) as exc:
            monitor._handle_trainer_exited(payload)
        assert exc.value.context.get("legacy_code") == "FUTURE_NEW_CODE_NOT_IN_THIS_VERSION"

    def test_sigkill_heuristic_routed_via_typed_dispatch(self) -> None:
        """``payload_source="sigkill_heuristic"`` flows through the same
        typed dispatch as ``"trainer_file"`` — the runner synthesised
        a TRAINING_OOM code we should honour."""
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        payload = _phase_d_payload(
            "TRAINING_OOM",
            "Trainer killed by signal SIGKILL — likely OOM (exit_code=137)",
            payload_source="sigkill_heuristic",
            exit_code=137,
            signal_name="SIGKILL",
        )
        with pytest.raises(TrainingOOMError):
            monitor._handle_trainer_exited(payload)


class TestLegacyFallback:
    """Pre-Phase-D payloads (no ``payload_source``, no ``code``) still work."""

    def test_legacy_payload_with_exit_zero_succeeds(self) -> None:
        monitor = _make_monitor()
        result = monitor._handle_trainer_exited({
            "exit_code": 0, "signal": None, "cancellation_requested": False,
        })
        assert result["status"] == "completed"

    def test_legacy_payload_nonzero_raises_training_failed(self) -> None:
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        with pytest.raises(TrainingFailedError):
            monitor._handle_trainer_exited({
                "exit_code": 1, "signal": None, "cancellation_requested": False,
            })

    def test_legacy_payload_with_missing_source_treated_as_legacy(self) -> None:
        """Payload has ``code`` but no ``payload_source`` — the consumer
        treats it as legacy (defensive: only the supervisor's own
        synthesised payload carries ``payload_source``)."""
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        # Note: no payload_source key.
        with pytest.raises(TrainingFailedError):
            monitor._handle_trainer_exited({
                "exit_code": 1, "signal": None,
                "cancellation_requested": False,
                "code": "TRAINING_OOM",
                "message": "shouldn't be trusted",
            })


class TestPhaseDWallSeconds:
    """``wall_seconds`` from the trainer flows through to ``duration``."""

    def test_wall_seconds_overrides_when_greater(self) -> None:
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        monitor._training_start_time = 1_000_000.0  # so time.time() - start ≈ 0
        payload = _phase_d_payload(
            "TRAINING_FAILED", "x", payload_source="trainer_file",
        )
        payload["wall_seconds"] = 99.0
        with pytest.raises(TrainingFailedError) as exc:
            monitor._handle_trainer_exited(payload)
        assert exc.value.context["duration_seconds"] >= 99.0

    def test_wall_seconds_does_not_go_backwards(self) -> None:
        """If the supervisor's local duration is bigger than the
        trainer's wall_seconds, keep the larger one."""
        import time as _time
        monitor = _make_monitor()
        monitor._collect_death_diagnostics = lambda: None
        # Pretend training started 100s ago locally.
        monitor._training_start_time = _time.time() - 100.0
        payload = _phase_d_payload(
            "TRAINING_FAILED", "x", payload_source="trainer_file",
        )
        payload["wall_seconds"] = 1.0  # smaller than local duration
        with pytest.raises(TrainingFailedError) as exc:
            monitor._handle_trainer_exited(payload)
        # Must keep local duration (~100s), not 1.0.
        assert exc.value.context["duration_seconds"] >= 50.0

"""Unit tests: pod-lifecycle event types (round-trip + invariants).

Six concrete event classes. Per-class TestPositive checks round-trip
through the codec; TestNegative checks payload extra=forbid + missing
required; TestInvariants pins the type Literal default and severity
default so a regression in either flips the union dispatch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.pod_lifecycle import (
    JobSubmittedEvent,
    JobSubmittedPayload,
    PluginsUnpackedEvent,
    PluginsUnpackedPayload,
    RunnerShutdownEvent,
    RunnerShutdownPayload,
    RunnerStartedEvent,
    RunnerStartedPayload,
    StopRequestedEvent,
    StopRequestedPayload,
    TrainerExitedEvent,
    TrainerExitedPayload,
    TrainerSpawnedEvent,
    TrainerSpawnedPayload,
    TrainerSpawnFailedEvent,
    TrainerSpawnFailedPayload,
)


def _runner_started() -> RunnerStartedEvent:
    return RunnerStartedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=0,
        payload=RunnerStartedPayload(version="1.0", git_sha="abc", gpu_count=2),
    )


def _runner_shutdown() -> RunnerShutdownEvent:
    return RunnerShutdownEvent(
        source="pod://r/runner",
        run_id="r",
        offset=1,
        payload=RunnerShutdownPayload(reason="signal", graceful=True),
    )


def _job_submitted() -> JobSubmittedEvent:
    return JobSubmittedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=2,
        payload=JobSubmittedPayload(
            job_id="j", config_hash="h", image_tag="t",
        ),
    )


def _trainer_spawned() -> TrainerSpawnedEvent:
    return TrainerSpawnedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=3,
        payload=TrainerSpawnedPayload(pid=123, cmdline="cmd", cwd="/tmp"),
    )


def _trainer_spawn_failed() -> TrainerSpawnFailedEvent:
    return TrainerSpawnFailedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=4,
        payload=TrainerSpawnFailedPayload(reason="enoexec", exit_code=126),
    )


def _trainer_exited() -> TrainerExitedEvent:
    return TrainerExitedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=5,
        payload=TrainerExitedPayload(exit_code=0, signal=None, duration_s=12.5),
    )


_ALL = [
    _runner_started,
    _runner_shutdown,
    _job_submitted,
    _trainer_spawned,
    _trainer_spawn_failed,
    _trainer_exited,
]


def _stop_requested() -> StopRequestedEvent:
    return StopRequestedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=5,
        payload=StopRequestedPayload(grace_seconds=30.0),
    )


def _plugins_unpacked() -> PluginsUnpackedEvent:
    return PluginsUnpackedEvent(
        source="pod://r/plugin_unpacker",
        run_id="r",
        offset=6,
        payload=PluginsUnpackedPayload(
            installed=["a", "b"], skipped=["c"], total_bytes=1024,
        ),
    )


_ALL.extend([_stop_requested, _plugins_unpacked])


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_runner_started_payload_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunnerStartedPayload(version="1", git_sha="a", gpu_count=1, extra="x")  # type: ignore[call-arg]

    def test_trainer_spawned_payload_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            TrainerSpawnedPayload(pid=1, cmdline="x")  # type: ignore[call-arg]


class TestInvariants:
    def test_runner_started_pins_kind_and_severity(self) -> None:
        event = _runner_started()
        assert event.kind == "ryotenkai.pod.lifecycle.runner_started"
        assert event.severity == "info"

    def test_trainer_spawn_failed_severity_is_error(self) -> None:
        # Failure cases must carry severity=error per taxonomy.
        assert _trainer_spawn_failed().severity == "error"

    def test_stop_requested_pins_kind_and_severity(self) -> None:
        ev = _stop_requested()
        assert ev.kind == "ryotenkai.pod.lifecycle.stop_requested"
        assert ev.severity == "info"

    def test_plugins_unpacked_pins_kind_and_severity(self) -> None:
        ev = _plugins_unpacked()
        assert ev.kind == "ryotenkai.pod.lifecycle.plugins_unpacked"
        assert ev.severity == "info"

    def test_schema_version_pinned_to_1(self) -> None:
        for factory in _ALL:
            assert factory().schema_version == 1

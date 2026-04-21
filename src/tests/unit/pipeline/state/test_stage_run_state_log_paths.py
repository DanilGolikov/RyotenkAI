"""Tests for StageRunState.log_paths serialization + backward compat."""

from __future__ import annotations

import pytest

from src.pipeline.state import StageRunState

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Positive — to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------

def test_to_dict_includes_log_paths() -> None:
    state = StageRunState(stage_name="s1", log_paths={"stage": "logs/s1.log"})
    assert state.to_dict()["log_paths"] == {"stage": "logs/s1.log"}


def test_from_dict_restores_log_paths() -> None:
    raw = {"stage_name": "s1", "status": "completed", "log_paths": {"stage": "logs/s1.log"}}
    state = StageRunState.from_dict(raw)
    assert state.log_paths == {"stage": "logs/s1.log"}


def test_roundtrip_preserves_log_paths() -> None:
    original = StageRunState(
        stage_name="training_monitor",
        log_paths={"stage": "logs/training_monitor.log", "remote_training": "logs/training.log"},
    )
    restored = StageRunState.from_dict(original.to_dict())
    assert restored.log_paths == original.log_paths


def test_to_dict_returns_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """to_dict() output must not alias internal state — avoids external mutation bleed."""
    state = StageRunState(stage_name="s1", log_paths={"stage": "logs/s1.log"})
    out = state.to_dict()
    out["log_paths"]["injected"] = "logs/evil.log"
    assert state.log_paths == {"stage": "logs/s1.log"}


# ---------------------------------------------------------------------------
# Regression — old state files (pre-log_paths) must still load
# ---------------------------------------------------------------------------

def test_from_dict_without_log_paths_defaults_to_empty() -> None:
    raw = {"stage_name": "legacy_stage", "status": "completed"}
    state = StageRunState.from_dict(raw)
    assert state.log_paths == {}


def test_default_log_paths_is_empty_dict() -> None:
    state = StageRunState(stage_name="s1")
    assert state.log_paths == {}


# ---------------------------------------------------------------------------
# Boundary — malformed / unexpected log_paths values
# ---------------------------------------------------------------------------

def test_from_dict_rejects_non_dict_log_paths() -> None:
    """If pipeline_state.json is corrupted (log_paths is a list), load cleanly as {}."""
    state = StageRunState.from_dict({"stage_name": "s1", "log_paths": ["oops"]})
    assert state.log_paths == {}


def test_from_dict_accepts_null_log_paths() -> None:
    state = StageRunState.from_dict({"stage_name": "s1", "log_paths": None})
    assert state.log_paths == {}


def test_from_dict_coerces_numeric_values_to_str() -> None:
    """Defensive: ints/floats are coerced; arbitrary objects are dropped."""
    state = StageRunState.from_dict({
        "stage_name": "s1",
        "log_paths": {"a": "logs/a.log", "b": 42, "c": 1.5, "d": ["list"]},
    })
    assert state.log_paths == {"a": "logs/a.log", "b": "42", "c": "1.5"}


def test_from_dict_stringifies_keys() -> None:
    state = StageRunState.from_dict({"stage_name": "s1", "log_paths": {1: "logs/a.log"}})
    assert state.log_paths == {"1": "logs/a.log"}


# ---------------------------------------------------------------------------
# Combinatorial — all status × log_paths combinations survive roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status",
    [
        StageRunState.STATUS_PENDING,
        StageRunState.STATUS_RUNNING,
        StageRunState.STATUS_COMPLETED,
        StageRunState.STATUS_FAILED,
        StageRunState.STATUS_INTERRUPTED,
        StageRunState.STATUS_SKIPPED,
    ],
)
@pytest.mark.parametrize(
    "log_paths",
    [
        {},
        {"stage": "logs/s1.log"},
        {"stage": "logs/s1.log", "remote_training": "logs/training.log"},
    ],
)
def test_status_log_paths_roundtrip(status: str, log_paths: dict[str, str]) -> None:
    state = StageRunState(stage_name="s1", status=status, log_paths=dict(log_paths))
    restored = StageRunState.from_dict(state.to_dict())
    assert restored.status == status
    assert restored.log_paths == log_paths

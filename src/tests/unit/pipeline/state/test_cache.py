"""Tests for the mtime-keyed ``PipelineState`` cache used by the web API."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.pipeline.state import PipelineStateLoadError, PipelineStateStore
from src.pipeline.state.cache import (
    cache_stats,
    clear_cache,
    load_state_snapshot,
)


def _init_run(run_dir: Path, *, logical_run_id: str = "run_1") -> PipelineStateStore:
    store = PipelineStateStore(run_dir)
    store.init_state(
        logical_run_id=logical_run_id,
        config_path="/tmp/config.yaml",
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
    )
    return store


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_cache()
    yield
    clear_cache()


def test_first_load_is_a_miss(tmp_path: Path) -> None:
    _init_run(tmp_path / "run_a")
    before = cache_stats()
    snap = load_state_snapshot(tmp_path / "run_a")

    assert snap.state.logical_run_id == "run_1"
    assert snap.mtime_ns > 0

    stats = cache_stats()
    assert stats["misses"] == before["misses"] + 1
    assert stats["hits"] == before["hits"]


def test_second_load_without_write_is_a_hit(tmp_path: Path) -> None:
    _init_run(tmp_path / "run_a")
    first = load_state_snapshot(tmp_path / "run_a")
    second = load_state_snapshot(tmp_path / "run_a")

    assert first.mtime_ns == second.mtime_ns
    # Same cached instance — re-parsing JSON would give us a new object.
    assert first.state is second.state

    stats = cache_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] == 1


def test_save_invalidates_cache_via_mtime(tmp_path: Path) -> None:
    store = _init_run(tmp_path / "run_a")
    first = load_state_snapshot(tmp_path / "run_a")

    # Bump mtime to simulate the orchestrator writing a new state. ``utime``
    # is enough — the cache only cares about mtime_ns, not file content.
    future_ns = first.mtime_ns + 1_000_000_000  # +1s
    os.utime(store.state_path, ns=(future_ns, future_ns))

    second = load_state_snapshot(tmp_path / "run_a")
    assert second.mtime_ns > first.mtime_ns
    # Must have re-loaded — fresh object, not the cached one.
    assert second.state is not first.state


def test_missing_state_raises(tmp_path: Path) -> None:
    with pytest.raises(PipelineStateLoadError):
        load_state_snapshot(tmp_path / "nonexistent")


def test_cache_survives_across_distinct_runs(tmp_path: Path) -> None:
    _init_run(tmp_path / "run_a", logical_run_id="run_a")
    _init_run(tmp_path / "run_b", logical_run_id="run_b")

    snap_a = load_state_snapshot(tmp_path / "run_a")
    snap_b = load_state_snapshot(tmp_path / "run_b")

    assert snap_a.state.logical_run_id == "run_a"
    assert snap_b.state.logical_run_id == "run_b"
    # Two independent cache entries.
    assert cache_stats()["entries"] >= 2


def test_clear_empties_cache(tmp_path: Path) -> None:
    _init_run(tmp_path / "run_a")
    load_state_snapshot(tmp_path / "run_a")
    assert cache_stats()["entries"] == 1
    clear_cache()
    assert cache_stats()["entries"] == 0

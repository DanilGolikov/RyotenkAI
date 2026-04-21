"""Comprehensive tests for lineage_manager (the 5 lineage mutation moments).

Seven categories:
1. Positive     — each function's happy path
2. Negative     — unknown start stage, malformed lineage values
3. Boundary     — first/last stage as start, empty lineage, empty stage list
4. Invariants   — pure functions: no input mutation, deterministic
5. Dep errors   — sync_root_from_stage raising, mark_stage_skipped raising
6. Regressions  — behaviour preserved after extraction from transitioner
7. Combinatorial — (start_stage × lineage entries × enabled_stages)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.pipeline.state.lineage_manager import (
    after_stage_completed,
    after_stage_failed,
    after_stage_skipped,
    invalidate_from,
    restore_reused,
)
from src.pipeline.state.models import (
    PipelineAttemptState,
    StageLineageRef,
    StageRunState,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


_STAGES = ["s0", "s1", "s2", "s3", "s4"]


def _make_attempt() -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id="a1",
        attempt_no=1,
        runtime_name="rt",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_RUNNING,
        started_at="2026-04-21T00:00:00+00:00",
    )


def _ref(name: str, outputs: dict[str, Any] | None = None) -> StageLineageRef:
    return StageLineageRef(attempt_id="src-a", stage_name=name, outputs=outputs or {f"{name}_key": "v"})


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_invalidate_from_removes_from_and_onward(self) -> None:
        lineage = {name: _ref(name) for name in _STAGES}
        new = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name="s2")
        assert set(new.keys()) == {"s0", "s1"}

    def test_restore_reused_materialises_pre_start_stages(self) -> None:
        attempt = _make_attempt()
        lineage = {"s0": _ref("s0", {"k": "v0"}), "s1": _ref("s1", {"k": "v1"})}
        context: dict = {}
        syncs: list[tuple[str, dict]] = []

        def sync(ctx: dict, name: str, outputs: dict) -> None:
            syncs.append((name, outputs))

        restore_reused(
            attempt=attempt,
            lineage=lineage,
            stage_names=_STAGES,
            start_stage_name="s2",
            enabled_stage_names=_STAGES,
            context=context,
            sync_root_from_stage=sync,
        )
        assert "s0" in context and "s1" in context
        assert syncs == [("s0", {"k": "v0"}), ("s1", {"k": "v1"})]
        assert attempt.stage_runs["s0"].execution_mode == StageRunState.MODE_REUSED

    def test_after_stage_completed_adds_entry(self) -> None:
        lineage: dict = {}
        new = after_stage_completed(
            lineage, stage_name="s1", attempt_id="a1", outputs={"foo": "bar"}
        )
        assert "s1" in new
        assert new["s1"].outputs == {"foo": "bar"}
        assert new["s1"].attempt_id == "a1"

    def test_after_stage_failed_drops_entry(self) -> None:
        lineage = {"s1": _ref("s1")}
        new = after_stage_failed(lineage, stage_name="s1", attempt_id="a1")
        assert "s1" not in new

    def test_after_stage_skipped_drops_entry(self) -> None:
        lineage = {"s1": _ref("s1")}
        new = after_stage_skipped(lineage, stage_name="s1", attempt_id="a1")
        assert "s1" not in new


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_invalidate_from_unknown_start_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage name"):
            invalidate_from(lineage={}, stage_names=_STAGES, start_stage_name="missing")

    def test_restore_reused_unknown_start_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage name"):
            restore_reused(
                attempt=_make_attempt(),
                lineage={},
                stage_names=_STAGES,
                start_stage_name="missing",
                enabled_stage_names=_STAGES,
                context={},
                sync_root_from_stage=lambda *_: None,
            )


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_invalidate_from_first_stage_drops_all(self) -> None:
        lineage = {name: _ref(name) for name in _STAGES}
        new = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name="s0")
        assert new == {}

    def test_invalidate_from_last_stage_drops_only_it(self) -> None:
        lineage = {name: _ref(name) for name in _STAGES}
        new = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name="s4")
        assert set(new.keys()) == {"s0", "s1", "s2", "s3"}

    def test_restore_reused_empty_lineage_noop(self) -> None:
        attempt = _make_attempt()
        ctx: dict = {}
        restore_reused(
            attempt=attempt,
            lineage={},
            stage_names=_STAGES,
            start_stage_name="s2",
            enabled_stage_names=_STAGES,
            context=ctx,
            sync_root_from_stage=lambda *_: None,
        )
        assert attempt.stage_runs == {}
        assert ctx == {}

    def test_restore_reused_disabled_stage_marked_skipped(self) -> None:
        attempt = _make_attempt()
        lineage = {"s0": _ref("s0", {"k": "v"})}
        restore_reused(
            attempt=attempt,
            lineage=lineage,
            stage_names=_STAGES,
            start_stage_name="s2",
            enabled_stage_names=["s1", "s2"],  # s0 disabled
            context={},
            sync_root_from_stage=lambda *_: None,
        )
        assert attempt.stage_runs["s0"].status == StageRunState.STATUS_SKIPPED
        assert attempt.stage_runs["s0"].skip_reason == "disabled_by_config"

    def test_after_stage_completed_on_missing_stage_still_adds(self) -> None:
        """Adding a brand new entry works; lineage was empty before."""
        new = after_stage_completed({}, stage_name="s1", attempt_id="a", outputs={"x": 1})
        assert new["s1"].outputs == {"x": 1}

    def test_after_stage_failed_on_missing_noop(self) -> None:
        """Removing a non-existent entry is a no-op, not an error."""
        new = after_stage_failed({"s2": _ref("s2")}, stage_name="s1", attempt_id="a")
        assert "s2" in new


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_invariant_invalidate_does_not_mutate_input(self) -> None:
        """INVARIANT: pure — input lineage dict stays unchanged."""
        original = {name: _ref(name) for name in _STAGES}
        snapshot_keys = set(original.keys())
        _ = invalidate_from(lineage=original, stage_names=_STAGES, start_stage_name="s2")
        assert set(original.keys()) == snapshot_keys

    def test_invariant_after_stage_returns_new_dict(self) -> None:
        """INVARIANT: all after_stage_* return a NEW dict, not mutated input."""
        original = {"s0": _ref("s0")}
        for fn in (after_stage_failed, after_stage_skipped):
            new = fn(original, stage_name="s0", attempt_id="a")
            assert new is not original
        new = after_stage_completed(original, stage_name="s1", attempt_id="a", outputs={})
        assert new is not original

    def test_invariant_deterministic_for_identical_input(self) -> None:
        """INVARIANT: same inputs → same outputs (no hidden state)."""
        lineage = {name: _ref(name) for name in _STAGES}
        a = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name="s2")
        b = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name="s2")
        assert a == b
        assert a is not b  # defensive copy

    def test_invariant_restore_reused_stops_at_start(self) -> None:
        """INVARIANT: only stages BEFORE start_stage are restored."""
        attempt = _make_attempt()
        lineage = {name: _ref(name) for name in _STAGES}
        restore_reused(
            attempt=attempt,
            lineage=lineage,
            stage_names=_STAGES,
            start_stage_name="s2",
            enabled_stage_names=_STAGES,
            context={},
            sync_root_from_stage=lambda *_: None,
        )
        assert set(attempt.stage_runs.keys()) == {"s0", "s1"}


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_sync_callback_exception_propagates(self) -> None:
        """If the injected sync_root callback raises, it bubbles up — caller decides."""
        attempt = _make_attempt()
        lineage = {"s0": _ref("s0")}

        def failing_sync(*_: Any) -> None:
            raise RuntimeError("sync broke")

        with pytest.raises(RuntimeError, match="sync broke"):
            restore_reused(
                attempt=attempt,
                lineage=lineage,
                stage_names=_STAGES,
                start_stage_name="s2",
                enabled_stage_names=_STAGES,
                context={},
                sync_root_from_stage=failing_sync,
            )


# =============================================================================
# 6. REGRESSIONS — behaviour preserved after transitioner → lineage_manager move
# =============================================================================


class TestRegressions:
    def test_regression_invalidate_from_equivalent_to_old_transitioner(self) -> None:
        """REGRESSION: ``invalidate_from`` has same contract as the pre-move
        ``transitioner.invalidate_lineage_from`` — same kw-args, same semantics."""
        lineage = {name: _ref(name) for name in _STAGES}
        new = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name="s2")
        # Original contract: drops start_stage_name + everything after
        assert "s2" not in new
        assert "s3" not in new
        assert "s4" not in new
        assert "s0" in new and "s1" in new

    def test_regression_after_stage_completed_preserves_stageref_attrs(self) -> None:
        """REGRESSION: StageLineageRef.attempt_id / stage_name / outputs preserved."""
        new = after_stage_completed(
            {},
            stage_name="s1",
            attempt_id="attempt-xyz",
            outputs={"preserved": True},
        )
        ref = new["s1"]
        assert ref.stage_name == "s1"
        assert ref.attempt_id == "attempt-xyz"
        assert ref.outputs == {"preserved": True}

    def test_regression_after_stage_skipped_and_failed_semantically_same(self) -> None:
        """REGRESSION: skipped and failed both remove the entry — only the
        semantic name (call site) differs. Sanity check that the two wrappers
        produce identical output so future divergence is intentional."""
        lineage = {"s0": _ref("s0")}
        assert after_stage_skipped(
            lineage, stage_name="s0", attempt_id="a"
        ) == after_stage_failed(lineage, stage_name="s0", attempt_id="a")


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


@pytest.mark.parametrize("start_idx", range(len(_STAGES)))
def test_combinatorial_invalidate_cuts_at_every_start(start_idx: int) -> None:
    """For every possible start stage, invalidate cuts ≥ start_idx exactly."""
    lineage = {name: _ref(name) for name in _STAGES}
    start = _STAGES[start_idx]
    new = invalidate_from(lineage=lineage, stage_names=_STAGES, start_stage_name=start)
    assert set(new.keys()) == set(_STAGES[:start_idx])


@pytest.mark.parametrize("n_entries", [0, 1, 3, 5])
@pytest.mark.parametrize("start_idx", [0, 2, 4])
def test_combinatorial_restore_reused_pre_loop(n_entries: int, start_idx: int) -> None:
    """Matrix of (existing entries × start stage) — restore materialises exactly
    the entries strictly before start_idx.
    """
    attempt = _make_attempt()
    lineage = {name: _ref(name) for name in _STAGES[:n_entries]}
    ctx: dict = {}
    syncs: list[str] = []
    restore_reused(
        attempt=attempt,
        lineage=lineage,
        stage_names=_STAGES,
        start_stage_name=_STAGES[start_idx],
        enabled_stage_names=_STAGES,
        context=ctx,
        sync_root_from_stage=lambda _c, n, _o: syncs.append(n),
    )
    expected = [name for name in _STAGES[:start_idx] if name in lineage]
    assert syncs == expected


@pytest.mark.parametrize("entry_count", [0, 1, 3])
def test_combinatorial_after_stage_completed_chain(entry_count: int) -> None:
    """Sequential after_stage_completed accumulates entries."""
    lineage: dict = {}
    for i in range(entry_count):
        lineage = after_stage_completed(
            lineage,
            stage_name=f"s{i}",
            attempt_id="a",
            outputs={"idx": i},
        )
    assert len(lineage) == entry_count


def test_combinatorial_completed_then_failed_leaves_no_trace() -> None:
    """Compound: completed then failed = absent."""
    lineage = after_stage_completed(
        {}, stage_name="s1", attempt_id="a", outputs={"x": 1}
    )
    lineage = after_stage_failed(lineage, stage_name="s1", attempt_id="a")
    assert "s1" not in lineage

"""Comprehensive tests for PipelineContext — the typed value object.

Seven categories:
1. Positive     — dict-compat + typed accessors + fork happy path
2. Negative     — missing keys raise (no silent defaults for required)
3. Boundary     — empty context, empty fork, non-ASCII keys
4. Invariants   — isinstance(ctx, dict), fork independence, type fidelity
5. Dep errors   — underlying RunContext malformed / missing
6. Regressions  — pre-refactor orchestrator behaviour preserved
7. Combinatorial — fork × various attempt configurations
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.pipeline.context.pipeline_context import PipelineContext
from src.pipeline.stages.constants import PipelineContextKeys


def _fake_run_ctx() -> Any:
    """A lightweight stand-in for RunContext — only attributes accessed in tests."""
    return SimpleNamespace(name="run-0001", logical_run_id="logical-xyz")


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_empty_factory(self) -> None:
        ctx = PipelineContext.empty()
        assert isinstance(ctx, PipelineContext)
        assert isinstance(ctx, dict)
        assert len(ctx) == 0

    def test_from_mapping(self) -> None:
        ctx = PipelineContext.from_mapping({"a": 1, "b": 2})
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_from_mapping_none(self) -> None:
        ctx = PipelineContext.from_mapping(None)
        assert len(ctx) == 0

    def test_dict_operations_work(self) -> None:
        ctx = PipelineContext()
        ctx["key"] = "value"
        assert ctx["key"] == "value"
        assert "key" in ctx
        assert list(ctx) == ["key"]
        assert ctx.get("missing") is None
        assert ctx.get("missing", "dflt") == "dflt"
        ctx.update({"other": 2})
        assert ctx["other"] == 2

    def test_typed_accessors_read(self) -> None:
        run = _fake_run_ctx()
        ctx = PipelineContext(
            {
                PipelineContextKeys.RUN: run,
                PipelineContextKeys.CONFIG_PATH: "/tmp/cfg.yaml",
                PipelineContextKeys.LOGICAL_RUN_ID: "L-1",
                PipelineContextKeys.ATTEMPT_ID: "A-1",
                PipelineContextKeys.ATTEMPT_NO: 3,
                PipelineContextKeys.RUN_DIRECTORY: "/tmp/runs/L-1",
                PipelineContextKeys.ATTEMPT_DIRECTORY: "/tmp/runs/L-1/A-3",
                PipelineContextKeys.FORCED_STAGES: {"Inference Deployer"},
            }
        )
        assert ctx.run_ctx is run
        assert ctx.config_path == "/tmp/cfg.yaml"
        assert ctx.logical_run_id == "L-1"
        assert ctx.attempt_id == "A-1"
        assert ctx.attempt_no == 3
        assert ctx.run_directory == "/tmp/runs/L-1"
        assert ctx.attempt_directory == "/tmp/runs/L-1/A-3"
        assert ctx.forced_stages == {"Inference Deployer"}

    def test_fork_happy_path(self) -> None:
        parent = PipelineContext(
            {
                PipelineContextKeys.CONFIG_PATH: "/tmp/cfg.yaml",
                PipelineContextKeys.RUN: _fake_run_ctx(),
            }
        )
        forked = parent.fork(
            attempt_id="A-2",
            attempt_no=2,
            attempt_directory=Path("/tmp/a-2"),
            logical_run_id="L-1",
            run_directory=Path("/tmp/L-1"),
            forced_stages={"Model Evaluator"},
        )
        assert forked.attempt_id == "A-2"
        assert forked.attempt_no == 2
        assert forked.attempt_directory == "/tmp/a-2"
        assert forked.logical_run_id == "L-1"
        assert forked.forced_stages == {"Model Evaluator"}
        # Inherits run-scoped keys
        assert forked.config_path == "/tmp/cfg.yaml"


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_run_ctx_missing_raises(self) -> None:
        """run_ctx is a required seed key — missing is a programming error."""
        ctx = PipelineContext()
        with pytest.raises(KeyError):
            _ = ctx.run_ctx

    def test_config_path_missing_raises(self) -> None:
        ctx = PipelineContext()
        with pytest.raises(KeyError):
            _ = ctx.config_path

    def test_optional_attempt_keys_return_none(self) -> None:
        """Attempt-scoped keys are None before LaunchPreparator forks — not an error."""
        ctx = PipelineContext({PipelineContextKeys.RUN: _fake_run_ctx()})
        assert ctx.attempt_id is None
        assert ctx.attempt_no is None
        assert ctx.logical_run_id is None
        assert ctx.run_directory is None
        assert ctx.attempt_directory is None
        assert ctx.forced_stages == set()

    def test_forced_stages_malformed_returns_empty(self) -> None:
        """Defensive: non-set FORCED_STAGES → empty set (no crash, no false positives)."""
        ctx = PipelineContext({PipelineContextKeys.FORCED_STAGES: ["not", "a", "set"]})
        assert ctx.forced_stages == set()
        ctx2 = PipelineContext({PipelineContextKeys.FORCED_STAGES: "string"})
        assert ctx2.forced_stages == set()


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_fork_with_no_inherited_keys(self) -> None:
        """fork from empty context: attempt keys present, nothing else."""
        forked = PipelineContext().fork(
            attempt_id="A-1",
            attempt_no=1,
            attempt_directory="/tmp",
            logical_run_id="L-1",
            run_directory="/tmp",
            forced_stages=None,
        )
        assert forked.attempt_id == "A-1"
        assert forked.forced_stages == set()  # None → empty set

    def test_fork_with_empty_forced_stages(self) -> None:
        ctx = PipelineContext()
        forked = ctx.fork(
            attempt_id="A-1",
            attempt_no=1,
            attempt_directory="/x",
            logical_run_id="L",
            run_directory="/x",
            forced_stages=set(),
        )
        assert forked.forced_stages == set()

    def test_fork_with_path_vs_str_directories(self) -> None:
        """Both Path and str inputs normalised to str in storage."""
        ctx = PipelineContext()
        fp = ctx.fork(
            attempt_id="A",
            attempt_no=1,
            attempt_directory=Path("/x/y"),
            logical_run_id="L",
            run_directory="/z",  # str
            forced_stages=None,
        )
        assert fp.attempt_directory == "/x/y"
        assert fp.run_directory == "/z"

    def test_attempt_no_zero_is_valid(self) -> None:
        """attempt_no=0 is a legitimate value (first attempt), not an error."""
        ctx = PipelineContext({PipelineContextKeys.ATTEMPT_NO: 0})
        assert ctx.attempt_no == 0


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_invariant_is_dict_instance(self) -> None:
        """INVARIANT: PipelineContext IS a dict — every isinstance(x, dict) passes.

        This is the backward-compat promise that lets every existing stage
        signature ``context: dict[str, Any]`` accept our value object.
        """
        ctx = PipelineContext()
        assert isinstance(ctx, dict)

    def test_invariant_fork_is_independent(self) -> None:
        """INVARIANT: mutations on fork do NOT leak to parent, and vice versa."""
        parent = PipelineContext(
            {PipelineContextKeys.RUN: _fake_run_ctx(), "shared": "initial"}
        )
        forked = parent.fork(
            attempt_id="A-1",
            attempt_no=1,
            attempt_directory="/x",
            logical_run_id="L",
            run_directory="/x",
            forced_stages=None,
        )
        forked["new_key"] = "child"
        forked["shared"] = "mutated"
        assert "new_key" not in parent
        assert parent["shared"] == "initial"

    def test_invariant_fork_deep_enough_for_stored_sets(self) -> None:
        """forced_stages on fork is a new set object, not aliased."""
        parent = PipelineContext({PipelineContextKeys.FORCED_STAGES: {"orig"}})
        forked = parent.fork(
            attempt_id="A",
            attempt_no=1,
            attempt_directory="/x",
            logical_run_id="L",
            run_directory="/x",
            forced_stages={"new"},
        )
        forked[PipelineContextKeys.FORCED_STAGES].add("added-to-fork")
        # parent's set must remain {"orig"}
        assert parent[PipelineContextKeys.FORCED_STAGES] == {"orig"}

    def test_invariant_typed_accessor_reflects_live_mutation(self) -> None:
        """INVARIANT: properties don't cache — writes via [] are visible immediately."""
        ctx = PipelineContext()
        ctx[PipelineContextKeys.ATTEMPT_ID] = "A-1"
        assert ctx.attempt_id == "A-1"
        ctx[PipelineContextKeys.ATTEMPT_ID] = "A-2"
        assert ctx.attempt_id == "A-2"

    def test_invariant_forced_stages_getter_returns_copy(self) -> None:
        """INVARIANT: ctx.forced_stages is a copy — mutating it doesn't affect storage."""
        ctx = PipelineContext({PipelineContextKeys.FORCED_STAGES: {"X"}})
        snapshot = ctx.forced_stages
        snapshot.add("Y")
        # Stored set is still just {"X"}
        assert ctx[PipelineContextKeys.FORCED_STAGES] == {"X"}


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_run_ctx_returns_any_stored_object(self) -> None:
        """Property returns whatever is stored — no type coercion.

        Doesn't try to validate RunContext shape — caller is responsible.
        """
        ctx = PipelineContext({PipelineContextKeys.RUN: "not a real run_ctx"})
        # No crash; returns what was stored
        assert ctx.run_ctx == "not a real run_ctx"

    def test_attempt_no_coerces_to_int(self) -> None:
        """int() coerces str to int; propagate TypeError for bad data."""
        ctx = PipelineContext({PipelineContextKeys.ATTEMPT_NO: "5"})
        assert ctx.attempt_no == 5
        ctx[PipelineContextKeys.ATTEMPT_NO] = "not-a-number"
        with pytest.raises(ValueError):
            _ = ctx.attempt_no


# =============================================================================
# 6. REGRESSIONS
# =============================================================================


class TestRegressions:
    def test_regression_passes_isinstance_dict_check(self) -> None:
        """REGRESSION: stages do ``isinstance(context, dict)`` checks.

        Before: context was raw dict. After: it's PipelineContext(dict).
        isinstance MUST still return True.
        """
        ctx = PipelineContext()
        assert isinstance(ctx, dict)

    def test_regression_dict_unpack_works(self) -> None:
        """REGRESSION: stages often do ``{**context, new_key: ...}`` — must still work."""
        ctx = PipelineContext({"a": 1, "b": 2})
        merged = {**ctx, "c": 3}
        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_regression_stage_receives_and_reads_typed_keys(self) -> None:
        """REGRESSION: stages read via ``context.get(PipelineContextKeys.RUN)`` — works."""
        run = _fake_run_ctx()
        ctx = PipelineContext({PipelineContextKeys.RUN: run})
        # Simulates what GPUDeployer / InferenceDeployer do
        assert ctx.get(PipelineContextKeys.RUN) is run

    def test_regression_preserves_stage_nested_dicts(self) -> None:
        """REGRESSION: stages store sub-dicts under StageNames.*; PipelineContext
        must pass them through untouched.
        """
        ctx = PipelineContext()
        ctx["Dataset Validator"] = {"sample_count": 1000, "metrics": {"x": 1}}
        assert ctx["Dataset Validator"] == {"sample_count": 1000, "metrics": {"x": 1}}

    def test_regression_orchestrator_self_context_update(self) -> None:
        """REGRESSION: orchestrator does ``self.context.update(stage_result)`` — works."""
        ctx = PipelineContext({"a": 1})
        stage_result = {"b": 2, "c": 3}
        ctx.update(stage_result)
        assert ctx == {"a": 1, "b": 2, "c": 3}


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


@pytest.mark.parametrize("attempt_no", [0, 1, 5, 999])
@pytest.mark.parametrize("forced_stages", [None, set(), {"X"}, {"X", "Y", "Z"}])
def test_combinatorial_fork_variants(attempt_no: int, forced_stages: set[str] | None) -> None:
    parent = PipelineContext({PipelineContextKeys.CONFIG_PATH: "/cfg"})
    forked = parent.fork(
        attempt_id=f"A-{attempt_no}",
        attempt_no=attempt_no,
        attempt_directory="/a",
        logical_run_id="L-1",
        run_directory="/r",
        forced_stages=forced_stages,
    )
    assert forked.attempt_no == attempt_no
    assert forked.forced_stages == (forced_stages or set())
    assert forked.config_path == "/cfg"


@pytest.mark.parametrize(
    ("keys", "expected_attempts"),
    [
        ({}, (None, None, None)),
        (
            {PipelineContextKeys.ATTEMPT_ID: "A", PipelineContextKeys.ATTEMPT_NO: 3},
            ("A", 3, None),
        ),
        (
            {
                PipelineContextKeys.ATTEMPT_ID: "A",
                PipelineContextKeys.ATTEMPT_NO: 3,
                PipelineContextKeys.LOGICAL_RUN_ID: "L",
            },
            ("A", 3, "L"),
        ),
    ],
)
def test_combinatorial_partial_seeding(keys: dict, expected_attempts: tuple) -> None:
    """Various partial-seed configurations — accessor returns None for absent keys."""
    ctx = PipelineContext(keys)
    assert ctx.attempt_id == expected_attempts[0]
    assert ctx.attempt_no == expected_attempts[1]
    assert ctx.logical_run_id == expected_attempts[2]


@pytest.mark.parametrize("value_type", [str, int, float, list, dict, set, tuple, bool])
def test_combinatorial_heterogeneous_values(value_type: type) -> None:
    """PipelineContext must accept and return arbitrary value types unchanged."""
    samples: dict[type, Any] = {
        str: "hello",
        int: 42,
        float: 3.14,
        list: [1, 2, 3],
        dict: {"nested": True},
        set: {"a", "b"},
        tuple: (1, 2),
        bool: True,
    }
    ctx = PipelineContext()
    ctx["key"] = samples[value_type]
    assert ctx["key"] == samples[value_type]
    assert type(ctx["key"]) is value_type


def test_combinatorial_fork_then_update_independent() -> None:
    """Compound: fork → update on fork → parent unchanged."""
    parent = PipelineContext({PipelineContextKeys.CONFIG_PATH: "/cfg", "shared": [1, 2]})
    forked = parent.fork(
        attempt_id="A",
        attempt_no=1,
        attempt_directory="/x",
        logical_run_id="L",
        run_directory="/x",
        forced_stages=None,
    )
    forked.update({"only_in_fork": True})
    # Parent unchanged
    assert "only_in_fork" not in parent
    assert parent["shared"] == [1, 2]
    # Fork has both
    assert forked["only_in_fork"] is True
    assert forked["shared"] == [1, 2]

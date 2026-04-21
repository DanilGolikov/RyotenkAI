"""Tests for :func:`src.pipeline.restart_rules.compute_restart_points`.

The rules module is the single source of truth for restart-point decisions;
both :class:`RestartPointsInspector` and
:func:`src.pipeline.restart_points.list_restart_points` delegate to it.
Locking the rules down here guarantees they stay synchronized.

Coverage: positive / negative / boundary / invariants / dep-errors /
regressions / combinatorial.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.pipeline.restart_rules import compute_restart_points
from src.pipeline.stages import StageNames
from src.pipeline.state.models import PipelineState, StageLineageRef, StageRunState

STAGE_ORDER = [
    StageNames.DATASET_VALIDATOR,
    StageNames.GPU_DEPLOYER,
    StageNames.TRAINING_MONITOR,
    StageNames.MODEL_RETRIEVER,
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
]


def _mk_state(
    *,
    lineage: dict[str, dict[str, Any]] | None = None,
    training_hash: str = "t_hash",
    late_hash: str = "l_hash",
    model_hash: str = "",
) -> PipelineState:
    lin = {
        name: StageLineageRef(attempt_id="a1", stage_name=name, outputs=outs)
        for name, outs in (lineage or {}).items()
    }
    return PipelineState(
        schema_version=1,
        logical_run_id="r",
        run_directory="/tmp/r",
        config_path="/tmp/c.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_PENDING,
        training_critical_config_hash=training_hash,
        late_stage_config_hash=late_hash,
        model_dataset_config_hash=model_hash,
        current_output_lineage=lin,
    )


def _hashes(training: str = "t_hash", late: str = "l_hash", model: str = "") -> dict[str, str]:
    return {"training_critical": training, "late_stage": late, "model_dataset": model}


def _always_healthy(_ctx: dict[str, Any] | None) -> bool:
    return True


def _always_unhealthy(_ctx: dict[str, Any] | None) -> bool:
    return False


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_fresh_state_first_stage_is_restart_allowed(self) -> None:
        points = compute_restart_points(
            state=_mk_state(),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        by_stage = {p["stage"]: p for p in points}
        assert by_stage[StageNames.DATASET_VALIDATOR]["available"] is True
        assert by_stage[StageNames.DATASET_VALIDATOR]["reason"] == "restart_allowed"
        assert by_stage[StageNames.DATASET_VALIDATOR]["mode"] == "fresh_only"

    def test_training_monitor_available_with_gpu_outputs(self) -> None:
        points = compute_restart_points(
            state=_mk_state(
                lineage={
                    StageNames.GPU_DEPLOYER: {
                        "ssh_host": "h",
                        "ssh_port": 22,
                        "workspace_path": "/w",
                    }
                },
            ),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        by_stage = {p["stage"]: p for p in points}
        tm = by_stage[StageNames.TRAINING_MONITOR]
        assert tm["available"] is True
        assert tm["mode"] == "reconnect_only"

    def test_evaluator_available_when_inference_live(self) -> None:
        points = compute_restart_points(
            state=_mk_state(
                lineage={
                    StageNames.GPU_DEPLOYER: {"ssh_host": "h", "ssh_port": 22, "workspace_path": "/w"},
                    StageNames.MODEL_RETRIEVER: {"hf_repo_id": "x"},
                    StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
                },
            ),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        by_stage = {p["stage"]: p for p in points}
        me = by_stage[StageNames.MODEL_EVALUATOR]
        assert me["available"] is True
        assert me["mode"] == "live_runtime_only"


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_training_monitor_blocked_without_gpu(self) -> None:
        points = compute_restart_points(
            state=_mk_state(),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        by_stage = {p["stage"]: p for p in points}
        tm = by_stage[StageNames.TRAINING_MONITOR]
        assert tm["available"] is False
        assert tm["reason"] == "missing_gpu_deployer_outputs"

    def test_evaluator_blocked_when_inference_unhealthy(self) -> None:
        points = compute_restart_points(
            state=_mk_state(
                lineage={
                    StageNames.GPU_DEPLOYER: {"ssh_host": "h", "ssh_port": 22, "workspace_path": "/w"},
                    StageNames.MODEL_RETRIEVER: {"hf_repo_id": "x"},
                    StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
                },
            ),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_unhealthy,
        )
        by_stage = {p["stage"]: p for p in points}
        me = by_stage[StageNames.MODEL_EVALUATOR]
        assert me["available"] is False
        assert me["reason"] == "inference_runtime_not_healthy"

    def test_training_drift_blocks_all_stages(self) -> None:
        points = compute_restart_points(
            state=_mk_state(training_hash="old"),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(training="new"),
            inference_health_checker=_always_healthy,
        )
        for p in points:
            assert p["available"] is False
            assert p["reason"] == "training_critical_config_changed"


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_late_stage_drift_exempts_inference_and_eval(self) -> None:
        points = compute_restart_points(
            state=_mk_state(
                late_hash="old_late",
                lineage={
                    StageNames.GPU_DEPLOYER: {"ssh_host": "h", "ssh_port": 22, "workspace_path": "/w"},
                    StageNames.MODEL_RETRIEVER: {"hf_repo_id": "x"},
                    StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
                },
            ),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(late="new_late"),
            inference_health_checker=_always_healthy,
        )
        by_stage = {p["stage"]: p for p in points}
        # exempt stages still available
        assert by_stage[StageNames.INFERENCE_DEPLOYER]["available"] is True
        assert by_stage[StageNames.MODEL_EVALUATOR]["available"] is True
        # non-exempt stages blocked by late-stage drift
        assert by_stage[StageNames.DATASET_VALIDATOR]["reason"] == "late_stage_config_changed"

    def test_model_dataset_hash_takes_precedence_over_legacy(self) -> None:
        points = compute_restart_points(
            state=_mk_state(model_hash="m_old", training_hash="t_hash"),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(training="t_hash", model="m_new"),
            inference_health_checker=_always_healthy,
        )
        for p in points:
            assert p["available"] is False
            assert p["reason"] == "training_critical_config_changed"

    def test_empty_stage_order_returns_empty(self) -> None:
        points = compute_restart_points(
            state=_mk_state(),
            stage_names=[],
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        assert points == []


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_one_entry_per_stage_in_order(self) -> None:
        points = compute_restart_points(
            state=_mk_state(),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        assert [p["stage"] for p in points] == STAGE_ORDER

    def test_every_entry_has_required_keys(self) -> None:
        points = compute_restart_points(
            state=_mk_state(),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_always_healthy,
        )
        for p in points:
            assert set(p.keys()) == {"stage", "available", "mode", "reason"}


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_health_checker_exception_propagates(self) -> None:
        def _raises(_ctx: dict[str, Any] | None) -> bool:
            raise RuntimeError("network boom")

        with pytest.raises(RuntimeError, match="network boom"):
            compute_restart_points(
                state=_mk_state(
                    lineage={
                        StageNames.GPU_DEPLOYER: {
                            "ssh_host": "h", "ssh_port": 22, "workspace_path": "/w"
                        },
                        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "x"},
                        StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
                    },
                ),
                stage_names=STAGE_ORDER,
                config_hashes=_hashes(),
                inference_health_checker=_raises,
            )


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_health_checker_only_called_for_model_evaluator(self) -> None:
        """Regression: health probe must NOT run for every stage — only
        MODEL_EVALUATOR triggers it."""
        calls: list[Any] = []

        def _track(ctx: dict[str, Any] | None) -> bool:
            calls.append(ctx)
            return True

        compute_restart_points(
            state=_mk_state(
                lineage={
                    StageNames.GPU_DEPLOYER: {"ssh_host": "h", "ssh_port": 22, "workspace_path": "/w"},
                    StageNames.MODEL_RETRIEVER: {"hf_repo_id": "x"},
                    StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
                },
            ),
            stage_names=STAGE_ORDER,
            config_hashes=_hashes(),
            inference_health_checker=_track,
        )
        assert len(calls) == 1  # only MODEL_EVALUATOR triggered it


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize(
    "lineage_keys",
    [
        frozenset(),
        frozenset([StageNames.GPU_DEPLOYER]),
        frozenset([StageNames.GPU_DEPLOYER, StageNames.MODEL_RETRIEVER]),
        frozenset(
            [
                StageNames.GPU_DEPLOYER,
                StageNames.MODEL_RETRIEVER,
                StageNames.INFERENCE_DEPLOYER,
            ]
        ),
    ],
)
def test_availability_matrix_by_lineage(lineage_keys: frozenset[str]) -> None:
    lineage: dict[str, dict[str, Any]] = {}
    if StageNames.GPU_DEPLOYER in lineage_keys:
        lineage[StageNames.GPU_DEPLOYER] = {
            "ssh_host": "h", "ssh_port": 22, "workspace_path": "/w",
        }
    if StageNames.MODEL_RETRIEVER in lineage_keys:
        lineage[StageNames.MODEL_RETRIEVER] = {"hf_repo_id": "x"}
    if StageNames.INFERENCE_DEPLOYER in lineage_keys:
        lineage[StageNames.INFERENCE_DEPLOYER] = {"endpoint_url": "http://x"}

    points = compute_restart_points(
        state=_mk_state(lineage=lineage),
        stage_names=STAGE_ORDER,
        config_hashes=_hashes(),
        inference_health_checker=_always_healthy,
    )
    by_stage = {p["stage"]: p["available"] for p in points}

    assert by_stage[StageNames.TRAINING_MONITOR] == (StageNames.GPU_DEPLOYER in lineage_keys)
    assert by_stage[StageNames.MODEL_RETRIEVER] == (StageNames.GPU_DEPLOYER in lineage_keys)
    assert by_stage[StageNames.INFERENCE_DEPLOYER] == (
        StageNames.MODEL_RETRIEVER in lineage_keys
    )
    assert by_stage[StageNames.MODEL_EVALUATOR] == (
        StageNames.INFERENCE_DEPLOYER in lineage_keys
    )

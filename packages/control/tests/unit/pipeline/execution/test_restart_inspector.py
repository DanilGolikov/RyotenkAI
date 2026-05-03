"""Tests for :class:`RestartPointsInspector`.

Coverage split: positive / negative / boundary / invariants / dep-errors /
regressions / combinatorial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.execution import RestartPointsInspector
from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineStateStore, StageLineageRef

if TYPE_CHECKING:
    from pathlib import Path


def _mk_stages() -> list[MagicMock]:
    return [
        MagicMock(stage_name=StageNames.DATASET_VALIDATOR),
        MagicMock(stage_name=StageNames.GPU_DEPLOYER),
        MagicMock(stage_name=StageNames.TRAINING_MONITOR),
        MagicMock(stage_name=StageNames.MODEL_RETRIEVER),
        MagicMock(stage_name=StageNames.INFERENCE_DEPLOYER),
        MagicMock(stage_name=StageNames.MODEL_EVALUATOR),
    ]


def _mk_drift(
    *,
    training_critical: str = "t_hash",
    late_stage: str = "l_hash",
    model_dataset: str = "",
) -> MagicMock:
    cd = MagicMock()
    cd.build_config_hashes.return_value = {
        "training_critical": training_critical,
        "late_stage": late_stage,
        "model_dataset": model_dataset,
    }
    return cd


def _mk_state_with_lineage(
    run_dir: Any,
    *,
    lineage: dict[str, dict[str, Any]] | None = None,
    training_hash: str = "t_hash",
    late_hash: str = "l_hash",
    model_hash: str = "",
) -> PipelineStateStore:
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id=run_dir.name,
        config_path=str(run_dir / "cfg.yaml"),
        training_critical_config_hash=training_hash,
        late_stage_config_hash=late_hash,
        model_dataset_config_hash=model_hash,
    )
    if lineage:
        for name, outputs in lineage.items():
            state.current_output_lineage[name] = StageLineageRef(
                attempt_id="a1", stage_name=name, outputs=outputs
            )
    store.save(state)
    return store


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_fresh_state_allows_fresh_restart_on_first_stage(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r1")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = inspector.inspect(tmp_path / "r1")
        by_stage = {p["stage"]: p for p in points}

        assert by_stage[StageNames.DATASET_VALIDATOR]["available"] is True
        assert by_stage[StageNames.DATASET_VALIDATOR]["mode"] == "fresh_only"

    def test_training_monitor_available_with_gpu_outputs(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(
            tmp_path / "r2",
            lineage={
                StageNames.GPU_DEPLOYER: {
                    "ssh_host": "h",
                    "ssh_port": 22,
                    "workspace_path": "/w",
                }
            },
        )
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r2")}
        tm = points[StageNames.TRAINING_MONITOR]
        assert tm["available"] is True
        assert tm["mode"] == "reconnect_only"

    def test_model_retriever_available_with_gpu_ref(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(
            tmp_path / "r3",
            lineage={StageNames.GPU_DEPLOYER: {"any": "value"}},
        )
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r3")}
        mr = points[StageNames.MODEL_RETRIEVER]
        assert mr["available"] is True
        assert mr["mode"] == "fresh_or_resume"


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_training_monitor_blocked_without_gpu_outputs(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r4")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r4")}
        tm = points[StageNames.TRAINING_MONITOR]
        assert tm["available"] is False
        assert tm["reason"] == "missing_gpu_deployer_outputs"

    def test_model_retriever_blocked_without_gpu(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r5")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r5")}
        mr = points[StageNames.MODEL_RETRIEVER]
        assert mr["available"] is False

    def test_inference_deployer_blocked_without_model_retriever(
        self, tmp_path: Path
    ) -> None:
        _mk_state_with_lineage(tmp_path / "r6")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r6")}
        id_ = points[StageNames.INFERENCE_DEPLOYER]
        assert id_["available"] is False
        assert id_["reason"] == "missing_model_retriever_outputs"

    def test_model_evaluator_blocked_without_inference(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r7")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r7")}
        me = points[StageNames.MODEL_EVALUATOR]
        assert me["available"] is False
        assert me["reason"] == "missing_inference_outputs"

    def test_training_drift_blocks_restart(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r8", training_hash="old_hash")
        inspector = RestartPointsInspector(
            stages=_mk_stages(), config_drift=_mk_drift(training_critical="new_hash")
        )

        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r8")}
        # Every stage is blocked by training-critical drift.
        for p in points.values():
            assert p["available"] is False
            assert p["reason"] == "training_critical_config_changed"


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_late_stage_drift_exempts_inference_and_evaluator(
        self, tmp_path: Path
    ) -> None:
        _mk_state_with_lineage(
            tmp_path / "r9",
            late_hash="old_late",
            lineage={
                StageNames.GPU_DEPLOYER: {
                    "ssh_host": "h",
                    "ssh_port": 22,
                    "workspace_path": "/w",
                },
                StageNames.MODEL_RETRIEVER: {"hf_repo_id": "foo/bar"},
                StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
            },
        )
        inspector = RestartPointsInspector(
            stages=_mk_stages(),
            config_drift=_mk_drift(late_stage="new_late"),
        )
        with patch(
            "src.pipeline.execution.restart_inspector.is_inference_runtime_healthy",
            return_value=True,
        ):
            points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r9")}

        # inference + evaluator are exempt from late-stage drift
        assert points[StageNames.INFERENCE_DEPLOYER]["available"] is True
        assert points[StageNames.MODEL_EVALUATOR]["available"] is True
        # Earlier stages are blocked by late-stage drift
        assert points[StageNames.DATASET_VALIDATOR]["reason"] == "late_stage_config_changed"

    def test_model_dataset_hash_supersedes_legacy_training_hash(
        self, tmp_path: Path
    ) -> None:
        _mk_state_with_lineage(
            tmp_path / "r10",
            model_hash="md_old",  # present → takes precedence
            training_hash="t_hash",  # matches inspector's; doesn't block
        )
        inspector = RestartPointsInspector(
            stages=_mk_stages(),
            config_drift=_mk_drift(model_dataset="md_new"),
        )
        points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r10")}
        for p in points.values():
            assert p["available"] is False
            assert p["reason"] == "training_critical_config_changed"


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_exactly_one_entry_per_stage(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r_inv")
        stages = _mk_stages()
        inspector = RestartPointsInspector(stages=stages, config_drift=_mk_drift())

        points = inspector.inspect(tmp_path / "r_inv")
        assert len(points) == len(stages)
        assert [p["stage"] for p in points] == [s.stage_name for s in stages]

    def test_every_entry_has_required_keys(self, tmp_path: Path) -> None:
        _mk_state_with_lineage(tmp_path / "r_inv2")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())

        points = inspector.inspect(tmp_path / "r_inv2")
        for p in points:
            assert set(p.keys()) == {"stage", "available", "mode", "reason"}


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_missing_state_file_raises(self, tmp_path: Path) -> None:
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())
        with pytest.raises(Exception):  # noqa: B017 - PipelineStateLoadError
            inspector.inspect(tmp_path / "nonexistent")

    def test_health_probe_failure_marks_evaluator_unhealthy(
        self, tmp_path: Path
    ) -> None:
        _mk_state_with_lineage(
            tmp_path / "r_health",
            lineage={
                StageNames.GPU_DEPLOYER: {
                    "ssh_host": "h",
                    "ssh_port": 22,
                    "workspace_path": "/w",
                },
                StageNames.MODEL_RETRIEVER: {"hf_repo_id": "foo/bar"},
                StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"},
            },
        )
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())
        with patch(
            "src.pipeline.execution.restart_inspector.is_inference_runtime_healthy",
            return_value=False,
        ):
            points = {p["stage"]: p for p in inspector.inspect(tmp_path / "r_health")}
        me = points[StageNames.MODEL_EVALUATOR]
        assert me["available"] is False
        assert me["reason"] == "inference_runtime_not_healthy"


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_inspector_does_not_mutate_attempt_controller(self, tmp_path: Path) -> None:
        """Regression: list_restart_points used to set
        ``self._pipeline_state = state`` which polluted live state.
        The inspector is pure read-only."""
        _mk_state_with_lineage(tmp_path / "r_pure")
        inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())
        # Calling inspect() twice returns the same shape; no hidden state.
        p1 = inspector.inspect(tmp_path / "r_pure")
        p2 = inspector.inspect(tmp_path / "r_pure")
        assert p1 == p2


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
def test_availability_matrix_by_lineage_keys(
    tmp_path: Path, lineage_keys: frozenset[str]
) -> None:
    lineage = {}
    if StageNames.GPU_DEPLOYER in lineage_keys:
        lineage[StageNames.GPU_DEPLOYER] = {
            "ssh_host": "h",
            "ssh_port": 22,
            "workspace_path": "/w",
        }
    if StageNames.MODEL_RETRIEVER in lineage_keys:
        lineage[StageNames.MODEL_RETRIEVER] = {"hf_repo_id": "foo/bar"}
    if StageNames.INFERENCE_DEPLOYER in lineage_keys:
        lineage[StageNames.INFERENCE_DEPLOYER] = {"endpoint_url": "http://x"}

    _mk_state_with_lineage(tmp_path / f"m_{len(lineage_keys)}", lineage=lineage)
    inspector = RestartPointsInspector(stages=_mk_stages(), config_drift=_mk_drift())
    with patch(
        "src.pipeline.execution.restart_inspector.is_inference_runtime_healthy",
        return_value=True,
    ):
        points = {
            p["stage"]: p["available"]
            for p in inspector.inspect(tmp_path / f"m_{len(lineage_keys)}")
        }

    assert points[StageNames.TRAINING_MONITOR] == (StageNames.GPU_DEPLOYER in lineage_keys)
    assert points[StageNames.MODEL_RETRIEVER] == (StageNames.GPU_DEPLOYER in lineage_keys)
    assert points[StageNames.INFERENCE_DEPLOYER] == (
        StageNames.MODEL_RETRIEVER in lineage_keys
    )
    assert points[StageNames.MODEL_EVALUATOR] == (
        StageNames.INFERENCE_DEPLOYER in lineage_keys
    )

"""Comprehensive tests for ContextPropagator across all 7 categories."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.artifacts import StageArtifactCollector
from src.pipeline.context.propagator import ContextPropagator
from src.pipeline.stages import StageNames


@pytest.fixture
def validation_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.build_dataset_validation_state_outputs.return_value = {"datasets": []}
    return mgr


@pytest.fixture
def prop(validation_mgr: MagicMock) -> ContextPropagator:
    return ContextPropagator(validation_mgr)


def _collector() -> StageArtifactCollector:
    return StageArtifactCollector(stage="s", artifact_name="s.json")


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_sync_root_copies_endpoint_keys(self, prop: ContextPropagator) -> None:
        ctx: dict = {}
        prop.sync_root_from_stage(
            context=ctx,
            stage_name=StageNames.INFERENCE_DEPLOYER,
            outputs={"endpoint_url": "http://x", "inference_model_name": "m"},
        )
        assert ctx == {"endpoint_url": "http://x", "inference_model_name": "m"}

    def test_extract_inference_outputs_with_endpoint_info(self, prop: ContextPropagator) -> None:
        ctx = {
            StageNames.INFERENCE_DEPLOYER: {
                "endpoint_url": "http://x",
                "endpoint_info": {"health_url": "http://x/h"},
            }
        }
        out = prop.extract_restart_outputs(context=ctx, stage_name=StageNames.INFERENCE_DEPLOYER)
        assert out["endpoint_info"] == {"health_url": "http://x/h"}

    def test_skip_reason_surfaced(self, prop: ContextPropagator) -> None:
        assert (
            prop.get_stage_skip_reason(
                context={StageNames.INFERENCE_DEPLOYER: {"inference_skipped": True, "reason": "off"}},
                stage_name=StageNames.INFERENCE_DEPLOYER,
            )
            == "off"
        )


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_extract_non_dict_returns_empty(self, prop: ContextPropagator) -> None:
        assert (
            prop.extract_restart_outputs(
                context={StageNames.GPU_DEPLOYER: "garbage"},
                stage_name=StageNames.GPU_DEPLOYER,
            )
            == {}
        )

    def test_sync_root_no_op_on_other_stages(self, prop: ContextPropagator) -> None:
        ctx: dict = {}
        prop.sync_root_from_stage(
            context=ctx,
            stage_name=StageNames.TRAINING_MONITOR,
            outputs={"endpoint_url": "should-not-propagate"},
        )
        assert ctx == {}

    def test_skip_reason_non_dict_returns_none(self, prop: ContextPropagator) -> None:
        assert (
            prop.get_stage_skip_reason(
                context={StageNames.MODEL_EVALUATOR: "garbage"},
                stage_name=StageNames.MODEL_EVALUATOR,
            )
            is None
        )

    def test_fill_collector_non_dict_noop(self, prop: ContextPropagator) -> None:
        collector = _collector()
        prop.fill_collector_from_context(
            context={StageNames.GPU_DEPLOYER: None},
            stage_name=StageNames.GPU_DEPLOYER,
            collector=collector,
        )
        assert not collector._data


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_extract_gpu_deployer_all_none_keys(self, prop: ContextPropagator) -> None:
        """Boundary: all None keys yield empty dict."""
        ctx = {StageNames.GPU_DEPLOYER: {k: None for k in ("resource_id", "ssh_host")}}
        out = prop.extract_restart_outputs(context=ctx, stage_name=StageNames.GPU_DEPLOYER)
        assert out == {}

    def test_extract_training_monitor_no_workspace_skips_remote_output_dir(
        self, prop: ContextPropagator
    ) -> None:
        ctx = {
            StageNames.TRAINING_MONITOR: {"status": "done"},
            StageNames.GPU_DEPLOYER: {"workspace_path": ""},
        }
        out = prop.extract_restart_outputs(context=ctx, stage_name=StageNames.TRAINING_MONITOR)
        assert "remote_output_dir" not in out

    def test_extract_unknown_stage_returns_defensive_copy(self, prop: ContextPropagator) -> None:
        src = {"a": 1, "b": 2}
        ctx = {"UnknownStage": src}
        out = prop.extract_restart_outputs(context=ctx, stage_name="UnknownStage")
        assert out == src
        assert out is not src  # defensive copy

    def test_extract_inference_endpoint_info_wrong_type_normalised(
        self, prop: ContextPropagator
    ) -> None:
        ctx = {StageNames.INFERENCE_DEPLOYER: {"endpoint_info": ["list", "not", "dict"]}}
        out = prop.extract_restart_outputs(context=ctx, stage_name=StageNames.INFERENCE_DEPLOYER)
        assert out["endpoint_info"] == {}


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_extract_never_raises_on_arbitrary_context(
        self, prop: ContextPropagator, validation_mgr: MagicMock
    ) -> None:
        """Invariant: extract_restart_outputs tolerates any shape."""
        for ctx in [{}, {"X": None}, {"X": "str"}, {"X": 42}, {"X": {}}]:
            for stage in (
                StageNames.DATASET_VALIDATOR,
                StageNames.GPU_DEPLOYER,
                StageNames.TRAINING_MONITOR,
                StageNames.MODEL_RETRIEVER,
                StageNames.INFERENCE_DEPLOYER,
                StageNames.MODEL_EVALUATOR,
            ):
                prop.extract_restart_outputs(context=ctx, stage_name=stage)

    def test_fill_never_raises_on_empty_context(self, prop: ContextPropagator) -> None:
        collector = _collector()
        for stage in (
            StageNames.GPU_DEPLOYER,
            StageNames.TRAINING_MONITOR,
            StageNames.MODEL_RETRIEVER,
            StageNames.INFERENCE_DEPLOYER,
            StageNames.MODEL_EVALUATOR,
        ):
            prop.fill_collector_from_context(context={}, stage_name=stage, collector=collector)


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_validation_mgr_exception_bubbles_up(self, validation_mgr: MagicMock) -> None:
        validation_mgr.build_dataset_validation_state_outputs.side_effect = RuntimeError("mgr down")
        prop = ContextPropagator(validation_mgr)
        with pytest.raises(RuntimeError):
            prop.extract_restart_outputs(
                context={StageNames.DATASET_VALIDATOR: {"x": 1}},
                stage_name=StageNames.DATASET_VALIDATOR,
            )


# =============================================================================
# 6. REGRESSIONS
# =============================================================================


class TestRegressions:
    def test_regression_upload_duration_shared_const(self, prop: ContextPropagator) -> None:
        """REGRESSION: `upload_duration_seconds` used to be a local literal in 3 files.
        Now it's CTX_UPLOAD_DURATION in pipeline.constants."""
        from src.pipeline.constants import CTX_UPLOAD_DURATION

        assert CTX_UPLOAD_DURATION == "upload_duration_seconds"

    def test_regression_model_evaluator_dict_summary_spreads(self, prop: ContextPropagator) -> None:
        """Regression: dict eval_summary was previously spread; ensure it still is."""
        collector = _collector()
        prop.fill_collector_from_context(
            context={StageNames.MODEL_EVALUATOR: {"eval_summary": {"a": 1}}},
            stage_name=StageNames.MODEL_EVALUATOR,
            collector=collector,
        )
        assert collector._data["a"] == 1


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


_STAGES_TO_EXTRACT = [
    StageNames.GPU_DEPLOYER,
    StageNames.TRAINING_MONITOR,
    StageNames.MODEL_RETRIEVER,
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
]


@pytest.mark.parametrize("stage", _STAGES_TO_EXTRACT)
@pytest.mark.parametrize("missing", [True, False])
def test_combinatorial_extract_stages_x_presence(
    prop: ContextPropagator, stage: str, missing: bool
) -> None:
    """Cross product of (stage, ctx present/absent) — should never raise."""
    ctx = {} if missing else {stage: {}}
    result = prop.extract_restart_outputs(context=ctx, stage_name=stage)
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    ("skip_key", "reason_key", "expected"),
    [
        ("inference_skipped", "reason", "custom-reason"),
        ("evaluation_skipped", "reason", "custom-reason"),
        ("inference_skipped", None, "inference_skipped"),
        ("evaluation_skipped", None, "evaluation_skipped"),
    ],
)
def test_combinatorial_skip_reason_defaults(
    prop: ContextPropagator, skip_key: str, reason_key: str | None, expected: str
) -> None:
    ctx: dict = {StageNames.INFERENCE_DEPLOYER: {skip_key: True}}
    if reason_key:
        ctx[StageNames.INFERENCE_DEPLOYER][reason_key] = "custom-reason"
    assert (
        prop.get_stage_skip_reason(
            context=ctx, stage_name=StageNames.INFERENCE_DEPLOYER
        )
        == expected
    )

"""Unit tests for ContextPropagator.

Each test exercises a single responsibility (sync / extract / fill / skip)
without standing up an orchestrator.
"""

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
def propagator(validation_mgr: MagicMock) -> ContextPropagator:
    return ContextPropagator(validation_mgr)


# -----------------------------------------------------------------------------
# sync_root_from_stage
# -----------------------------------------------------------------------------


def test_sync_copies_inference_keys(propagator: ContextPropagator) -> None:
    context: dict = {}
    propagator.sync_root_from_stage(
        context=context,
        stage_name=StageNames.INFERENCE_DEPLOYER,
        outputs={"endpoint_url": "http://host/run", "inference_model_name": "gpt2"},
    )
    assert context["endpoint_url"] == "http://host/run"
    assert context["inference_model_name"] == "gpt2"


def test_sync_skips_missing_keys(propagator: ContextPropagator) -> None:
    context: dict = {}
    propagator.sync_root_from_stage(
        context=context,
        stage_name=StageNames.INFERENCE_DEPLOYER,
        outputs={"endpoint_url": "http://x"},
    )
    assert context == {"endpoint_url": "http://x"}


def test_sync_ignores_unrelated_stages(propagator: ContextPropagator) -> None:
    context: dict = {}
    propagator.sync_root_from_stage(
        context=context,
        stage_name=StageNames.GPU_DEPLOYER,
        outputs={"endpoint_url": "http://x"},
    )
    assert context == {}


# -----------------------------------------------------------------------------
# extract_restart_outputs
# -----------------------------------------------------------------------------


def test_extract_empty_for_missing_stage(propagator: ContextPropagator) -> None:
    assert propagator.extract_restart_outputs(context={}, stage_name=StageNames.GPU_DEPLOYER) == {}


def test_extract_returns_empty_for_non_dict(propagator: ContextPropagator) -> None:
    assert (
        propagator.extract_restart_outputs(
            context={StageNames.GPU_DEPLOYER: "not-a-dict"},
            stage_name=StageNames.GPU_DEPLOYER,
        )
        == {}
    )


def test_extract_dataset_validator_delegates_to_validation_mgr(
    propagator: ContextPropagator, validation_mgr: MagicMock
) -> None:
    stage_ctx = {"some": "data"}
    context = {StageNames.DATASET_VALIDATOR: stage_ctx}
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.DATASET_VALIDATOR)
    validation_mgr.build_dataset_validation_state_outputs.assert_called_once_with(stage_ctx=stage_ctx)
    assert out == {"datasets": []}


def test_extract_gpu_deployer_filters_none_keys(propagator: ContextPropagator) -> None:
    context = {
        StageNames.GPU_DEPLOYER: {
            "resource_id": "r1",
            "ssh_host": "host",
            "ssh_port": 22,
            "ssh_user": None,  # filtered
            "workspace_path": "/w",
            "provider_type": "runpod",
        }
    }
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.GPU_DEPLOYER)
    assert "ssh_user" not in out
    assert out["resource_id"] == "r1"
    assert out["workspace_path"] == "/w"


def test_extract_training_monitor_derives_remote_output_dir(propagator: ContextPropagator) -> None:
    context = {
        StageNames.TRAINING_MONITOR: {
            "status": "completed",
            "training_duration_seconds": 123.4,
            "training_info": {"loss": 0.1},
        },
        StageNames.GPU_DEPLOYER: {"workspace_path": "/workspace"},
    }
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.TRAINING_MONITOR)
    assert out["remote_output_dir"] == "/workspace/output"
    assert out["status"] == "completed"


def test_extract_training_monitor_no_remote_output_when_no_workspace(
    propagator: ContextPropagator,
) -> None:
    context = {
        StageNames.TRAINING_MONITOR: {"status": "completed"},
        StageNames.GPU_DEPLOYER: {},
    }
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.TRAINING_MONITOR)
    assert "remote_output_dir" not in out


def test_extract_inference_deployer_preserves_endpoint_info(propagator: ContextPropagator) -> None:
    ei = {"health_url": "http://x/h"}
    context = {
        StageNames.INFERENCE_DEPLOYER: {
            "endpoint_url": "http://x",
            "inference_model_name": "m",
            "endpoint_info": ei,
        }
    }
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.INFERENCE_DEPLOYER)
    assert out["endpoint_info"] is ei
    assert out["endpoint_url"] == "http://x"


def test_extract_inference_deployer_non_dict_endpoint_info_is_normalised(
    propagator: ContextPropagator,
) -> None:
    context = {
        StageNames.INFERENCE_DEPLOYER: {
            "endpoint_url": "http://x",
            "endpoint_info": "garbage",
        }
    }
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.INFERENCE_DEPLOYER)
    assert out["endpoint_info"] == {}


def test_extract_model_retriever_keys(propagator: ContextPropagator) -> None:
    context = {
        StageNames.MODEL_RETRIEVER: {
            "hf_repo_id": "org/m",
            "hf_uploaded": True,
            "model_size_mb": 500.0,
        }
    }
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.MODEL_RETRIEVER)
    assert out == {"hf_repo_id": "org/m", "hf_uploaded": True, "model_size_mb": 500.0}


def test_extract_model_evaluator_keys(propagator: ContextPropagator) -> None:
    context = {StageNames.MODEL_EVALUATOR: {"eval_passed": True}}
    out = propagator.extract_restart_outputs(context=context, stage_name=StageNames.MODEL_EVALUATOR)
    assert out == {"eval_passed": True}


def test_extract_unknown_stage_returns_copy(propagator: ContextPropagator) -> None:
    src = {"a": 1}
    context = {"Unknown": src}
    out = propagator.extract_restart_outputs(context=context, stage_name="Unknown")
    assert out == {"a": 1}
    assert out is not src  # defensive copy


# -----------------------------------------------------------------------------
# fill_collector_from_context
# -----------------------------------------------------------------------------


def _make_collector() -> StageArtifactCollector:
    return StageArtifactCollector(stage="s", artifact_name="s.json")


def test_fill_noop_for_non_dict(propagator: ContextPropagator) -> None:
    collector = _make_collector()
    propagator.fill_collector_from_context(
        context={StageNames.GPU_DEPLOYER: "garbage"},
        stage_name=StageNames.GPU_DEPLOYER,
        collector=collector,
    )


def test_fill_gpu_deployer_populates_collector(propagator: ContextPropagator) -> None:
    collector = _make_collector()
    propagator.fill_collector_from_context(
        context={
            StageNames.GPU_DEPLOYER: {
                "upload_duration_seconds": 10,
                "deps_duration_seconds": 5,
                "provider_name": "runpod",
                "resource_id": "r1",
            }
        },
        stage_name=StageNames.GPU_DEPLOYER,
        collector=collector,
    )
    # Collector stores kwargs via put()
    assert collector._data["upload_duration_seconds"] == 10
    assert collector._data["resource_id"] == "r1"


def test_fill_model_evaluator_spreads_dict_summary(propagator: ContextPropagator) -> None:
    collector = _make_collector()
    propagator.fill_collector_from_context(
        context={StageNames.MODEL_EVALUATOR: {"eval_summary": {"accuracy": 0.9, "f1": 0.85}}},
        stage_name=StageNames.MODEL_EVALUATOR,
        collector=collector,
    )
    assert collector._data["accuracy"] == 0.9
    assert collector._data["f1"] == 0.85


def test_fill_model_evaluator_stringifies_non_dict_summary(propagator: ContextPropagator) -> None:
    collector = _make_collector()
    propagator.fill_collector_from_context(
        context={StageNames.MODEL_EVALUATOR: {"eval_summary": "raw-string"}},
        stage_name=StageNames.MODEL_EVALUATOR,
        collector=collector,
    )
    assert collector._data["eval_summary"] == "raw-string"


# -----------------------------------------------------------------------------
# get_stage_skip_reason
# -----------------------------------------------------------------------------


def test_skip_reason_none_when_not_skipped(propagator: ContextPropagator) -> None:
    assert (
        propagator.get_stage_skip_reason(
            context={StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"}},
            stage_name=StageNames.INFERENCE_DEPLOYER,
        )
        is None
    )


def test_skip_reason_inference_skipped(propagator: ContextPropagator) -> None:
    assert (
        propagator.get_stage_skip_reason(
            context={StageNames.INFERENCE_DEPLOYER: {"inference_skipped": True, "reason": "disabled_in_config"}},
            stage_name=StageNames.INFERENCE_DEPLOYER,
        )
        == "disabled_in_config"
    )


def test_skip_reason_evaluation_skipped_default(propagator: ContextPropagator) -> None:
    assert (
        propagator.get_stage_skip_reason(
            context={StageNames.MODEL_EVALUATOR: {"evaluation_skipped": True}},
            stage_name=StageNames.MODEL_EVALUATOR,
        )
        == "evaluation_skipped"
    )


def test_skip_reason_non_dict_returns_none(propagator: ContextPropagator) -> None:
    assert (
        propagator.get_stage_skip_reason(
            context={StageNames.MODEL_EVALUATOR: "garbage"},
            stage_name=StageNames.MODEL_EVALUATOR,
        )
        is None
    )

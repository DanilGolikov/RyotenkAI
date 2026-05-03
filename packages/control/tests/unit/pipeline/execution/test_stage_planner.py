"""Unit tests for StagePlanner and is_inference_runtime_healthy.

Covers stage reference normalisation, enabled-stage computation, resume
derivation and prerequisite validation without standing up a full orchestrator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.execution.stage_planner import (
    StagePlanner,
    is_inference_runtime_healthy,
)
from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

_ORDER = [
    StageNames.DATASET_VALIDATOR,
    StageNames.GPU_DEPLOYER,
    StageNames.TRAINING_MONITOR,
    StageNames.MODEL_RETRIEVER,
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
]


def _mock_stage(name: str) -> MagicMock:
    stage = MagicMock()
    stage.stage_name = name
    return stage


def _build_config(*, inference_enabled: bool = False, evaluation_enabled: bool = False) -> MagicMock:
    config = MagicMock()
    config.inference.enabled = inference_enabled
    config.evaluation.enabled = evaluation_enabled
    return config


@pytest.fixture
def stages() -> list[MagicMock]:
    return [_mock_stage(name) for name in _ORDER]


@pytest.fixture
def planner(stages: list[MagicMock]) -> StagePlanner:
    return StagePlanner(stages, _build_config())


# -----------------------------------------------------------------------------
# get_stage_index
# -----------------------------------------------------------------------------


def test_get_stage_index_known_name(planner: StagePlanner) -> None:
    assert planner.get_stage_index(StageNames.DATASET_VALIDATOR) == 0
    assert planner.get_stage_index(StageNames.MODEL_EVALUATOR) == 5


def test_get_stage_index_unknown_raises(planner: StagePlanner) -> None:
    with pytest.raises(ValueError, match="Unknown stage name"):
        planner.get_stage_index("Imaginary Stage")


# -----------------------------------------------------------------------------
# normalize_stage_ref
# -----------------------------------------------------------------------------


def test_normalize_none_raises(planner: StagePlanner) -> None:
    with pytest.raises(ValueError, match="required"):
        planner.normalize_stage_ref(None)


def test_normalize_empty_string_raises(planner: StagePlanner) -> None:
    with pytest.raises(ValueError, match="empty"):
        planner.normalize_stage_ref("   ")


def test_normalize_int_in_range(planner: StagePlanner) -> None:
    assert planner.normalize_stage_ref(1) == StageNames.DATASET_VALIDATOR
    assert planner.normalize_stage_ref(6) == StageNames.MODEL_EVALUATOR


def test_normalize_int_out_of_range(planner: StagePlanner) -> None:
    with pytest.raises(ValueError, match="out of range"):
        planner.normalize_stage_ref(0)
    with pytest.raises(ValueError, match="out of range"):
        planner.normalize_stage_ref(99)


def test_normalize_str_digit(planner: StagePlanner) -> None:
    assert planner.normalize_stage_ref("1") == StageNames.DATASET_VALIDATOR
    assert planner.normalize_stage_ref("6") == StageNames.MODEL_EVALUATOR


def test_normalize_str_digit_out_of_range(planner: StagePlanner) -> None:
    with pytest.raises(ValueError, match="out of range"):
        planner.normalize_stage_ref("99")


def test_normalize_case_insensitive(planner: StagePlanner) -> None:
    assert planner.normalize_stage_ref("inference deployer") == StageNames.INFERENCE_DEPLOYER
    assert planner.normalize_stage_ref("INFERENCE DEPLOYER") == StageNames.INFERENCE_DEPLOYER


def test_normalize_underscore_variant(planner: StagePlanner) -> None:
    assert planner.normalize_stage_ref("inference_deployer") == StageNames.INFERENCE_DEPLOYER
    assert planner.normalize_stage_ref("model_evaluator") == StageNames.MODEL_EVALUATOR


def test_normalize_unknown_raises(planner: StagePlanner) -> None:
    with pytest.raises(ValueError, match="Unknown stage reference"):
        planner.normalize_stage_ref("foo")


# -----------------------------------------------------------------------------
# forced_stage_names
# -----------------------------------------------------------------------------


def test_forced_stage_names_inference_when_disabled(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(inference_enabled=False))
    forced = planner.forced_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER)
    assert forced == {StageNames.INFERENCE_DEPLOYER}


def test_forced_stage_names_evaluator_when_disabled(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(evaluation_enabled=False))
    forced = planner.forced_stage_names(start_stage_name=StageNames.MODEL_EVALUATOR)
    assert forced == {StageNames.MODEL_EVALUATOR}


def test_forced_stage_names_none_when_enabled(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(inference_enabled=True, evaluation_enabled=True))
    assert planner.forced_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER) == set()
    assert planner.forced_stage_names(start_stage_name=StageNames.MODEL_EVALUATOR) == set()


def test_forced_stage_names_empty_for_other_starts(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config())
    assert planner.forced_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR) == set()


# -----------------------------------------------------------------------------
# compute_enabled_stage_names
# -----------------------------------------------------------------------------


def test_compute_enabled_default_4_stages(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config())
    enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR)
    assert enabled == _ORDER[:4]


def test_compute_enabled_with_inference(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(inference_enabled=True))
    enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR)
    assert enabled == _ORDER[:5]


def test_compute_enabled_with_inference_and_evaluation(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(inference_enabled=True, evaluation_enabled=True))
    enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR)
    assert enabled == _ORDER


def test_compute_enabled_forces_inference_when_explicit_start(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(inference_enabled=False))
    enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER)
    assert StageNames.INFERENCE_DEPLOYER in enabled


def test_compute_enabled_does_not_duplicate_forced(stages: list[MagicMock]) -> None:
    planner = StagePlanner(stages, _build_config(inference_enabled=True))
    enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER)
    assert enabled.count(StageNames.INFERENCE_DEPLOYER) == 1


# -----------------------------------------------------------------------------
# derive_resume_stage
# -----------------------------------------------------------------------------


def _empty_state() -> PipelineState:
    return PipelineState(
        schema_version=1,
        logical_run_id="r1",
        run_directory="/tmp/run",
        config_path="/tmp/cfg.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_PENDING,
        training_critical_config_hash="",
        late_stage_config_hash="",
    )


def _state_with_runs(**stage_statuses: str) -> PipelineState:
    state = _empty_state()
    attempt = PipelineAttemptState(
        attempt_id="a1",
        attempt_no=1,
        runtime_name="runtime",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_RUNNING,
        started_at="2026-04-21T00:00:00+00:00",
    )
    for name, status in stage_statuses.items():
        attempt.stage_runs[name] = StageRunState(stage_name=name, status=status)
    state.attempts.append(attempt)
    return state


def test_derive_resume_no_attempts_returns_first(planner: StagePlanner) -> None:
    assert planner.derive_resume_stage(_empty_state()) == StageNames.DATASET_VALIDATOR


def test_derive_resume_returns_first_missing_stage(planner: StagePlanner) -> None:
    state = _state_with_runs(**{StageNames.DATASET_VALIDATOR: StageRunState.STATUS_COMPLETED})
    assert planner.derive_resume_stage(state) == StageNames.GPU_DEPLOYER


@pytest.mark.parametrize(
    "status",
    [
        StageRunState.STATUS_FAILED,
        StageRunState.STATUS_INTERRUPTED,
        StageRunState.STATUS_PENDING,
        StageRunState.STATUS_RUNNING,
        StageRunState.STATUS_STALE,
    ],
)
def test_derive_resume_returns_incomplete_stage(planner: StagePlanner, status: str) -> None:
    state = _state_with_runs(
        **{
            StageNames.DATASET_VALIDATOR: StageRunState.STATUS_COMPLETED,
            StageNames.GPU_DEPLOYER: status,
        }
    )
    assert planner.derive_resume_stage(state) == StageNames.GPU_DEPLOYER


def test_derive_resume_all_completed_returns_none(planner: StagePlanner) -> None:
    state = _state_with_runs(**{name: StageRunState.STATUS_COMPLETED for name in _ORDER})
    assert planner.derive_resume_stage(state) is None


def test_derive_resume_skipped_counts_as_done(planner: StagePlanner) -> None:
    state = _state_with_runs(
        **{
            StageNames.DATASET_VALIDATOR: StageRunState.STATUS_COMPLETED,
            StageNames.GPU_DEPLOYER: StageRunState.STATUS_SKIPPED,
        }
    )
    assert planner.derive_resume_stage(state) == StageNames.TRAINING_MONITOR


# -----------------------------------------------------------------------------
# validate_stage_prerequisites
# -----------------------------------------------------------------------------


def test_prereq_training_monitor_missing_gpu_context(planner: StagePlanner) -> None:
    err = planner.validate_stage_prerequisites(
        stage_name=StageNames.TRAINING_MONITOR,
        start_stage_name=StageNames.TRAINING_MONITOR,
        context={},
    )
    assert err is not None
    assert err.code == "MISSING_TRAINING_MONITOR_PREREQUISITES"


def test_prereq_training_monitor_ok(planner: StagePlanner) -> None:
    context = {
        StageNames.GPU_DEPLOYER: {
            "ssh_host": "host",
            "ssh_port": 22,
            "workspace_path": "/workspace",
        }
    }
    assert (
        planner.validate_stage_prerequisites(
            stage_name=StageNames.TRAINING_MONITOR,
            start_stage_name=StageNames.TRAINING_MONITOR,
            context=context,
        )
        is None
    )


def test_prereq_inference_deployer_missing_retriever(planner: StagePlanner) -> None:
    err = planner.validate_stage_prerequisites(
        stage_name=StageNames.INFERENCE_DEPLOYER,
        start_stage_name=StageNames.INFERENCE_DEPLOYER,
        context={},
    )
    assert err is not None
    assert err.code == "MISSING_INFERENCE_PREREQUISITES"


def test_prereq_inference_deployer_accepts_hf_repo(planner: StagePlanner) -> None:
    context = {StageNames.MODEL_RETRIEVER: {"hf_repo_id": "org/model"}}
    assert (
        planner.validate_stage_prerequisites(
            stage_name=StageNames.INFERENCE_DEPLOYER,
            start_stage_name=StageNames.INFERENCE_DEPLOYER,
            context=context,
        )
        is None
    )


def test_prereq_inference_deployer_accepts_local_path(planner: StagePlanner) -> None:
    context = {StageNames.MODEL_RETRIEVER: {"local_model_path": "/tmp/model"}}
    assert (
        planner.validate_stage_prerequisites(
            stage_name=StageNames.INFERENCE_DEPLOYER,
            start_stage_name=StageNames.INFERENCE_DEPLOYER,
            context=context,
        )
        is None
    )


def test_prereq_model_evaluator_inference_unhealthy(planner: StagePlanner) -> None:
    with patch(
        "src.pipeline.execution.stage_planner.is_inference_runtime_healthy",
        return_value=False,
    ):
        err = planner.validate_stage_prerequisites(
            stage_name=StageNames.MODEL_EVALUATOR,
            start_stage_name=StageNames.MODEL_EVALUATOR,
            context={StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"}},
        )
    assert err is not None
    assert err.code == "INFERENCE_RUNTIME_NOT_HEALTHY"


def test_prereq_model_evaluator_inference_healthy(planner: StagePlanner) -> None:
    with patch(
        "src.pipeline.execution.stage_planner.is_inference_runtime_healthy",
        return_value=True,
    ):
        err = planner.validate_stage_prerequisites(
            stage_name=StageNames.MODEL_EVALUATOR,
            start_stage_name=StageNames.MODEL_EVALUATOR,
            context={StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"}},
        )
    assert err is None


def test_prereq_unrelated_stage_no_error(planner: StagePlanner) -> None:
    assert (
        planner.validate_stage_prerequisites(
            stage_name=StageNames.DATASET_VALIDATOR,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            context={},
        )
        is None
    )


# -----------------------------------------------------------------------------
# is_inference_runtime_healthy
# -----------------------------------------------------------------------------


def test_health_none_returns_false() -> None:
    assert is_inference_runtime_healthy(None) is False


def test_health_non_dict_returns_false() -> None:
    assert is_inference_runtime_healthy("not a dict") is False  # type: ignore[arg-type]


def test_health_empty_dict_returns_false() -> None:
    assert is_inference_runtime_healthy({}) is False


def test_health_missing_url_returns_false() -> None:
    assert is_inference_runtime_healthy({"endpoint_info": {}}) is False


def test_health_uses_endpoint_info_health_url() -> None:
    response = MagicMock()
    response.status = 200
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *_: None
    with patch(
        "src.pipeline.execution.stage_planner.urlopen",
        return_value=response,
    ) as mock:
        assert is_inference_runtime_healthy(
            {"endpoint_info": {"health_url": "http://host/health"}}
        ) is True
    mock.assert_called_once()
    assert mock.call_args.args[0] == "http://host/health"


def test_health_falls_back_to_endpoint_url() -> None:
    response = MagicMock()
    response.status = 200
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *_: None
    with patch(
        "src.pipeline.execution.stage_planner.urlopen",
        return_value=response,
    ) as mock:
        assert is_inference_runtime_healthy({"endpoint_url": "http://host/ping"}) is True
    assert mock.call_args.args[0] == "http://host/ping"


def test_health_non_2xx_returns_false() -> None:
    response = MagicMock()
    response.status = 500
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *_: None
    with patch("src.pipeline.execution.stage_planner.urlopen", return_value=response):
        assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is False


def test_health_network_exception_returns_false() -> None:
    with patch(
        "src.pipeline.execution.stage_planner.urlopen",
        side_effect=OSError("network down"),
    ):
        assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is False

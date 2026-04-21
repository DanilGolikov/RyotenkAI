"""Unit tests for ConfigDriftValidator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.config_drift.validator import ConfigDriftValidator
from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineState, StageRunState
from src.utils.result import ConfigDriftError


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _build_config(**overrides: object) -> MagicMock:
    config = MagicMock()
    config.get_active_provider_name.return_value = "single_node"
    config.get_provider_config.return_value = {"host": "localhost"}
    config.model.model_dump.return_value = {"name": "gpt2"}
    config.training.model_dump.return_value = {"type": "sft"}
    config.datasets = {}
    config.inference.model_dump.return_value = {"enabled": False}
    config.evaluation.model_dump.return_value = {"enabled": False}
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build_state(
    *,
    training_critical: str = "",
    late_stage: str = "",
    model_dataset: str = "",
) -> PipelineState:
    return PipelineState(
        schema_version=1,
        logical_run_id="r1",
        run_directory="/tmp/run",
        config_path="/tmp/cfg.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_PENDING,
        training_critical_config_hash=training_critical,
        late_stage_config_hash=late_stage,
        model_dataset_config_hash=model_dataset,
    )


@pytest.fixture
def validator() -> ConfigDriftValidator:
    return ConfigDriftValidator(_build_config())


# -----------------------------------------------------------------------------
# build_config_hashes
# -----------------------------------------------------------------------------


def test_build_config_hashes_returns_three_keys(validator: ConfigDriftValidator) -> None:
    hashes = validator.build_config_hashes()
    assert set(hashes.keys()) == {"training_critical", "late_stage", "model_dataset"}
    for value in hashes.values():
        assert isinstance(value, str) and value != ""


def test_build_config_hashes_is_deterministic(validator: ConfigDriftValidator) -> None:
    assert validator.build_config_hashes() == validator.build_config_hashes()


def test_build_config_hashes_changes_on_training_config(validator: ConfigDriftValidator) -> None:
    first = validator.build_config_hashes()
    validator._config.training.model_dump.return_value = {"type": "dpo"}
    second = validator.build_config_hashes()
    assert first["training_critical"] != second["training_critical"]
    assert first["model_dataset"] != second["model_dataset"]
    # late_stage (inference + evaluation) didn't change
    assert first["late_stage"] == second["late_stage"]


def test_build_config_hashes_provider_change_does_not_change_model_dataset(
    validator: ConfigDriftValidator,
) -> None:
    """Provider swap must not invalidate the model/dataset scope."""
    first = validator.build_config_hashes()
    validator._config.get_active_provider_name.return_value = "runpod"
    validator._config.get_provider_config.return_value = {"api_key": "x"}
    second = validator.build_config_hashes()
    # training_critical changes (provider is part of it), model_dataset stays
    assert first["training_critical"] != second["training_critical"]
    assert first["model_dataset"] == second["model_dataset"]


# -----------------------------------------------------------------------------
# validate_drift
# -----------------------------------------------------------------------------


def _hashes(
    *, training_critical: str = "tc", late_stage: str = "ls", model_dataset: str = "md"
) -> dict[str, str]:
    return {
        "training_critical": training_critical,
        "late_stage": late_stage,
        "model_dataset": model_dataset,
    }


def test_no_drift_returns_none(validator: ConfigDriftValidator) -> None:
    state = _build_state(training_critical="tc", late_stage="ls", model_dataset="md")
    assert (
        validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(),
            resume=True,
        )
        is None
    )


def test_model_dataset_drift_is_fatal(validator: ConfigDriftValidator) -> None:
    state = _build_state(model_dataset="OLD", training_critical="tc", late_stage="ls")
    err = validator.validate_drift(
        state=state,
        start_stage_name=StageNames.DATASET_VALIDATOR,
        config_hashes=_hashes(),
        resume=True,
    )
    assert isinstance(err, ConfigDriftError)
    assert err.details["scope"] == "training_critical"


def test_legacy_state_uses_training_critical_hash(validator: ConfigDriftValidator) -> None:
    """State without model_dataset_config_hash falls back to training_critical."""
    state = _build_state(training_critical="OLD", late_stage="ls", model_dataset="")
    err = validator.validate_drift(
        state=state,
        start_stage_name=StageNames.DATASET_VALIDATOR,
        config_hashes=_hashes(),
        resume=True,
    )
    assert isinstance(err, ConfigDriftError)


def test_legacy_state_matching_training_critical_is_ok(validator: ConfigDriftValidator) -> None:
    state = _build_state(training_critical="tc", late_stage="ls", model_dataset="")
    assert (
        validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(),
            resume=True,
        )
        is None
    )


def test_late_stage_drift_blocks_full_resume(validator: ConfigDriftValidator) -> None:
    state = _build_state(training_critical="tc", late_stage="OLD", model_dataset="md")
    err = validator.validate_drift(
        state=state,
        start_stage_name=StageNames.DATASET_VALIDATOR,
        config_hashes=_hashes(),
        resume=True,
    )
    assert isinstance(err, ConfigDriftError)
    assert err.details["scope"] == "late_stage"


def test_late_stage_drift_allowed_for_inference_restart(validator: ConfigDriftValidator) -> None:
    state = _build_state(training_critical="tc", late_stage="OLD", model_dataset="md")
    assert (
        validator.validate_drift(
            state=state,
            start_stage_name=StageNames.INFERENCE_DEPLOYER,
            config_hashes=_hashes(),
            resume=False,
        )
        is None
    )


def test_late_stage_drift_allowed_for_evaluator_restart(validator: ConfigDriftValidator) -> None:
    state = _build_state(training_critical="tc", late_stage="OLD", model_dataset="md")
    assert (
        validator.validate_drift(
            state=state,
            start_stage_name=StageNames.MODEL_EVALUATOR,
            config_hashes=_hashes(),
            resume=False,
        )
        is None
    )


def test_late_stage_drift_still_fatal_when_resume_true_from_inference(
    validator: ConfigDriftValidator,
) -> None:
    """Even restart-from-Inference-Deployer is blocked when the user sets resume=True."""
    state = _build_state(training_critical="tc", late_stage="OLD", model_dataset="md")
    err = validator.validate_drift(
        state=state,
        start_stage_name=StageNames.INFERENCE_DEPLOYER,
        config_hashes=_hashes(),
        resume=True,
    )
    assert isinstance(err, ConfigDriftError)

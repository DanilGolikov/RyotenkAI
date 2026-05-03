"""Comprehensive tests for ConfigDriftValidator.

7 test categories: positive, negative, boundary, invariants, dependency errors,
regressions (scope-naming bug), combinatorial.
"""

from __future__ import annotations

from itertools import product
from unittest.mock import MagicMock

import pytest

from src.pipeline.config_drift.validator import ConfigDriftValidator
from src.pipeline.stages import StageNames
from src.pipeline.state import PipelineState, StageRunState
from src.utils.result import ConfigDriftError


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _build_config(**overrides) -> MagicMock:
    cfg = MagicMock()
    cfg.get_active_provider_name.return_value = "single_node"
    cfg.get_provider_config.return_value = {"host": "localhost"}
    cfg.model.model_dump.return_value = {"name": "gpt2"}
    cfg.training.model_dump.return_value = {"type": "sft"}
    cfg.datasets = {}
    cfg.inference.model_dump.return_value = {"enabled": False}
    cfg.evaluation.model_dump.return_value = {"enabled": False}
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


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


def _hashes(*, tc: str = "tc", ls: str = "ls", md: str = "md") -> dict[str, str]:
    return {"training_critical": tc, "late_stage": ls, "model_dataset": md}


@pytest.fixture
def validator() -> ConfigDriftValidator:
    return ConfigDriftValidator(_build_config())


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_matching_hashes_no_error(self, validator: ConfigDriftValidator) -> None:
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

    def test_build_hashes_returns_three(self, validator: ConfigDriftValidator) -> None:
        hashes = validator.build_config_hashes()
        assert set(hashes.keys()) == {"training_critical", "late_stage", "model_dataset"}
        assert all(len(v) > 0 for v in hashes.values())

    def test_late_stage_drift_allowed_for_inference_restart(
        self, validator: ConfigDriftValidator
    ) -> None:
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


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_model_dataset_drift_blocks(self, validator: ConfigDriftValidator) -> None:
        state = _build_state(model_dataset="OLD", training_critical="tc", late_stage="ls")
        err = validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(),
            resume=True,
        )
        assert isinstance(err, ConfigDriftError)
        assert err.details["scope"] == "model_dataset"

    def test_late_stage_drift_blocks_full_resume(self, validator: ConfigDriftValidator) -> None:
        state = _build_state(training_critical="tc", late_stage="OLD", model_dataset="md")
        err = validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(),
            resume=True,
        )
        assert isinstance(err, ConfigDriftError)
        assert err.details["scope"] == "late_stage"


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_all_empty_state_hashes_not_considered_drift_if_current_also_empty(
        self, validator: ConfigDriftValidator
    ) -> None:
        """Legacy state with training_critical="" and current hash="" → no drift."""
        state = _build_state(training_critical="", late_stage="", model_dataset="")
        assert (
            validator.validate_drift(
                state=state,
                start_stage_name=StageNames.DATASET_VALIDATOR,
                config_hashes=_hashes(tc="", ls="", md=""),
                resume=False,
            )
            is None
        )

    def test_hash_different_by_single_char(self, validator: ConfigDriftValidator) -> None:
        """Any change (even 1 char) must be detected."""
        state = _build_state(training_critical="tc", late_stage="ls", model_dataset="md")
        err = validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(md="Md"),  # capital M
            resume=True,
        )
        assert err is not None

    def test_late_stage_drift_allowed_for_model_evaluator(
        self, validator: ConfigDriftValidator
    ) -> None:
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


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_build_hashes_deterministic(self, validator: ConfigDriftValidator) -> None:
        """Same config → same hashes."""
        first = validator.build_config_hashes()
        second = validator.build_config_hashes()
        assert first == second

    def test_provider_change_affects_only_training_critical(self) -> None:
        """Invariant: provider change invalidates training_critical but NOT model_dataset."""
        cfg = _build_config()
        validator = ConfigDriftValidator(cfg)
        first = validator.build_config_hashes()
        cfg.get_active_provider_name.return_value = "runpod"
        cfg.get_provider_config.return_value = {"api_key": "x"}
        second = validator.build_config_hashes()
        assert first["model_dataset"] == second["model_dataset"]
        assert first["training_critical"] != second["training_critical"]

    def test_validate_drift_never_mutates_state(self, validator: ConfigDriftValidator) -> None:
        """Invariant: validate_drift is read-only on state."""
        state = _build_state(training_critical="tc", late_stage="ls", model_dataset="md")
        snapshot = (
            state.training_critical_config_hash,
            state.late_stage_config_hash,
            state.model_dataset_config_hash,
        )
        validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(md="OLD"),
            resume=True,
        )
        assert (
            state.training_critical_config_hash,
            state.late_stage_config_hash,
            state.model_dataset_config_hash,
        ) == snapshot


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_config_model_dump_raises_bubbles_up(self) -> None:
        cfg = _build_config()
        cfg.training.model_dump.side_effect = RuntimeError("serialisation broken")
        validator = ConfigDriftValidator(cfg)
        with pytest.raises(RuntimeError):
            validator.build_config_hashes()

    def test_config_missing_required_attr(self) -> None:
        cfg = _build_config()
        del cfg.inference
        validator = ConfigDriftValidator(cfg)
        with pytest.raises(AttributeError):
            validator.build_config_hashes()


# =============================================================================
# 6. REGRESSIONS
# =============================================================================


class TestRegressions:
    def test_regression_scope_matches_hash_that_actually_drifted(
        self, validator: ConfigDriftValidator
    ) -> None:
        """REGRESSION: error used to always say scope='training_critical' even when
        the drift was on model_dataset. Now the scope matches the real hash."""
        state = _build_state(training_critical="tc", late_stage="ls", model_dataset="md_OLD")
        err = validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(),
            resume=True,
        )
        assert isinstance(err, ConfigDriftError)
        assert err.details["scope"] == "model_dataset"
        assert "model_dataset" in err.message

    def test_regression_legacy_state_still_uses_training_critical_scope(
        self, validator: ConfigDriftValidator
    ) -> None:
        """REGRESSION: legacy states without model_dataset hash fall back to
        training_critical — scope name should match that."""
        state = _build_state(training_critical="tc_OLD", late_stage="ls", model_dataset="")
        err = validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(),
            resume=True,
        )
        assert isinstance(err, ConfigDriftError)
        assert err.details["scope"] == "training_critical"


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


# (model_dataset_drift, late_drift, start_stage, resume) → should_error?
_COMBOS: list[tuple[bool, bool, str, bool, bool]] = [
    # No drift → always OK
    (False, False, StageNames.DATASET_VALIDATOR, True, False),
    (False, False, StageNames.DATASET_VALIDATOR, False, False),
    # model_dataset drift is ALWAYS fatal
    (True, False, StageNames.DATASET_VALIDATOR, True, True),
    (True, False, StageNames.INFERENCE_DEPLOYER, False, True),
    (True, True, StageNames.MODEL_EVALUATOR, False, True),
    # late_stage drift: allowed only when start is Inference/Evaluator AND resume=False
    (False, True, StageNames.DATASET_VALIDATOR, True, True),
    (False, True, StageNames.DATASET_VALIDATOR, False, True),
    (False, True, StageNames.INFERENCE_DEPLOYER, False, False),
    (False, True, StageNames.MODEL_EVALUATOR, False, False),
    (False, True, StageNames.INFERENCE_DEPLOYER, True, True),
    (False, True, StageNames.MODEL_EVALUATOR, True, True),
]


@pytest.mark.parametrize(
    ("md_drift", "ls_drift", "start_stage", "resume", "expect_error"), _COMBOS
)
def test_combinatorial_drift_policy(
    validator: ConfigDriftValidator,
    md_drift: bool,
    ls_drift: bool,
    start_stage: str,
    resume: bool,
    expect_error: bool,
) -> None:
    state = _build_state(
        training_critical="tc",
        late_stage="ls_OLD" if ls_drift else "ls",
        model_dataset="md_OLD" if md_drift else "md",
    )
    err = validator.validate_drift(
        state=state,
        start_stage_name=start_stage,
        config_hashes=_hashes(),
        resume=resume,
    )
    if expect_error:
        assert err is not None
    else:
        assert err is None


@pytest.mark.parametrize("prefix_len", [1, 3, 8, 16, 64])
def test_combinatorial_hash_length_tolerance(
    validator: ConfigDriftValidator, prefix_len: int
) -> None:
    """Hashes of various lengths should compare correctly (string equality)."""
    h = "x" * prefix_len
    state = _build_state(training_critical=h, late_stage="ls", model_dataset=h)
    assert (
        validator.validate_drift(
            state=state,
            start_stage_name=StageNames.DATASET_VALIDATOR,
            config_hashes=_hashes(tc=h, md=h),
            resume=False,
        )
        is None
    )

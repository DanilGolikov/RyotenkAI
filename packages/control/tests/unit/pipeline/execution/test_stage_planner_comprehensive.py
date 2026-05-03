"""Comprehensive tests for StagePlanner and is_inference_runtime_healthy.

Tests are organised into seven categories per the review checklist:

1. ``TestPositive``      — happy-path behaviour.
2. ``TestNegative``      — error paths (invalid inputs, unknown names).
3. ``TestBoundary``      — boundary values (1, N, empty, whitespace, unicode).
4. ``TestInvariants``    — class invariants hold across calls.
5. ``TestDependencyErrors`` — how the component reacts when its collaborators fail.
6. ``TestRegressions``   — specific bugs found in review (bool-is-int, magic ``[:4]``).
7. ``TestCombinatorial`` — parametrised cross-product of config + start_stage.
"""

from __future__ import annotations

import socket
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


_CANONICAL_ORDER = [
    StageNames.DATASET_VALIDATOR,
    StageNames.GPU_DEPLOYER,
    StageNames.TRAINING_MONITOR,
    StageNames.MODEL_RETRIEVER,
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
]

_MANDATORY = _CANONICAL_ORDER[:4]


def _mock_stage(name: str) -> MagicMock:
    stage = MagicMock()
    stage.stage_name = name
    return stage


def _build_config(*, inference: bool = False, evaluation: bool = False) -> MagicMock:
    config = MagicMock()
    config.inference.enabled = inference
    config.evaluation.enabled = evaluation
    return config


@pytest.fixture
def stages() -> list[MagicMock]:
    return [_mock_stage(name) for name in _CANONICAL_ORDER]


@pytest.fixture
def planner(stages: list[MagicMock]) -> StagePlanner:
    return StagePlanner(stages, _build_config())


def _build_state() -> PipelineState:
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


def _build_attempt(**stage_statuses: str) -> PipelineAttemptState:
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
    return attempt


# =============================================================================
# 1. POSITIVE — happy path
# =============================================================================


class TestPositive:
    def test_get_stage_index_returns_zero_for_first(self, planner: StagePlanner) -> None:
        assert planner.get_stage_index(StageNames.DATASET_VALIDATOR) == 0

    def test_normalize_by_name(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref(StageNames.MODEL_EVALUATOR) == StageNames.MODEL_EVALUATOR

    def test_normalize_by_index(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref(1) == StageNames.DATASET_VALIDATOR

    def test_compute_enabled_default_only_mandatory(self, stages: list[MagicMock]) -> None:
        planner = StagePlanner(stages, _build_config())
        assert planner.compute_enabled_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR) == _MANDATORY

    def test_derive_resume_no_attempts_returns_first(self, planner: StagePlanner) -> None:
        assert planner.derive_resume_stage(_build_state()) == StageNames.DATASET_VALIDATOR

    def test_validate_prereq_no_error_when_not_restart(self, planner: StagePlanner) -> None:
        assert (
            planner.validate_stage_prerequisites(
                stage_name=StageNames.DATASET_VALIDATOR,
                start_stage_name=StageNames.DATASET_VALIDATOR,
                context={},
            )
            is None
        )


# =============================================================================
# 2. NEGATIVE — error paths
# =============================================================================


class TestNegative:
    def test_normalize_none_raises_value_error(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="required"):
            planner.normalize_stage_ref(None)

    def test_normalize_unknown_string_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="Unknown stage reference"):
            planner.normalize_stage_ref("nonexistent-stage")

    def test_normalize_empty_string_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="empty"):
            planner.normalize_stage_ref("")

    def test_normalize_whitespace_string_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="empty"):
            planner.normalize_stage_ref("   \t\n  ")

    def test_get_stage_index_unknown_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="Unknown stage name"):
            planner.get_stage_index("Imaginary Stage")

    def test_prereq_training_monitor_missing_gpu_workspace(self, planner: StagePlanner) -> None:
        err = planner.validate_stage_prerequisites(
            stage_name=StageNames.TRAINING_MONITOR,
            start_stage_name=StageNames.TRAINING_MONITOR,
            context={StageNames.GPU_DEPLOYER: {"ssh_host": "h", "ssh_port": 22}},
        )
        assert err is not None
        assert err.code == "MISSING_TRAINING_MONITOR_PREREQUISITES"

    def test_prereq_training_monitor_non_dict_gpu_context(self, planner: StagePlanner) -> None:
        err = planner.validate_stage_prerequisites(
            stage_name=StageNames.TRAINING_MONITOR,
            start_stage_name=StageNames.TRAINING_MONITOR,
            context={StageNames.GPU_DEPLOYER: "garbage"},
        )
        assert err is not None

    def test_prereq_inference_missing_retriever(self, planner: StagePlanner) -> None:
        err = planner.validate_stage_prerequisites(
            stage_name=StageNames.INFERENCE_DEPLOYER,
            start_stage_name=StageNames.INFERENCE_DEPLOYER,
            context={StageNames.MODEL_RETRIEVER: {}},
        )
        assert err.code == "MISSING_INFERENCE_PREREQUISITES"


# =============================================================================
# 3. BOUNDARY — boundary inputs
# =============================================================================


class TestBoundary:
    def test_normalize_int_first_index(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref(1) == _CANONICAL_ORDER[0]

    def test_normalize_int_last_index(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref(6) == _CANONICAL_ORDER[-1]

    def test_normalize_int_zero_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="out of range"):
            planner.normalize_stage_ref(0)

    def test_normalize_int_negative_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="out of range"):
            planner.normalize_stage_ref(-1)

    def test_normalize_int_past_last_raises(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="out of range"):
            planner.normalize_stage_ref(7)

    def test_normalize_str_digit_first(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref("1") == _CANONICAL_ORDER[0]

    def test_normalize_str_digit_last(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref("6") == _CANONICAL_ORDER[-1]

    def test_normalize_str_digit_out_of_range(self, planner: StagePlanner) -> None:
        with pytest.raises(ValueError, match="out of range"):
            planner.normalize_stage_ref("0")
        with pytest.raises(ValueError, match="out of range"):
            planner.normalize_stage_ref("99")

    def test_normalize_leading_trailing_whitespace(self, planner: StagePlanner) -> None:
        assert planner.normalize_stage_ref("  Inference Deployer  ") == StageNames.INFERENCE_DEPLOYER

    def test_derive_resume_all_complete_returns_none(self, planner: StagePlanner) -> None:
        attempt = _build_attempt(**{name: StageRunState.STATUS_COMPLETED for name in _CANONICAL_ORDER})
        state = _build_state()
        state.attempts.append(attempt)
        assert planner.derive_resume_stage(state) is None

    def test_empty_stages_list_raises_on_index_0(self) -> None:
        planner = StagePlanner(stages=[], config=_build_config())
        with pytest.raises(ValueError, match="Unknown stage name"):
            planner.get_stage_index(StageNames.DATASET_VALIDATOR)

    def test_single_stage_only(self) -> None:
        planner = StagePlanner(stages=[_mock_stage(StageNames.DATASET_VALIDATOR)], config=_build_config())
        assert planner.normalize_stage_ref(1) == StageNames.DATASET_VALIDATOR
        with pytest.raises(ValueError, match="out of range"):
            planner.normalize_stage_ref(2)


# =============================================================================
# 4. INVARIANTS — properties that hold across calls
# =============================================================================


class TestInvariants:
    def test_mandatory_stages_always_enabled_regardless_of_config(
        self, stages: list[MagicMock]
    ) -> None:
        """Invariant: mandatory stages appear in every enabled list."""
        for inf in (True, False):
            for evl in (True, False):
                planner = StagePlanner(stages, _build_config(inference=inf, evaluation=evl))
                enabled = planner.compute_enabled_stage_names(start_stage_name=_CANONICAL_ORDER[0])
                for mand in _MANDATORY:
                    assert mand in enabled, f"{mand} missing with inference={inf}, evaluation={evl}"

    def test_enabled_list_is_prefix_of_canonical_order(self, stages: list[MagicMock]) -> None:
        """Invariant: enabled stages preserve canonical order."""
        planner = StagePlanner(stages, _build_config(inference=True, evaluation=True))
        enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR)
        # each name must appear in the canonical order index monotonically
        indices = [_CANONICAL_ORDER.index(n) for n in enabled]
        assert indices == sorted(indices)

    def test_enabled_list_has_no_duplicates(self, stages: list[MagicMock]) -> None:
        """Invariant: forced stages aren't appended twice."""
        planner = StagePlanner(stages, _build_config(inference=True))
        enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER)
        assert len(enabled) == len(set(enabled))

    def test_normalize_is_idempotent_on_canonical_names(self, planner: StagePlanner) -> None:
        """Invariant: normalizing a canonical name returns itself."""
        for name in _CANONICAL_ORDER:
            assert planner.normalize_stage_ref(name) == name

    def test_get_stage_index_matches_canonical_position(self, planner: StagePlanner) -> None:
        """Invariant: get_stage_index matches the canonical order."""
        for i, name in enumerate(_CANONICAL_ORDER):
            assert planner.get_stage_index(name) == i


# =============================================================================
# 5. DEPENDENCY ERRORS — broken collaborators
# =============================================================================


class TestDependencyErrors:
    def test_config_missing_inference_attribute_bubbles_up(self, stages: list[MagicMock]) -> None:
        """Invariant: we don't silently swallow misconfigured Config objects."""
        cfg = MagicMock()
        # Explicitly nuke the attribute: MagicMock auto-creates them otherwise.
        del cfg.inference
        planner = StagePlanner(stages, cfg)
        with pytest.raises(AttributeError):
            planner.forced_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER)

    def test_health_check_handles_socket_timeout(self) -> None:
        with patch(
            "src.pipeline.execution.stage_planner.urlopen", side_effect=socket.timeout("slow")
        ):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is False

    def test_health_check_handles_value_error(self) -> None:
        with patch(
            "src.pipeline.execution.stage_planner.urlopen", side_effect=ValueError("bad url")
        ):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is False

    def test_health_check_handles_response_without_status_attr(self) -> None:
        response = MagicMock()
        # Pretend the response object doesn't expose `.status`; fallback to _HTTP_OK_MIN.
        del response.status
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=response):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is True

    def test_prereq_model_evaluator_with_unhealthy_runtime_failing_urlopen(
        self, planner: StagePlanner
    ) -> None:
        with patch(
            "src.pipeline.execution.stage_planner.urlopen",
            side_effect=ConnectionError("boom"),
        ):
            err = planner.validate_stage_prerequisites(
                stage_name=StageNames.MODEL_EVALUATOR,
                start_stage_name=StageNames.MODEL_EVALUATOR,
                context={StageNames.INFERENCE_DEPLOYER: {"endpoint_url": "http://x"}},
            )
        assert err is not None
        assert err.code == "INFERENCE_RUNTIME_NOT_HEALTHY"


# =============================================================================
# 6. REGRESSIONS — specific bugs fixed during review
# =============================================================================


class TestRegressions:
    def test_regression_true_as_int_rejected(self, planner: StagePlanner) -> None:
        """REGRESSION: ``True is int`` — used to silently resolve to stage #1."""
        with pytest.raises(TypeError, match="bool"):
            planner.normalize_stage_ref(True)

    def test_regression_false_as_int_rejected(self, planner: StagePlanner) -> None:
        """REGRESSION: ``False is int`` → used to fall through to ``0 out of range``."""
        with pytest.raises(TypeError, match="bool"):
            planner.normalize_stage_ref(False)

    def test_regression_magic_slice_gone_shuffled_stages_still_work(self) -> None:
        """REGRESSION: old ``self._stages[:4]`` tied mandatory-ness to position.

        After the fix, even if someone builds StagePlanner with stages in a
        funky order (e.g. evaluator first), compute_enabled_stage_names still
        produces the canonical mandatory list.
        """
        weird_order = [
            _mock_stage(StageNames.MODEL_EVALUATOR),
            _mock_stage(StageNames.DATASET_VALIDATOR),
            _mock_stage(StageNames.GPU_DEPLOYER),
            _mock_stage(StageNames.TRAINING_MONITOR),
            _mock_stage(StageNames.MODEL_RETRIEVER),
            _mock_stage(StageNames.INFERENCE_DEPLOYER),
        ]
        planner = StagePlanner(weird_order, _build_config())
        enabled = planner.compute_enabled_stage_names(start_stage_name=StageNames.DATASET_VALIDATOR)
        assert enabled == _MANDATORY  # canonical, not reshuffled

    def test_regression_mocked_stages_dont_break_enabled_computation(self) -> None:
        """REGRESSION: stages[i].stage_name being MagicMock used to break the
        new name-based enabled logic. Canonical order removes that coupling."""
        stages = [MagicMock() for _ in range(6)]  # .stage_name not configured — raw MagicMock
        planner = StagePlanner(stages, _build_config(inference=True))
        enabled = planner.compute_enabled_stage_names(
            start_stage_name=StageNames.INFERENCE_DEPLOYER
        )
        assert StageNames.INFERENCE_DEPLOYER in enabled


# =============================================================================
# 7. COMBINATORIAL — parametrised cross-product
# =============================================================================


@pytest.mark.parametrize("inference_on", [True, False])
@pytest.mark.parametrize("evaluation_on", [True, False])
@pytest.mark.parametrize(
    "start_stage",
    [
        StageNames.DATASET_VALIDATOR,
        StageNames.GPU_DEPLOYER,
        StageNames.TRAINING_MONITOR,
        StageNames.MODEL_RETRIEVER,
        StageNames.INFERENCE_DEPLOYER,
        StageNames.MODEL_EVALUATOR,
    ],
)
def test_combinatorial_enabled_contents(
    stages: list[MagicMock], inference_on: bool, evaluation_on: bool, start_stage: str
) -> None:
    """24 combinations of (inference, evaluation, start_stage) — no exceptions."""
    planner = StagePlanner(stages, _build_config(inference=inference_on, evaluation=evaluation_on))
    enabled = planner.compute_enabled_stage_names(start_stage_name=start_stage)

    # Mandatory always present
    for mand in _MANDATORY:
        assert mand in enabled

    # Inference appears iff its flag on, or it's the start stage
    assert (StageNames.INFERENCE_DEPLOYER in enabled) == (
        inference_on or start_stage == StageNames.INFERENCE_DEPLOYER
    )
    # Evaluator — symmetric
    assert (StageNames.MODEL_EVALUATOR in enabled) == (
        evaluation_on or start_stage == StageNames.MODEL_EVALUATOR
    )


@pytest.mark.parametrize(
    ("status", "expected_resume"),
    [
        (StageRunState.STATUS_FAILED, StageNames.GPU_DEPLOYER),
        (StageRunState.STATUS_INTERRUPTED, StageNames.GPU_DEPLOYER),
        (StageRunState.STATUS_PENDING, StageNames.GPU_DEPLOYER),
        (StageRunState.STATUS_RUNNING, StageNames.GPU_DEPLOYER),
        (StageRunState.STATUS_STALE, StageNames.GPU_DEPLOYER),
        (StageRunState.STATUS_COMPLETED, StageNames.TRAINING_MONITOR),
        (StageRunState.STATUS_SKIPPED, StageNames.TRAINING_MONITOR),
    ],
)
def test_combinatorial_resume_derivation(
    planner: StagePlanner, status: str, expected_resume: str
) -> None:
    """Cross-check all StageRunState statuses — which advance the resume pointer vs halt it."""
    attempt = _build_attempt(
        **{
            StageNames.DATASET_VALIDATOR: StageRunState.STATUS_COMPLETED,
            StageNames.GPU_DEPLOYER: status,
        }
    )
    state = _build_state()
    state.attempts.append(attempt)
    assert planner.derive_resume_stage(state) == expected_resume


# -----------------------------------------------------------------------------
# is_inference_runtime_healthy — dedicated positive/negative/boundary trio
# -----------------------------------------------------------------------------


class TestHealthCheckPositiveNegative:
    def test_positive_200_ok(self) -> None:
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=resp):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is True

    def test_positive_299_ok(self) -> None:
        resp = MagicMock()
        resp.status = 299
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=resp):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is True

    def test_boundary_status_399_ok(self) -> None:
        """3xx redirects are still treated as healthy (upper boundary exclusive at 400)."""
        resp = MagicMock()
        resp.status = 399
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=resp):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is True

    def test_boundary_status_400_unhealthy(self) -> None:
        resp = MagicMock()
        resp.status = 400
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=resp):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is False

    def test_boundary_status_199_unhealthy(self) -> None:
        resp = MagicMock()
        resp.status = 199
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=resp):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is False

    def test_negative_empty_url_returns_false(self) -> None:
        assert is_inference_runtime_healthy({"endpoint_url": ""}) is False

    def test_negative_none_url_returns_false(self) -> None:
        assert is_inference_runtime_healthy({"endpoint_url": None}) is False

    def test_boundary_int_status_coercion(self) -> None:
        """status is coerced to int; '200' (str) → int('200') → 200."""
        resp = MagicMock()
        resp.status = "200"
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda self, *_: None
        with patch("src.pipeline.execution.stage_planner.urlopen", return_value=resp):
            assert is_inference_runtime_healthy({"endpoint_url": "http://x"}) is True

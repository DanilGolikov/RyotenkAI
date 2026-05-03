"""Comprehensive tests for :mod:`src.pipeline.launch.launch_preparator`.

Coverage split by category (required by project policy):

1. **Positive**           — happy-path prepare() builds PreparedAttempt end-to-end.
2. **Negative**           — missing state, unknown stage, empty stages list.
3. **Boundary**           — restart on first/last stage, empty lineage, no run_dir.
4. **Invariants**         — controller adopts state; state_store cached; single
                              rejection write.
5. **Dependency errors**  — corrupted state file, drift validator failures.
6. **Regressions**        — rejection uses cached state_store; run_dir resolution.
7. **Combinatorial**      — resume×restart×state matrix.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.pipeline.launch import (
    LaunchPreparationError,
    LaunchPreparator,
    PreparedAttempt,
)
from src.pipeline.state import (
    AttemptController,
    PipelineAttemptState,
    PipelineState,
    PipelineStateLoadError,
    PipelineStateStore,
    StageRunState,
)
from src.utils.result import AppError

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage(stage_name: str) -> MagicMock:
    stage = MagicMock()
    stage.stage_name = stage_name
    return stage


STAGE_A = "Stage A"
STAGE_B = "Stage B"
STAGE_C = "Stage C"


def _build_stages() -> list[MagicMock]:
    return [_make_stage(STAGE_A), _make_stage(STAGE_B), _make_stage(STAGE_C)]


def _build_stage_planner(stages: list[MagicMock]) -> MagicMock:
    """Fake StagePlanner that answers enough queries for prepare() to run.

    Uses a MagicMock with realistic return values so tests don't have to
    construct a real StagePlanner (which needs a full PipelineConfig).
    """
    planner = MagicMock()
    names = [s.stage_name for s in stages]

    def _normalize(ref: str | int | None) -> str:
        if isinstance(ref, int):
            return names[ref - 1]
        if isinstance(ref, str) and ref.isdigit():
            return names[int(ref) - 1]
        if ref in names:
            return ref
        raise ValueError(f"Unknown stage reference: {ref!r}")

    def _get_idx(name: str) -> int:
        return names.index(name)

    def _enabled(**_: str) -> list[str]:
        return list(names)

    def _forced(**_: str) -> set[str]:
        return set()

    planner.normalize_stage_ref.side_effect = _normalize
    planner.get_stage_index.side_effect = _get_idx
    planner.compute_enabled_stage_names.side_effect = _enabled
    planner.forced_stage_names.side_effect = _forced
    planner.derive_resume_stage.side_effect = lambda state: STAGE_B if state.attempts else names[0]
    return planner


def _build_config_drift(*, drift: AppError | None = None) -> MagicMock:
    cd = MagicMock()
    cd.validate_drift.return_value = drift
    cd.build_config_hashes.return_value = {
        "training_critical": "t_hash",
        "late_stage": "l_hash",
        "model_dataset": "",
    }
    return cd


def _build_preparator(
    tmp_path: Path,
    *,
    stages: list[MagicMock] | None = None,
    drift: AppError | None = None,
    save_fn: MagicMock | None = None,
) -> tuple[LaunchPreparator, AttemptController, MagicMock, MagicMock]:
    """Build a ready-to-use preparator + the dependencies needed for assertions."""
    stages = stages or _build_stages()
    run_ctx = SimpleNamespace(name="run_x", run_id="rid_x")
    settings = SimpleNamespace(runs_base_dir=tmp_path / "runs")
    stage_planner = _build_stage_planner(stages)
    config_drift = _build_config_drift(drift=drift)
    save_fn = save_fn or MagicMock()
    attempt_controller = AttemptController(save_fn=save_fn, run_ctx=run_ctx)

    preparator = LaunchPreparator(
        config_path=tmp_path / "cfg.yaml",
        run_ctx=run_ctx,
        settings=settings,
        stages=stages,
        stage_planner=stage_planner,
        config_drift=config_drift,
        attempt_controller=attempt_controller,
    )
    return preparator, attempt_controller, stage_planner, config_drift


def _config_hashes() -> dict[str, str]:
    return {"training_critical": "t_hash", "late_stage": "l_hash", "model_dataset": ""}


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_fresh_run_creates_state_and_prepared_attempt(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logical_run_1"
        preparator, controller, _, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )

        assert isinstance(prepared, PreparedAttempt)
        assert prepared.start_stage_name == STAGE_A
        assert prepared.start_idx == 0
        assert prepared.stop_idx == 3
        assert prepared.requested_action == "fresh"
        assert prepared.effective_action == "fresh"
        assert (run_dir / "pipeline_state.json").exists()
        assert controller.has_state  # state adopted
        assert isinstance(prepared.attempt, PipelineAttemptState)

    def test_resume_loads_existing_state(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logical_run_resume"
        store = PipelineStateStore(run_dir)
        state = store.init_state(
            logical_run_id="logical_run_resume",
            config_path=str(tmp_path / "cfg.yaml"),
            training_critical_config_hash="t_hash",
            late_stage_config_hash="l_hash",
        )
        state.attempts.append(_make_failed_attempt())
        store.save(state)

        preparator, _, planner, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=True,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )

        assert prepared.requested_action == "resume"
        assert prepared.effective_action == "auto_resume"
        assert prepared.start_stage_name == STAGE_B  # planner.derive_resume_stage
        planner.derive_resume_stage.assert_called_once()

    def test_restart_from_explicit_stage(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logical_run_restart"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="logical_run_restart",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, _, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=STAGE_C,
            config_hashes=_config_hashes(),
        )

        assert prepared.requested_action == "restart"
        assert prepared.effective_action == "restart"
        assert prepared.start_stage_name == STAGE_C
        assert prepared.start_idx == 2

    def test_prepared_attempt_is_frozen(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logical_run_freeze"
        preparator, _, _, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )
        with pytest.raises((AttributeError, TypeError)):
            prepared.start_idx = 999  # type: ignore[misc]

    def test_cached_state_store_accessible(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logical_run_cache"
        preparator, _, _, _ = _build_preparator(tmp_path)

        preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )

        assert preparator.last_state_store is not None
        assert preparator.last_run_directory == run_dir.resolve()
        assert preparator.last_logical_run_id == run_dir.name

    def test_record_launch_rejection_emits_single_persist(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logical_run_reject"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="logical_run_reject",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="old_hash",
                late_stage_config_hash="l_hash",
            )
        )

        drift_err = AppError(message="training_critical config changed", code="DRIFT")
        save_fn = MagicMock()
        preparator, controller, _, _ = _build_preparator(tmp_path, drift=drift_err, save_fn=save_fn)

        with pytest.raises(LaunchPreparationError):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage=STAGE_B,
                config_hashes=_config_hashes(),
            )

        # The preparation raised — rejection path needs to emit exactly one
        # persist with a FAILED pipeline_status.
        launch_err = LaunchPreparationError(
            drift_err,
            state=controller.state,
            requested_action="restart",
            effective_action="restart",
            start_stage_name=STAGE_B,
        )
        save_fn.reset_mock()
        preparator.record_launch_rejection(
            launch_error=launch_err, config_hashes=_config_hashes()
        )
        assert save_fn.call_count == 1
        persisted_state = save_fn.call_args.args[0]
        assert persisted_state.pipeline_status == StageRunState.STATUS_FAILED
        assert persisted_state.active_attempt_id is None
        assert persisted_state.attempts[-1].status == StageRunState.STATUS_FAILED
        assert persisted_state.attempts[-1].error == "training_critical config changed"


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_resume_without_state_raises_load_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "missing_run"
        preparator, _, _, _ = _build_preparator(tmp_path)

        with pytest.raises(PipelineStateLoadError, match=r"Missing pipeline_state\.json"):
            preparator.prepare(
                run_dir=run_dir,
                resume=True,
                restart_from_stage=None,
                config_hashes=_config_hashes(),
            )

    def test_restart_without_state_raises_load_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "missing_restart"
        preparator, _, _, _ = _build_preparator(tmp_path)

        with pytest.raises(PipelineStateLoadError):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage=STAGE_B,
                config_hashes=_config_hashes(),
            )

    def test_drift_validation_raises_launch_preparation_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "drift_run"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="drift_run",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="old",
                late_stage_config_hash="l",
            )
        )
        drift = AppError(message="training_critical config changed", code="DRIFT")
        preparator, _, _, _ = _build_preparator(tmp_path, drift=drift)

        with pytest.raises(LaunchPreparationError, match="training_critical config changed"):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage=STAGE_B,
                config_hashes=_config_hashes(),
            )

    def test_require_state_store_before_prepare_raises(self, tmp_path: Path) -> None:
        preparator, _, _, _ = _build_preparator(tmp_path)
        with pytest.raises(RuntimeError, match="state_store has not been created"):
            preparator._require_state_store()

    def test_require_run_directory_before_prepare_raises(self, tmp_path: Path) -> None:
        preparator, _, _, _ = _build_preparator(tmp_path)
        with pytest.raises(RuntimeError, match="run_directory has not been resolved"):
            preparator._require_run_directory()


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_restart_on_first_stage(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "first_stage"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="first_stage",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, _, _ = _build_preparator(tmp_path)
        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=STAGE_A,
            config_hashes=_config_hashes(),
        )
        assert prepared.start_idx == 0
        assert prepared.stop_idx == 3

    def test_restart_on_last_stage(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "last_stage"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="last_stage",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, _, _ = _build_preparator(tmp_path)
        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=STAGE_C,
            config_hashes=_config_hashes(),
        )
        assert prepared.start_idx == 2
        assert prepared.stop_idx == 3  # only one stage will run

    def test_no_run_dir_uses_settings_base(self, tmp_path: Path) -> None:
        # When run_dir is None and state doesn't exist, preparator uses
        # settings.runs_base_dir / run_ctx.name.
        preparator, _, _, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=None,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )
        assert prepared.run_directory == (tmp_path / "runs" / "run_x").resolve()
        assert prepared.logical_run_id == "run_x"  # from run_ctx.name

    def test_fresh_run_with_explicit_dir_uses_dir_name_as_logical_id(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "my_explicit_run"
        preparator, _, _, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )
        assert prepared.logical_run_id == "my_explicit_run"

    def test_numeric_restart_reference_resolved(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "numeric_ref"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="numeric_ref",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, _, _ = _build_preparator(tmp_path)

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=2,  # Stage B
            config_hashes=_config_hashes(),
        )
        assert prepared.start_stage_name == STAGE_B


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_prepare_adopts_state_into_controller(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "adopt_state"
        preparator, controller, _, _ = _build_preparator(tmp_path)

        assert not controller.has_state
        preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )
        assert controller.has_state

    def test_config_hashes_stamped_onto_state(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "stamp"
        preparator, controller, _, _ = _build_preparator(tmp_path)

        hashes = {"training_critical": "NEW_T", "late_stage": "NEW_L", "model_dataset": "NEW_M"}
        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=None,
            config_hashes=hashes,
        )
        assert prepared.state.training_critical_config_hash == "NEW_T"
        assert prepared.state.late_stage_config_hash == "NEW_L"
        assert prepared.state.model_dataset_config_hash == "NEW_M"
        # Same live object on controller — stamp propagated.
        assert controller.state.training_critical_config_hash == "NEW_T"

    def test_state_store_cached_even_when_prepare_raises(
        self, tmp_path: Path
    ) -> None:
        """Regression: record_launch_rejection needs to read state_store after raise."""
        run_dir = tmp_path / "cache_on_raise"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="cache_on_raise",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t",
                late_stage_config_hash="l",
            )
        )
        drift = AppError(message="drift!", code="DRIFT")
        preparator, _, _, _ = _build_preparator(tmp_path, drift=drift)

        with pytest.raises(LaunchPreparationError):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage=STAGE_B,
                config_hashes=_config_hashes(),
            )
        assert preparator.last_state_store is not None
        assert preparator.last_run_directory == run_dir.resolve()

    def test_attempt_directory_is_per_attempt_no(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "per_attempt"
        store = PipelineStateStore(run_dir)
        state = store.init_state(
            logical_run_id="per_attempt",
            config_path=str(tmp_path / "cfg.yaml"),
            training_critical_config_hash="t_hash",
            late_stage_config_hash="l_hash",
        )
        # Seed one existing attempt so next one is attempt_2
        state.attempts.append(_make_failed_attempt(attempt_no=1))
        store.save(state)

        preparator, _, _, _ = _build_preparator(tmp_path)
        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=STAGE_B,
            config_hashes=_config_hashes(),
        )
        assert prepared.attempt_directory == run_dir / "attempts" / "attempt_2"

    def test_record_launch_rejection_without_cached_store_is_noop(
        self, tmp_path: Path
    ) -> None:
        """Very-early failure before state_store is created leaves rejection a no-op."""
        save_fn = MagicMock()
        preparator, _, _, _ = _build_preparator(tmp_path, save_fn=save_fn)
        # Construct a rejection-carrying error that never went through prepare().
        err = LaunchPreparationError(
            AppError(message="never got a store", code="E"),
            state=None,
        )
        preparator.record_launch_rejection(launch_error=err, config_hashes=_config_hashes())
        save_fn.assert_not_called()


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_corrupted_state_file_raises_load_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "corrupted"
        run_dir.mkdir(parents=True)
        (run_dir / "pipeline_state.json").write_text("not valid json")
        preparator, _, _, _ = _build_preparator(tmp_path)

        with pytest.raises(PipelineStateLoadError):
            preparator.prepare(
                run_dir=run_dir,
                resume=True,
                restart_from_stage=None,
                config_hashes=_config_hashes(),
            )

    def test_planner_normalize_raises_propagates(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "bad_ref"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="bad_ref",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, _, _ = _build_preparator(tmp_path)

        with pytest.raises(ValueError, match="Unknown stage reference"):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage="NOT_A_STAGE",
                config_hashes=_config_hashes(),
            )

    def test_save_fn_failure_propagates_through_rejection(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "save_fail"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="save_fail",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        def _fail(_state: PipelineState) -> None:
            raise OSError("disk full")

        drift = AppError(message="drift", code="D")
        preparator, controller, _, _ = _build_preparator(tmp_path, drift=drift, save_fn=_fail)  # type: ignore[arg-type]

        with pytest.raises(LaunchPreparationError):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage=STAGE_B,
                config_hashes=_config_hashes(),
            )
        err = LaunchPreparationError(
            drift,
            state=controller.state,
            requested_action="restart",
            effective_action="restart",
            start_stage_name=STAGE_B,
        )
        with pytest.raises(OSError, match="disk full"):
            preparator.record_launch_rejection(launch_error=err, config_hashes=_config_hashes())


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_rejection_does_not_overwrite_successful_attempt(
        self, tmp_path: Path
    ) -> None:
        """Regression: rejection appends a NEW attempt instead of clobbering
        any previously-successful one on the same state."""
        run_dir = tmp_path / "rejection_append"
        store = PipelineStateStore(run_dir)
        state = store.init_state(
            logical_run_id="rejection_append",
            config_path=str(tmp_path / "cfg.yaml"),
            training_critical_config_hash="t_hash",
            late_stage_config_hash="l_hash",
        )
        completed = _make_completed_attempt(attempt_no=1)
        state.attempts.append(completed)
        store.save(state)

        drift = AppError(message="drift!", code="D")
        save_fn = MagicMock()
        preparator, controller, _, _ = _build_preparator(
            tmp_path, drift=drift, save_fn=save_fn
        )
        with pytest.raises(LaunchPreparationError):
            preparator.prepare(
                run_dir=run_dir,
                resume=False,
                restart_from_stage=STAGE_B,
                config_hashes=_config_hashes(),
            )
        err = LaunchPreparationError(
            drift,
            state=controller.state,
            requested_action="restart",
            effective_action="restart",
            start_stage_name=STAGE_B,
        )
        preparator.record_launch_rejection(launch_error=err, config_hashes=_config_hashes())

        assert len(controller.state.attempts) == 2
        assert controller.state.attempts[0].status == StageRunState.STATUS_COMPLETED
        assert controller.state.attempts[1].status == StageRunState.STATUS_FAILED

    def test_fresh_run_without_run_dir_preserves_run_ctx_name(
        self, tmp_path: Path
    ) -> None:
        """Regression: the settings-based fallback uses run_ctx.name, not a random UUID."""
        preparator, _, _, _ = _build_preparator(tmp_path)
        prepared = preparator.prepare(
            run_dir=None,
            resume=False,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )
        assert prepared.run_directory.name == "run_x"
        assert prepared.logical_run_id == "run_x"

    def test_resume_when_no_failed_attempts_falls_through_to_planner(
        self, tmp_path: Path
    ) -> None:
        """Regression: resume with a clean state asks the planner for the
        resume stage (it may return None in some configurations)."""
        run_dir = tmp_path / "clean_resume"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="clean_resume",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, planner, _ = _build_preparator(tmp_path)
        planner.derive_resume_stage.side_effect = lambda _state: STAGE_A  # override

        prepared = preparator.prepare(
            run_dir=run_dir,
            resume=True,
            restart_from_stage=None,
            config_hashes=_config_hashes(),
        )
        assert prepared.start_stage_name == STAGE_A

    def test_resume_when_planner_returns_none_raises_prep_error(
        self, tmp_path: Path
    ) -> None:
        """Regression: 'no resumable stage' path raises LaunchPreparationError
        with the expected RESUME_NOT_AVAILABLE code."""
        run_dir = tmp_path / "no_resume"
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id="no_resume",
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
        preparator, _, planner, _ = _build_preparator(tmp_path)
        planner.derive_resume_stage.side_effect = lambda _state: None

        with pytest.raises(LaunchPreparationError) as excinfo:
            preparator.prepare(
                run_dir=run_dir,
                resume=True,
                restart_from_stage=None,
                config_hashes=_config_hashes(),
            )
        assert excinfo.value.app_error.code == "RESUME_NOT_AVAILABLE"


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


def _make_failed_attempt(*, attempt_no: int = 1) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id=f"a_{attempt_no}",
        attempt_no=attempt_no,
        runtime_name="rt",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_FAILED,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
    )


def _make_completed_attempt(*, attempt_no: int = 1) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id=f"a_{attempt_no}",
        attempt_no=attempt_no,
        runtime_name="rt",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_COMPLETED,
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
    )


@pytest.mark.parametrize(
    ("has_state", "resume", "restart"),
    [
        (False, False, None),  # fresh run
        (True, False, STAGE_B),  # restart
        (True, True, None),  # resume
        (True, False, None),  # fresh on existing dir (drifts may apply)
    ],
)
def test_action_matrix(
    tmp_path: Path,
    has_state: bool,
    resume: bool,
    restart: str | None,
) -> None:
    run_dir = tmp_path / f"matrix_{has_state}_{resume}_{restart}"
    if has_state:
        store = PipelineStateStore(run_dir)
        store.save(
            store.init_state(
                logical_run_id=run_dir.name,
                config_path=str(tmp_path / "cfg.yaml"),
                training_critical_config_hash="t_hash",
                late_stage_config_hash="l_hash",
            )
        )
    preparator, _, _, _ = _build_preparator(tmp_path)

    if not has_state and (resume or restart is not None):
        # resume/restart without state must raise
        with pytest.raises(PipelineStateLoadError):
            preparator.prepare(
                run_dir=run_dir,
                resume=resume,
                restart_from_stage=restart,
                config_hashes=_config_hashes(),
            )
        return

    prepared = preparator.prepare(
        run_dir=run_dir,
        resume=resume,
        restart_from_stage=restart,
        config_hashes=_config_hashes(),
    )
    if not has_state:
        assert prepared.requested_action == "fresh"
    elif restart is not None:
        assert prepared.requested_action == "restart"
        assert prepared.start_stage_name == restart
    elif resume:
        assert prepared.requested_action == "resume"
    else:
        # has_state but no resume/restart → treated as fresh
        assert prepared.requested_action == "fresh"


@pytest.mark.parametrize("restart_idx", [1, 2, 3])
def test_restart_idx_maps_to_correct_stage(tmp_path: Path, restart_idx: int) -> None:
    run_dir = tmp_path / f"restart_idx_{restart_idx}"
    store = PipelineStateStore(run_dir)
    store.save(
        store.init_state(
            logical_run_id=run_dir.name,
            config_path=str(tmp_path / "cfg.yaml"),
            training_critical_config_hash="t_hash",
            late_stage_config_hash="l_hash",
        )
    )
    preparator, _, _, _ = _build_preparator(tmp_path)

    prepared = preparator.prepare(
        run_dir=run_dir,
        resume=False,
        restart_from_stage=restart_idx,
        config_hashes=_config_hashes(),
    )
    expected_stage = [STAGE_A, STAGE_B, STAGE_C][restart_idx - 1]
    assert prepared.start_stage_name == expected_stage
    assert prepared.start_idx == restart_idx - 1

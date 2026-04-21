"""Comprehensive tests for MLflowAttemptManager.

Focus: partial-cleanup invariant, _require_manager semantics, teardown hook
ordering, encapsulation via public MLflowManager API.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.mlflow_attempt.manager import (
    MLflowAttemptManager,
    MLflowManagerNotInitializedError,
)
from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _build_config() -> MagicMock:
    cfg = MagicMock()
    cfg.experiment_tracking.mlflow.tracking_uri = "http://localhost:5002"
    cfg.experiment_tracking.mlflow.local_tracking_uri = None
    cfg.experiment_tracking.mlflow.ca_bundle_path = None
    cfg.experiment_tracking.mlflow.system_metrics_callback_enabled = False
    return cfg


def _build_state() -> PipelineState:
    return PipelineState(
        schema_version=1,
        logical_run_id="logical-1",
        run_directory="/tmp/run",
        config_path="/tmp/cfg.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_RUNNING,
        training_critical_config_hash="tc",
        late_stage_config_hash="ls",
    )


def _build_attempt() -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id="a1",
        attempt_no=1,
        runtime_name="runtime",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_RUNNING,
        started_at="2026-04-21T00:00:00+00:00",
    )


def _build_active_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.is_active = True
    mgr.run_id = "root-run-id"
    mgr.get_runtime_tracking_uri.return_value = "http://tracking"

    root_run = MagicMock()
    root_run.__enter__ = MagicMock(return_value=root_run)
    root_run.__exit__ = MagicMock(return_value=None)
    mgr.start_run.return_value = root_run

    attempt_run = MagicMock()
    attempt_run.__enter__ = MagicMock(return_value=attempt_run)
    attempt_run.__exit__ = MagicMock(return_value=None)
    mgr.start_nested_run.return_value = attempt_run
    mgr.adopt_existing_run.return_value = MagicMock(name="adopted-run")
    return mgr


@pytest.fixture
def manager(tmp_path: Path) -> MLflowAttemptManager:
    return MLflowAttemptManager(_build_config(), tmp_path / "cfg.yaml")


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_bootstrap_returns_active_manager(self, manager: MLflowAttemptManager) -> None:
        mock_mgr = MagicMock()
        mock_mgr.is_active = True
        with patch("src.pipeline.mlflow_attempt.manager.MLflowManager", return_value=mock_mgr):
            out = manager.bootstrap()
        assert out is mock_mgr
        assert manager.manager is mock_mgr
        assert manager.is_active

    def test_setup_for_attempt_with_injected_manager(self, manager: MLflowAttemptManager) -> None:
        mgr = _build_active_mgr()
        manager.setup_for_attempt(
            state=_build_state(),
            attempt=_build_attempt(),
            start_stage_idx=0,
            context={},
            total_stages=6,
            run_directory=Path("/tmp/run"),
            manager=mgr,
        )
        mgr.log_pipeline_config.assert_called_once()

    def test_teardown_success(self, manager: MLflowAttemptManager) -> None:
        mgr = MagicMock()
        manager._manager = mgr
        manager.teardown_attempt(pipeline_success=True, attempt_run_id="rid")
        mgr.end_run.assert_called_once_with(status="FINISHED")
        mgr.cleanup.assert_called_once()


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_bootstrap_exception_returns_none(self, manager: MLflowAttemptManager) -> None:
        with patch(
            "src.pipeline.mlflow_attempt.manager.MLflowManager",
            side_effect=RuntimeError("broken"),
        ):
            assert manager.bootstrap() is None
        assert manager.manager is None

    def test_preflight_setup_failed_returns_apperror(self, manager: MLflowAttemptManager) -> None:
        err = manager.ensure_preflight()
        assert err is not None
        assert err.code == "MLFLOW_PREFLIGHT_SETUP_FAILED"

    def test_preflight_unreachable_returns_apperror(self, manager: MLflowAttemptManager) -> None:
        mgr = MagicMock()
        mgr.is_active = True
        mgr.check_mlflow_connectivity.return_value = False
        mgr.get_last_connectivity_error.return_value = None
        manager._manager = mgr
        err = manager.ensure_preflight()
        assert err is not None
        assert err.code == "MLFLOW_PREFLIGHT_UNREACHABLE"


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_get_run_id_prefers_public_over_legacy(self, manager: MLflowAttemptManager) -> None:
        from types import SimpleNamespace

        manager._manager = SimpleNamespace(run_id="new-id", _run_id="old-id")  # type: ignore[assignment]
        assert manager.get_run_id() == "new-id"

    def test_get_run_id_empty_strings_both(self, manager: MLflowAttemptManager) -> None:
        from types import SimpleNamespace

        manager._manager = SimpleNamespace(run_id="", _run_id="")  # type: ignore[assignment]
        assert manager.get_run_id() is None

    def test_setup_for_attempt_inactive_manager_is_noop(self, manager: MLflowAttemptManager) -> None:
        mgr = MagicMock()
        mgr.is_active = False
        manager.setup_for_attempt(
            state=_build_state(),
            attempt=_build_attempt(),
            start_stage_idx=0,
            context={},
            total_stages=6,
            run_directory=Path("/tmp"),
            manager=mgr,
        )
        mgr.start_run.assert_not_called()
        mgr.start_nested_run.assert_not_called()


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_invariant_no_orphan_runs_after_setup_failure(
        self, manager: MLflowAttemptManager
    ) -> None:
        """INVARIANT: if any step in setup_for_attempt raises, the partially-opened
        runs are closed before re-raising — no orphan MLflow runs remain."""
        mgr = _build_active_mgr()
        mgr.log_pipeline_config.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            manager.setup_for_attempt(
                state=_build_state(),
                attempt=_build_attempt(),
                start_stage_idx=0,
                context={},
                total_stages=6,
                run_directory=Path("/tmp"),
                manager=mgr,
            )

        # Both runs should have been __exit__'d by _cleanup_partial_runs
        assert manager._attempt_run is None
        assert manager._run_context is None
        mgr.start_run.return_value.__exit__.assert_called()
        mgr.start_nested_run.return_value.__exit__.assert_called()

    def test_invariant_teardown_attempt_closes_attempt_then_root(
        self, manager: MLflowAttemptManager
    ) -> None:
        """INVARIANT: teardown closes attempt run BEFORE end_run on root."""
        mgr = MagicMock()
        attempt_run = MagicMock()
        attempt_run.__exit__ = MagicMock(return_value=None)
        run_context = MagicMock()
        run_context.__exit__ = MagicMock(return_value=None)
        manager._manager = mgr
        manager._attempt_run = attempt_run
        manager._run_context = run_context

        call_order: list[str] = []
        attempt_run.__exit__.side_effect = lambda *_: call_order.append("attempt_run.exit")
        mgr.end_run.side_effect = lambda status: call_order.append(f"end_run.{status}")
        run_context.__exit__.side_effect = lambda *_: call_order.append("run_context.exit")

        manager.teardown_attempt(pipeline_success=True, attempt_run_id="rid")
        assert call_order[0] == "attempt_run.exit"
        assert call_order[1] == "end_run.FINISHED"
        assert call_order[2] == "run_context.exit"

    def test_invariant_teardown_hook_failure_does_not_skip_cleanup(
        self, manager: MLflowAttemptManager
    ) -> None:
        """INVARIANT: a hook failure is logged, not propagated; cleanup still runs."""
        mgr = MagicMock()
        manager._manager = mgr

        def bad_hook() -> None:
            raise RuntimeError("hook boom")

        def bad_after(_: str | None) -> None:
            raise RuntimeError("after boom")

        manager.teardown_attempt(
            pipeline_success=True,
            attempt_run_id="rid",
            on_before_end=bad_hook,
            on_after_end=bad_after,
        )
        mgr.end_run.assert_called_once()
        mgr.cleanup.assert_called_once()


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_require_manager_raises_explicit_error_not_assert(
        self, manager: MLflowAttemptManager
    ) -> None:
        """Must raise MLflowManagerNotInitializedError (not assert, so -O won't silence it)."""
        with pytest.raises(MLflowManagerNotInitializedError):
            manager._require_manager()

    def test_open_existing_root_run_without_manager_raises(
        self, manager: MLflowAttemptManager
    ) -> None:
        with pytest.raises(MLflowManagerNotInitializedError):
            manager.open_existing_root_run("root-id")

    def test_adopt_existing_run_returns_none_propagates(self, manager: MLflowAttemptManager) -> None:
        mgr = MagicMock()
        mgr.adopt_existing_run.return_value = None
        manager._manager = mgr
        assert manager.open_existing_root_run("r") is None

    def test_bootstrap_mlflow_module_missing(self, manager: MLflowAttemptManager) -> None:
        """If `import mlflow` raises inside bootstrap.disable_system_metrics_logging,
        bootstrap still completes (best-effort try/except)."""
        mgr = MagicMock()
        mgr.is_active = True
        with (
            patch("src.pipeline.mlflow_attempt.manager.MLflowManager", return_value=mgr),
            patch.dict("sys.modules", {"mlflow": None}),
        ):
            out = manager.bootstrap()
        # Bootstrap should succeed even though the system-metrics disabler errored
        assert out is mgr


# =============================================================================
# 6. REGRESSIONS
# =============================================================================


class TestRegressions:
    def test_regression_open_existing_root_uses_public_adopt(
        self, manager: MLflowAttemptManager
    ) -> None:
        """REGRESSION: previously mutated mgr._mlflow / mgr._run / mgr._run_id directly.
        Now goes through MLflowManager.adopt_existing_run()."""
        mgr = MagicMock()
        mgr.adopt_existing_run.return_value = "adopted"
        manager._manager = mgr
        result = manager.open_existing_root_run("root-xyz")
        mgr.adopt_existing_run.assert_called_once_with("root-xyz")
        assert result == "adopted"

    def test_regression_no_direct_private_access(self, manager: MLflowAttemptManager) -> None:
        """REGRESSION: _mlflow / _gateway / _run_id should NOT be read in production paths.

        This test asserts open_existing_root_run doesn't touch those attrs (even
        if they're absent on the mock).
        """
        mgr = MagicMock(spec=["adopt_existing_run"])  # only adopt_existing_run allowed
        mgr.adopt_existing_run.return_value = None
        manager._manager = mgr
        # Should not raise AttributeError — we only use the speced interface
        assert manager.open_existing_root_run("x") is None

    def test_regression_require_manager_not_silent_under_optimize(self) -> None:
        """REGRESSION: asserts get stripped by python -O. _require_manager uses a
        real raise so production is safe."""
        manager = MLflowAttemptManager(_build_config(), Path("/tmp/cfg.yaml"))
        # This behaves identically with or without -O.
        with pytest.raises(MLflowManagerNotInitializedError):
            manager._require_manager()


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


@pytest.mark.parametrize("pipeline_success", [True, False])
@pytest.mark.parametrize("has_attempt_run", [True, False])
@pytest.mark.parametrize("has_run_context", [True, False])
def test_combinatorial_teardown_variants(
    manager: MLflowAttemptManager,
    pipeline_success: bool,
    has_attempt_run: bool,
    has_run_context: bool,
) -> None:
    mgr = MagicMock()
    manager._manager = mgr
    if has_attempt_run:
        manager._attempt_run = MagicMock()
        manager._attempt_run.__exit__ = MagicMock(return_value=None)
    if has_run_context:
        manager._run_context = MagicMock()
        manager._run_context.__exit__ = MagicMock(return_value=None)

    manager.teardown_attempt(
        pipeline_success=pipeline_success,
        attempt_run_id="rid" if pipeline_success else None,
    )

    expected_status = "FINISHED" if pipeline_success else "FAILED"
    mgr.end_run.assert_called_once_with(status=expected_status)
    mgr.cleanup.assert_called_once()
    # After teardown, both run attrs must be reset to None
    assert manager._attempt_run is None
    assert manager._run_context is None


@pytest.mark.parametrize(
    ("manager_state", "expected_run_id"),
    [
        (None, None),
        (MagicMock(run_id="R1", _run_id=""), "R1"),
        (MagicMock(run_id="", _run_id="legacy"), "legacy"),
        (MagicMock(run_id=None, _run_id="legacy"), "legacy"),
        (MagicMock(run_id="", _run_id=""), None),
    ],
)
def test_combinatorial_get_run_id(
    manager: MLflowAttemptManager, manager_state, expected_run_id: str | None
) -> None:
    manager._manager = manager_state
    assert manager.get_run_id() == expected_run_id

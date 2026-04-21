"""Unit tests for MLflowAttemptManager.

Focus areas:
- bootstrap: happy path + exception swallowed
- get_run_id: prefers public property, falls back to legacy attr, handles None
- ensure_preflight: returns AppError for missing manager / not active / unreachable, None when healthy
- setup_for_attempt: opens root + attempt runs, partial-cleanup on failure (double-close mitigation)
- open_existing_root_run: wires existing run into manager internals
- teardown_attempt: hook order + state-path artifact upload + run closing
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.mlflow_attempt.manager import MLflowAttemptManager
from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _build_config() -> MagicMock:
    config = MagicMock()
    config.experiment_tracking.mlflow.tracking_uri = "http://localhost:5002"
    config.experiment_tracking.mlflow.local_tracking_uri = None
    config.experiment_tracking.mlflow.ca_bundle_path = None
    config.experiment_tracking.mlflow.system_metrics_callback_enabled = False
    return config


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


@pytest.fixture
def manager_under_test(tmp_path: Path) -> MLflowAttemptManager:
    return MLflowAttemptManager(_build_config(), tmp_path / "cfg.yaml")


# -----------------------------------------------------------------------------
# bootstrap
# -----------------------------------------------------------------------------


def test_bootstrap_success_sets_manager(manager_under_test: MLflowAttemptManager) -> None:
    mock_mgr = MagicMock()
    mock_mgr.is_active = True
    with patch("src.pipeline.mlflow_attempt.manager.MLflowManager", return_value=mock_mgr):
        out = manager_under_test.bootstrap()
    assert out is mock_mgr
    assert manager_under_test.manager is mock_mgr
    mock_mgr.setup.assert_called_once_with(disable_system_metrics=True)


def test_bootstrap_exception_returns_none(manager_under_test: MLflowAttemptManager) -> None:
    with patch(
        "src.pipeline.mlflow_attempt.manager.MLflowManager",
        side_effect=RuntimeError("boom"),
    ):
        assert manager_under_test.bootstrap() is None
    assert manager_under_test.manager is None


# -----------------------------------------------------------------------------
# get_run_id
# -----------------------------------------------------------------------------


def test_get_run_id_none_when_no_manager(manager_under_test: MLflowAttemptManager) -> None:
    assert manager_under_test.get_run_id() is None


def test_get_run_id_prefers_public_property(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr.run_id = "run-123"
    mgr._run_id = "legacy"
    manager_under_test._manager = mgr
    assert manager_under_test.get_run_id() == "run-123"


def test_get_run_id_falls_back_to_legacy_attr(manager_under_test: MLflowAttemptManager) -> None:
    mgr = SimpleNamespace(run_id=None, _run_id="legacy-id")
    manager_under_test._manager = mgr  # type: ignore[assignment]
    assert manager_under_test.get_run_id() == "legacy-id"


def test_get_run_id_none_when_both_empty(manager_under_test: MLflowAttemptManager) -> None:
    mgr = SimpleNamespace(run_id="", _run_id="")
    manager_under_test._manager = mgr  # type: ignore[assignment]
    assert manager_under_test.get_run_id() is None


# -----------------------------------------------------------------------------
# ensure_preflight
# -----------------------------------------------------------------------------


def test_preflight_returns_setup_failed_when_no_manager(manager_under_test: MLflowAttemptManager) -> None:
    err = manager_under_test.ensure_preflight()
    assert err is not None
    assert err.code == "MLFLOW_PREFLIGHT_SETUP_FAILED"


def test_preflight_returns_unreachable_when_connectivity_fails(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr.is_active = True
    mgr.get_runtime_tracking_uri.return_value = "http://fake"
    mgr.check_mlflow_connectivity.return_value = False
    mgr.get_last_connectivity_error.return_value = None
    manager_under_test._manager = mgr
    err = manager_under_test.ensure_preflight()
    assert err is not None
    assert err.code == "MLFLOW_PREFLIGHT_UNREACHABLE"


def test_preflight_surfaces_gateway_error_code(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr.is_active = True
    mgr.get_runtime_tracking_uri.return_value = "http://fake"
    mgr.check_mlflow_connectivity.return_value = False
    gateway_err = MagicMock()
    gateway_err.code = "GW_SPECIFIC"
    gateway_err.message = "specific reason"
    gateway_err.to_log_dict.return_value = {"code": "GW_SPECIFIC"}
    mgr.get_last_connectivity_error.return_value = gateway_err
    manager_under_test._manager = mgr
    err = manager_under_test.ensure_preflight()
    assert err is not None
    assert err.code == "GW_SPECIFIC"
    assert "specific reason" in err.message


def test_preflight_returns_none_when_healthy(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr.is_active = True
    mgr.get_runtime_tracking_uri.return_value = "http://ok"
    mgr.check_mlflow_connectivity.return_value = True
    manager_under_test._manager = mgr
    assert manager_under_test.ensure_preflight() is None


# -----------------------------------------------------------------------------
# log_config_artifact
# -----------------------------------------------------------------------------


def test_log_config_artifact_uploads_when_path_exists(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("x: 1")
    m = MLflowAttemptManager(_build_config(), cfg_path)
    mgr = MagicMock()
    m._manager = mgr
    m.log_config_artifact()
    mgr.log_artifact.assert_called_once_with(str(cfg_path))


def test_log_config_artifact_skips_when_no_manager(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("x: 1")
    m = MLflowAttemptManager(_build_config(), cfg_path)
    m.log_config_artifact()  # No manager — no crash


# -----------------------------------------------------------------------------
# setup_for_attempt
# -----------------------------------------------------------------------------


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
    return mgr


def test_setup_for_attempt_opens_root_and_attempt_runs(manager_under_test: MLflowAttemptManager) -> None:
    state = _build_state()
    attempt = _build_attempt()
    mgr = _build_active_mgr()

    with patch("src.pipeline.mlflow_attempt.manager.MLflowManager", return_value=mgr):
        context: dict = {}
        manager_under_test.setup_for_attempt(
            state=state,
            attempt=attempt,
            start_stage_idx=0,
            context=context,
            total_stages=6,
            run_directory=Path("/tmp/run"),
        )

    assert state.root_mlflow_run_id == "root-run-id"
    assert attempt.root_mlflow_run_id == "root-run-id"
    mgr.log_pipeline_config.assert_called_once()
    mgr.log_dataset_config.assert_called_once()
    mgr.log_params.assert_called_once()


def test_setup_for_attempt_uses_injected_manager(manager_under_test: MLflowAttemptManager) -> None:
    state = _build_state()
    attempt = _build_attempt()
    mgr = _build_active_mgr()

    # Patch must NOT be used — injected manager should short-circuit bootstrap.
    with patch(
        "src.pipeline.mlflow_attempt.manager.MLflowManager",
        side_effect=AssertionError("bootstrap should not be called"),
    ):
        manager_under_test.setup_for_attempt(
            state=state,
            attempt=attempt,
            start_stage_idx=0,
            context={},
            total_stages=6,
            run_directory=Path("/tmp/run"),
            manager=mgr,
        )
    assert manager_under_test.manager is mgr


def test_setup_for_attempt_partial_cleanup_on_failure(manager_under_test: MLflowAttemptManager) -> None:
    """If log_pipeline_config fails, both runs must be closed (double-close mitigation)."""
    state = _build_state()
    attempt = _build_attempt()
    mgr = _build_active_mgr()
    mgr.log_pipeline_config.side_effect = RuntimeError("log failed")

    with pytest.raises(RuntimeError, match="log failed"):
        manager_under_test.setup_for_attempt(
            state=state,
            attempt=attempt,
            start_stage_idx=0,
            context={},
            total_stages=6,
            run_directory=Path("/tmp/run"),
            manager=mgr,
        )

    # Both runs should be closed by partial cleanup
    mgr.start_nested_run.return_value.__exit__.assert_called()
    mgr.start_run.return_value.__exit__.assert_called()
    assert manager_under_test._attempt_run is None
    assert manager_under_test._run_context is None


def test_setup_for_attempt_reopens_existing_root(manager_under_test: MLflowAttemptManager) -> None:
    state = _build_state()
    state.root_mlflow_run_id = "existing-root"
    attempt = _build_attempt()
    mgr = _build_active_mgr()
    mgr._mlflow = MagicMock()
    mgr._mlflow.start_run.return_value = MagicMock()

    manager_under_test.setup_for_attempt(
        state=state,
        attempt=attempt,
        start_stage_idx=0,
        context={},
        total_stages=6,
        run_directory=Path("/tmp/run"),
        manager=mgr,
    )

    # When root already exists, start_run (context-manager) must NOT be called — we reopen.
    mgr.start_run.assert_not_called()
    mgr._mlflow.start_run.assert_called_once()
    assert attempt.root_mlflow_run_id == "existing-root"


def test_setup_for_attempt_noop_when_manager_inactive(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr.is_active = False
    state = _build_state()
    attempt = _build_attempt()
    manager_under_test.setup_for_attempt(
        state=state,
        attempt=attempt,
        start_stage_idx=0,
        context={},
        total_stages=6,
        run_directory=Path("/tmp/run"),
        manager=mgr,
    )
    mgr.start_run.assert_not_called()
    mgr.start_nested_run.assert_not_called()


# -----------------------------------------------------------------------------
# open_existing_root_run
# -----------------------------------------------------------------------------


def test_open_existing_root_run_wires_internals(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr._mlflow = MagicMock()
    mgr._mlflow.start_run.return_value = MagicMock(name="opened-run")
    manager_under_test._manager = mgr

    run = manager_under_test.open_existing_root_run("root-abc")
    assert run is mgr._mlflow.start_run.return_value
    assert mgr._run_id == "root-abc"
    assert mgr._parent_run_id == "root-abc"


def test_open_existing_root_run_handles_missing_mlflow(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    mgr._mlflow = None
    manager_under_test._manager = mgr
    assert manager_under_test.open_existing_root_run("x") is None


# -----------------------------------------------------------------------------
# teardown_attempt
# -----------------------------------------------------------------------------


def test_teardown_invokes_hooks_in_order(manager_under_test: MLflowAttemptManager, tmp_path: Path) -> None:
    mgr = MagicMock()
    attempt_run = MagicMock()
    attempt_run.__exit__ = MagicMock(return_value=None)
    run_context = MagicMock()
    run_context.__exit__ = MagicMock(return_value=None)
    manager_under_test._manager = mgr
    manager_under_test._attempt_run = attempt_run
    manager_under_test._run_context = run_context

    state_path = tmp_path / "state.json"
    state_path.write_text("{}")

    call_order: list[str] = []

    def before_end() -> None:
        call_order.append("before_end")

    def sync_state() -> Path | None:
        call_order.append("sync_state")
        return state_path

    def after_end(run_id: str | None) -> None:
        call_order.append(f"after_end:{run_id}")

    manager_under_test.teardown_attempt(
        pipeline_success=True,
        attempt_run_id="rid-1",
        on_before_end=before_end,
        state_path_supplier=sync_state,
        on_after_end=after_end,
    )

    assert call_order == ["before_end", "sync_state", "after_end:rid-1"]
    mgr.log_artifact.assert_called_once_with(str(state_path))
    mgr.end_run.assert_called_once_with(status="FINISHED")
    mgr.cleanup.assert_called_once()


def test_teardown_passes_failure_status_on_error(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    attempt_run = MagicMock()
    attempt_run.__exit__ = MagicMock(return_value=None)
    manager_under_test._manager = mgr
    manager_under_test._attempt_run = attempt_run

    manager_under_test.teardown_attempt(
        pipeline_success=False,
        attempt_run_id=None,
    )
    mgr.end_run.assert_called_once_with(status="FAILED")
    # attempt_run should be exited with an exception tuple on failure
    args, _ = attempt_run.__exit__.call_args
    assert args[0] is RuntimeError


def test_teardown_hook_failures_are_logged_not_raised(manager_under_test: MLflowAttemptManager) -> None:
    mgr = MagicMock()
    manager_under_test._manager = mgr

    def before_end() -> None:
        raise RuntimeError("aggregate crashed")

    def after_end(_: str | None) -> None:
        raise RuntimeError("report crashed")

    # Must not raise
    manager_under_test.teardown_attempt(
        pipeline_success=True,
        attempt_run_id=None,
        on_before_end=before_end,
        on_after_end=after_end,
    )
    mgr.end_run.assert_called_once_with(status="FINISHED")


def test_teardown_skips_state_artifact_when_supplier_returns_none(
    manager_under_test: MLflowAttemptManager,
) -> None:
    mgr = MagicMock()
    manager_under_test._manager = mgr
    manager_under_test.teardown_attempt(
        pipeline_success=True,
        attempt_run_id=None,
        state_path_supplier=lambda: None,
    )
    mgr.log_artifact.assert_not_called()
    mgr.end_run.assert_called_once_with(status="FINISHED")

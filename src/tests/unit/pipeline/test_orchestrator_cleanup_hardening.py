"""Comprehensive tests for the hardened ``_run_stateful`` finally block
and the newly-public MLflow API (adopt_existing_run / tracking_uri).

Covers all 7 categories with a focus on two review-fixes:

1. Finally block isolates cleanup-step failures so ``run_lock.release()``
   is always reached.
2. ``MLflowManager.adopt_existing_run()`` + ``.tracking_uri`` replace
   private-attribute mutation that was previously done from
   ``MLflowAttemptManager``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.result import Ok


# -----------------------------------------------------------------------------
# Orchestrator-building helpers (match style of existing restart_policy tests)
# -----------------------------------------------------------------------------


def _build_mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "gpt2"
    cfg.model.model_dump.return_value = {"name": "gpt2"}
    cfg.training.type = "sft"
    cfg.training.strategies = []
    cfg.training.model_dump.return_value = {"type": "sft"}
    cfg.training.get_strategy_chain.return_value = []
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.training.hyperparams.per_device_train_batch_size = 4
    cfg.datasets = {}
    cfg.get_active_provider_name.return_value = "single_node"
    cfg.get_provider_config.return_value = {}
    cfg.experiment_tracking.mlflow = None
    cfg.inference.enabled = False
    cfg.inference.model_dump.return_value = {"enabled": False}
    cfg.evaluation.enabled = False
    cfg.evaluation.model_dump.return_value = {"enabled": False}
    return cfg


def _build_orchestrator(config_path: Path) -> PipelineOrchestrator:
    cfg = _build_mock_config()
    secrets = MagicMock()
    secrets.hf_token = "test-token"
    with (
        patch("src.pipeline.orchestrator.load_config", return_value=cfg),
        patch("src.pipeline.orchestrator.load_secrets", return_value=secrets),
        patch("src.pipeline.orchestrator.validate_strategy_chain", return_value=Ok(None)),
        patch("src.pipeline.orchestrator.DatasetValidator"),
        patch("src.pipeline.orchestrator.GPUDeployer"),
        patch("src.pipeline.orchestrator.TrainingMonitor"),
        patch("src.pipeline.orchestrator.ModelRetriever"),
        patch("src.pipeline.orchestrator.InferenceDeployer"),
        patch("src.pipeline.orchestrator.ModelEvaluator"),
    ):
        return PipelineOrchestrator(config_path)


# =============================================================================
# 1. POSITIVE — finally block executes all steps in order on success path
# =============================================================================


class TestFinallyPositive:
    def test_finally_runs_all_three_cleanup_steps(self, tmp_path: Path) -> None:
        """Positive: all three cleanup steps execute exactly once on normal success."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock()
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        orch._run_lock = lock

        # Trigger finally via a synthetic exception path
        with patch.object(
            orch, "_prepare_stateful_attempt", side_effect=RuntimeError("just to trigger finally")
        ):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        orch._flush_pending_collectors.assert_called_once()
        orch._cleanup_resources.assert_called_once()
        orch._teardown_mlflow_attempt.assert_called_once()
        lock.release.assert_called_once()
        assert orch._run_lock is None


# =============================================================================
# 2. NEGATIVE — cleanup step raising does not break the rest
# =============================================================================


class TestFinallyNegative:
    def test_flush_exception_still_allows_cleanup_resources(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock(side_effect=RuntimeError("flush broke"))
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep failed")):
            result = orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        # flush raised — but cleanup_resources still ran
        orch._cleanup_resources.assert_called_once()
        orch._teardown_mlflow_attempt.assert_called_once()
        lock.release.assert_called_once()
        assert orch._run_lock is None  # finally sets it to None after release
        assert result.is_failure()

    def test_cleanup_exception_still_allows_teardown(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock()
        orch._cleanup_resources = MagicMock(side_effect=RuntimeError("cleanup broke"))
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep failed")):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        orch._teardown_mlflow_attempt.assert_called_once()
        lock.release.assert_called_once()

    def test_teardown_exception_still_releases_run_lock(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock()
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock(side_effect=RuntimeError("teardown broke"))
        lock = MagicMock()
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep failed")):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        lock.release.assert_called_once()
        assert orch._run_lock is None  # set to None after release


# =============================================================================
# 3. BOUNDARY — no run lock, no-op
# =============================================================================


class TestFinallyBoundary:
    def test_no_run_lock_no_crash(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock()
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        orch._run_lock = None  # never acquired (prepare failed early)

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("early")):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)
        # no crash

    def test_run_lock_release_exception_swallowed(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock()
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        lock.release.side_effect = OSError("fs error on release")
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("x")):
            # Must not propagate the OSError from the finally block
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)
        assert orch._run_lock is None


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestFinallyInvariants:
    @pytest.mark.parametrize(
        "break_step",
        ["flush_pending_collectors", "cleanup_resources", "teardown_mlflow_attempt"],
    )
    def test_invariant_run_lock_always_released(self, tmp_path: Path, break_step: str) -> None:
        """INVARIANT: no matter which step fails, run_lock.release() is called."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock()
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        orch._run_lock = lock

        getattr(orch, f"_{break_step}").side_effect = RuntimeError(f"{break_step} broke")

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep")):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        lock.release.assert_called_once()

    def test_invariant_all_steps_attempted_even_with_multiple_failures(
        self, tmp_path: Path
    ) -> None:
        """INVARIANT: every step is attempted; no short-circuit after one failure."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock(side_effect=RuntimeError("a"))
        orch._cleanup_resources = MagicMock(side_effect=RuntimeError("b"))
        orch._teardown_mlflow_attempt = MagicMock(side_effect=RuntimeError("c"))
        lock = MagicMock()
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("d")):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        orch._flush_pending_collectors.assert_called_once()
        orch._cleanup_resources.assert_called_once()
        orch._teardown_mlflow_attempt.assert_called_once()
        lock.release.assert_called_once()


# =============================================================================
# 5. DEPENDENCY ERRORS — different exception types do not break semantics
# =============================================================================


class TestFinallyDependencyErrors:
    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("boom"),
            OSError("fs failure"),
            ValueError("bad arg"),
            ConnectionError("network gone"),
        ],
    )
    def test_any_exception_type_swallowed_in_finally(self, tmp_path: Path, exc: Exception) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock(side_effect=exc)
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep")):
            # Must not propagate ``exc``; result is an Err from the prep RuntimeError handler
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        orch._cleanup_resources.assert_called_once()
        lock.release.assert_called_once()


# =============================================================================
# 6. REGRESSIONS — the specific review-reported bug
# =============================================================================


class TestFinallyRegressions:
    def test_regression_unprotected_finally_would_leak_run_lock(self, tmp_path: Path) -> None:
        """REGRESSION: pre-fix, if _flush_pending_collectors raised, the finally
        block aborted before run_lock.release() — subsequent launches got stuck.

        This test asserts the new behaviour: lock is always released.
        """
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model:\n  name: gpt2\n")
        orch = _build_orchestrator(cfg_path)

        orch._flush_pending_collectors = MagicMock(
            side_effect=FileNotFoundError("disk error during flush")
        )
        orch._cleanup_resources = MagicMock()
        orch._teardown_mlflow_attempt = MagicMock()
        lock = MagicMock()
        orch._run_lock = lock

        with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep")):
            orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

        lock.release.assert_called_once()


# =============================================================================
# 7. COMBINATORIAL — cross product of (step_fail, lock_present)
# =============================================================================


_FINALLY_STEPS = ["flush_pending_collectors", "cleanup_resources", "teardown_mlflow_attempt"]


@pytest.mark.parametrize("failing_step", [*_FINALLY_STEPS, "none"])
@pytest.mark.parametrize("has_lock", [True, False])
def test_combinatorial_finally_matrix(
    tmp_path: Path, failing_step: str, has_lock: bool
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model:\n  name: gpt2\n")
    orch = _build_orchestrator(cfg_path)

    for step in _FINALLY_STEPS:
        mock = MagicMock()
        if step == failing_step:
            mock.side_effect = RuntimeError(f"{step} broke")
        setattr(orch, f"_{step}", mock)

    lock = MagicMock() if has_lock else None
    orch._run_lock = lock

    with patch.object(orch, "_prepare_stateful_attempt", side_effect=RuntimeError("prep")):
        orch._run_stateful(run_dir=None, resume=False, restart_from_stage=None)

    # Invariant: every step attempted, exactly once.
    for step in _FINALLY_STEPS:
        assert getattr(orch, f"_{step}").call_count == 1

    if has_lock:
        lock.release.assert_called_once()


# =============================================================================
# MLflowManager.adopt_existing_run + tracking_uri (new public API)
# =============================================================================


class TestMLflowManagerPublicAPI:
    """Comprehensive set for the two new public members on MLflowManager."""

    def test_positive_tracking_uri_reads_gateway(self) -> None:
        from src.training.managers.mlflow_manager.manager import MLflowManager

        mgr = MLflowManager.__new__(MLflowManager)
        mgr._mlflow = object()  # truthy is_active
        mgr._gateway = MagicMock()
        mgr._gateway.uri = "http://tracking"
        assert mgr.tracking_uri == "http://tracking"

    def test_negative_tracking_uri_none_when_not_active(self) -> None:
        from src.training.managers.mlflow_manager.manager import MLflowManager

        mgr = MLflowManager.__new__(MLflowManager)
        mgr._mlflow = None  # setup() not called
        # _gateway may or may not exist; shouldn't matter
        mgr._gateway = MagicMock()
        assert mgr.tracking_uri is None

    def test_positive_adopt_existing_run_wires_fields(self) -> None:
        from src.training.managers.mlflow_manager.manager import MLflowManager

        mgr = MLflowManager.__new__(MLflowManager)
        mlflow_module = MagicMock()
        opened = MagicMock(name="opened-run")
        mlflow_module.start_run.return_value = opened
        mgr._mlflow = mlflow_module

        result = mgr.adopt_existing_run("root-xyz")
        assert result is opened
        assert mgr._run is opened
        assert mgr._run_id == "root-xyz"
        assert mgr._parent_run_id == "root-xyz"
        mlflow_module.start_run.assert_called_once_with(
            run_id="root-xyz", nested=False, log_system_metrics=False
        )

    def test_negative_adopt_returns_none_when_inactive(self) -> None:
        from src.training.managers.mlflow_manager.manager import MLflowManager

        mgr = MLflowManager.__new__(MLflowManager)
        mgr._mlflow = None
        assert mgr.adopt_existing_run("x") is None

    def test_invariant_adopt_does_not_touch_gateway(self) -> None:
        """INVARIANT: adopt_existing_run only touches run lifecycle attrs, not gateway."""
        from src.training.managers.mlflow_manager.manager import MLflowManager

        mgr = MLflowManager.__new__(MLflowManager)
        mgr._mlflow = MagicMock()
        mgr._mlflow.start_run.return_value = MagicMock()
        original_gateway = object()
        mgr._gateway = original_gateway  # type: ignore[assignment]
        mgr.adopt_existing_run("r")
        assert mgr._gateway is original_gateway

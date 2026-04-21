"""Tests for :class:`PipelineBootstrap`.

Bootstrap is a factory — it runs once per orchestrator construction. Its
contract: given (config_path, run_ctx, settings, attempt_controller, hooks),
it produces a frozen :class:`BootstrapResult` containing every collaborator
the orchestrator needs.

Coverage: positive / negative / boundary / invariants / dep-errors /
regressions / combinatorial.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.bootstrap import BootstrapResult, PipelineBootstrap
from src.pipeline.state import AttemptController

if TYPE_CHECKING:
    from pathlib import Path


def _mk_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "gpt2"
    cfg.model.model_dump.return_value = {"name": "gpt2"}
    cfg.training.type = "sft"
    cfg.training.strategies = []
    cfg.training.get_strategy_chain.return_value = []
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.training.hyperparams.per_device_train_batch_size = 4
    cfg.training.model_dump.return_value = {"type": "sft"}
    dataset_cfg = MagicMock()
    dataset_cfg.model_dump.return_value = {"path": "data/train.jsonl"}
    cfg.datasets = {"default": dataset_cfg}
    cfg.get_active_provider_name.return_value = "single_node"
    cfg.get_provider_config.return_value = {"cleanup": {"on_interrupt": True}}
    cfg.experiment_tracking.mlflow = SimpleNamespace(
        tracking_uri="http://localhost:5002",
        system_metrics_callback_enabled=False,
    )
    cfg.inference.enabled = False
    cfg.inference.model_dump.return_value = {"enabled": False}
    cfg.inference.common.keep_inference_after_eval = False
    cfg.evaluation.enabled = False
    cfg.evaluation.model_dump.return_value = {"enabled": False}
    return cfg


def _mk_secrets(*, hf_token: str = "hf_test") -> MagicMock:
    s = MagicMock()
    s.hf_token = hf_token
    s.runpod_api_key = None
    return s


def _mk_run_ctx() -> SimpleNamespace:
    return SimpleNamespace(name="run_x", run_id="rid_x")


def _mk_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(runs_base_dir=tmp_path / "runs")


def _make_attempt_controller() -> AttemptController:
    return AttemptController(save_fn=MagicMock(), run_ctx=_mk_run_ctx())


def _build_bootstrap(
    tmp_path: Path,
    *,
    config: MagicMock | None = None,
    secrets: MagicMock | None = None,
) -> BootstrapResult:
    config = config or _mk_config()
    secrets = secrets or _mk_secrets()
    controller = _make_attempt_controller()
    with (
        patch(
            "src.pipeline.bootstrap.pipeline_bootstrap.load_config",
            return_value=config,
        ),
        patch(
            "src.pipeline.bootstrap.pipeline_bootstrap.load_secrets",
            return_value=secrets,
        ),
        # Stages need real-ish constructors; mock _build_stages to skip them.
        patch(
            "src.pipeline.execution.stage_registry.StageRegistry._build_stages",
            return_value=[MagicMock(stage_name=f"Stage {i}") for i in range(3)],
        ),
    ):
        return PipelineBootstrap.build(
            config_path=tmp_path / "cfg.yaml",
            run_ctx=_mk_run_ctx(),
            settings=_mk_settings(tmp_path),
            attempt_controller=controller,
            on_stage_completed=lambda _: None,
            on_shutdown_signal=lambda _: None,
        )


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_build_returns_frozen_result_with_all_fields(self, tmp_path: Path) -> None:
        result = _build_bootstrap(tmp_path)
        assert isinstance(result, BootstrapResult)
        # Verify every public field is populated.
        assert result.config is not None
        assert result.secrets is not None
        assert result.registry is not None
        assert result.launch_preparator is not None
        assert result.restart_inspector is not None
        assert result.stage_execution_loop is not None
        assert result.attempt_controller is not None

    def test_bootstrap_result_is_frozen(self, tmp_path: Path) -> None:
        result = _build_bootstrap(tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            result.config = "new"  # type: ignore[misc]

    def test_collectors_dict_is_shared_with_registry(self, tmp_path: Path) -> None:
        """Critical invariant: ValidationArtifactManager and StageRegistry
        must watch the SAME collectors dict."""
        result = _build_bootstrap(tmp_path)
        assert result.collectors is result.registry.collectors


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_invalid_strategy_chain_raises(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        s1 = MagicMock()
        s1.strategy_type = "sft"
        s2 = MagicMock()
        s2.strategy_type = "grpo"
        cfg.training.strategies = [s1, s2]
        with patch(
            "src.pipeline.bootstrap.startup_validator.validate_strategy_chain",
            return_value=MagicMock(is_failure=lambda: True, unwrap_err=lambda: "chain broken"),
        ), pytest.raises(Exception, match="Invalid strategy chain"):
            _build_bootstrap(tmp_path, config=cfg)

    def test_missing_secrets_file_propagates(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_config",
                return_value=_mk_config(),
            ),
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_secrets",
                side_effect=FileNotFoundError("no secrets.env"),
            ),pytest.raises(FileNotFoundError)
        ):
            PipelineBootstrap.build(
                config_path=tmp_path / "cfg.yaml",
                run_ctx=_mk_run_ctx(),
                settings=_mk_settings(tmp_path),
                attempt_controller=_make_attempt_controller(),
                on_stage_completed=lambda _: None,
                on_shutdown_signal=lambda _: None,
            )


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_empty_strategy_list_succeeds(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        cfg.training.strategies = []
        result = _build_bootstrap(tmp_path, config=cfg)
        assert isinstance(result, BootstrapResult)

    def test_stages_reflect_registry(self, tmp_path: Path) -> None:
        result = _build_bootstrap(tmp_path)
        assert result.stages is result.registry.stages


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_attempt_controller_is_passed_through(self, tmp_path: Path) -> None:
        """Bootstrap must NOT create its own AttemptController — the
        orchestrator's instance (bound to its per-run state) must be used."""
        controller = _make_attempt_controller()
        with (
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_config",
                return_value=_mk_config(),
            ),
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_secrets",
                return_value=_mk_secrets(),
            ),
            patch(
                "src.pipeline.execution.stage_registry.StageRegistry._build_stages",
                return_value=[],
            ),
        ):
            result = PipelineBootstrap.build(
                config_path=tmp_path / "cfg.yaml",
                run_ctx=_mk_run_ctx(),
                settings=_mk_settings(tmp_path),
                attempt_controller=controller,
                on_stage_completed=lambda _: None,
                on_shutdown_signal=lambda _: None,
            )
        assert result.attempt_controller is controller

    def test_hooks_wired_into_execution_loop(self, tmp_path: Path) -> None:
        """Invariant: the hooks passed to build() are wired into the
        StageExecutionLoop — we test this by firing them via the loop's
        private attributes."""
        completed_calls: list[str] = []
        shutdown_calls: list[str] = []
        with (
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_config",
                return_value=_mk_config(),
            ),
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_secrets",
                return_value=_mk_secrets(),
            ),
            patch(
                "src.pipeline.execution.stage_registry.StageRegistry._build_stages",
                return_value=[],
            ),
        ):
            result = PipelineBootstrap.build(
                config_path=tmp_path / "cfg.yaml",
                run_ctx=_mk_run_ctx(),
                settings=_mk_settings(tmp_path),
                attempt_controller=_make_attempt_controller(),
                on_stage_completed=completed_calls.append,
                on_shutdown_signal=shutdown_calls.append,
            )

        # Fire the loop's own hooks to verify wiring.
        result.stage_execution_loop._on_stage_completed("test_stage")  # type: ignore[attr-defined]
        result.stage_execution_loop._on_shutdown_signal("SIGINT")  # type: ignore[attr-defined]
        assert completed_calls == ["test_stage"]
        assert shutdown_calls == ["SIGINT"]


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_load_config_failure_propagates(self, tmp_path: Path) -> None:
        with (
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_config",
                side_effect=OSError("config read fail"),
            ),
            patch(
                "src.pipeline.bootstrap.pipeline_bootstrap.load_secrets",
                return_value=_mk_secrets(),
            ),pytest.raises(OSError, match="config read fail")
        ):
            PipelineBootstrap.build(
                config_path=tmp_path / "cfg.yaml",
                run_ctx=_mk_run_ctx(),
                settings=_mk_settings(tmp_path),
                attempt_controller=_make_attempt_controller(),
                on_stage_completed=lambda _: None,
                on_shutdown_signal=lambda _: None,
            )


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_collectors_instance_shared_across_components(self, tmp_path: Path) -> None:
        """Regression: ValidationArtifactManager, StageRegistry, and the
        bootstrap result all share the same dict — if ANY component copies
        it, flush state becomes inconsistent."""
        result = _build_bootstrap(tmp_path)
        assert result.collectors is result.registry.collectors
        # ValidationArtifactManager also holds a reference via its ctor;
        # changes to the dict must be visible there.
        assert result.validation_artifact_mgr._collectors is result.collectors

    def test_stages_passed_to_all_components(self, tmp_path: Path) -> None:
        """Regression: stages list must be identical across launch_preparator,
        restart_inspector, stage_execution_loop, and registry."""
        result = _build_bootstrap(tmp_path)
        assert result.launch_preparator._stages is result.registry.stages
        assert result.restart_inspector._stages is result.registry.stages


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize("num_stages", [0, 1, 3, 6])
def test_varying_stage_counts(tmp_path: Path, num_stages: int) -> None:
    with (
        patch(
            "src.pipeline.bootstrap.pipeline_bootstrap.load_config",
            return_value=_mk_config(),
        ),
        patch(
            "src.pipeline.bootstrap.pipeline_bootstrap.load_secrets",
            return_value=_mk_secrets(),
        ),
        patch(
            "src.pipeline.execution.stage_registry.StageRegistry._build_stages",
            return_value=[MagicMock(stage_name=f"S{i}") for i in range(num_stages)],
        ),
    ):
        result = PipelineBootstrap.build(
            config_path=tmp_path / "cfg.yaml",
            run_ctx=_mk_run_ctx(),
            settings=_mk_settings(tmp_path),
            attempt_controller=_make_attempt_controller(),
            on_stage_completed=lambda _: None,
            on_shutdown_signal=lambda _: None,
        )
    assert len(result.stages) == num_stages

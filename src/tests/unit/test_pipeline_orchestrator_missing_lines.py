from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator, run_pipeline
from src.pipeline.stages.constants import StageNames
from src.pipeline.state.models import PipelineState, StageRunState
from src.pipeline.state.store import SCHEMA_VERSION
from src.utils.result import AppError, Err, Ok


def _mk_orchestrator(
    *,
    config_path: Path,
    config: Any,
    secrets: Any,
    stages: list[Any] | None = None,
) -> PipelineOrchestrator:
    with (
        patch("src.pipeline.orchestrator.load_config", return_value=config),
        patch("src.pipeline.orchestrator.load_secrets", return_value=secrets),
        patch.object(PipelineOrchestrator, "_init_stages", return_value=stages or []),
    ):
        return PipelineOrchestrator(config_path)


def _mk_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "gpt2"
    cfg.training.type = "sft"
    cfg.training.strategies = []
    cfg.training.hyperparams.per_device_train_batch_size = 4
    cfg.training.get_strategy_chain.return_value = []
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.get_provider_config.return_value = {}
    cfg.get_primary_dataset.return_value = SimpleNamespace(
        get_source_type=lambda: "local",
        source_hf=None,
        source_local=SimpleNamespace(local_paths=SimpleNamespace(train="data/train.jsonl", eval=None)),
        adapter_type="chat",
    )
    cfg.experiment_tracking.mlflow = SimpleNamespace(
        tracking_uri="http://localhost:5002",
        system_metrics_callback_enabled=False,
    )
    cfg.get_adapter_config.side_effect = ValueError("no adapter")  # default: non-LoRA
    return cfg


def _mk_pipeline_state(run_dir: Path) -> PipelineState:
    return PipelineState(
        schema_version=SCHEMA_VERSION,
        logical_run_id="test_run",
        run_directory=str(run_dir),
        config_path=str(run_dir / "cfg.yaml"),
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_PENDING,
        training_critical_config_hash="h",
        late_stage_config_hash="h",
        attempts=[],
        current_output_lineage={},
    )


def _mk_secrets() -> MagicMock:
    secrets = MagicMock()
    secrets.hf_token = "hf_test"
    return secrets


class TestMissingInitAndMlflowSetupLines:
    def test_init_handles_provider_config_exception(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        cfg.get_provider_config.side_effect = RuntimeError("boom")
        secrets = _mk_secrets()

        # Should not raise: provider_type becomes None via except path
        _ = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=secrets, stages=[])

    def test_ensure_mlflow_preflight_raises_when_manager_missing(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=_mk_secrets(), stages=[])
        state = _mk_pipeline_state(tmp_path)
        with pytest.raises(Exception, match="MLflow setup failed"):
            orch._ensure_mlflow_preflight(state=state)

    def test_ensure_mlflow_preflight_surfaces_effective_uri_and_gateway_error(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        cfg.experiment_tracking.mlflow = SimpleNamespace(
            tracking_uri="https://public.example.ts.net",
            local_tracking_uri="http://localhost:5002",
            system_metrics_callback_enabled=False,
        )
        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=_mk_secrets(), stages=[])
        orch._mlflow_manager = MagicMock()
        orch._mlflow_manager.is_active = True
        orch._mlflow_manager.get_runtime_tracking_uri.return_value = "http://localhost:5002"
        orch._mlflow_manager.check_mlflow_connectivity.return_value = False
        orch._mlflow_manager.get_last_connectivity_error.return_value = AppError(
            message="certificate verify failed",
            code="MLFLOW_TLS_CERT_VERIFY_FAILED",
        )

        state = _mk_pipeline_state(tmp_path)
        with pytest.raises(Exception, match="effective_uri=http://localhost:5002"):
            orch._ensure_mlflow_preflight(state=state)

    def test_setup_mlflow_disable_system_metrics_logging_exception_is_ignored_and_manager_returns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = _mk_config()
        cfg.experiment_tracking.mlflow = SimpleNamespace(
            tracking_uri="http://localhost:5002",
            system_metrics_callback_enabled=True,
        )

        secrets = _mk_secrets()
        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=secrets, stages=[])

        # Fake mlflow module where disable_system_metrics_logging raises
        monkeypatch.setitem(
            __import__("sys").modules,
            "mlflow",
            SimpleNamespace(disable_system_metrics_logging=lambda: (_ for _ in ()).throw(RuntimeError("nope"))),
        )

        manager = MagicMock()
        manager.is_active = True

        with patch("src.pipeline.orchestrator.MLflowManager", return_value=manager):
            out = orch._setup_mlflow()

        assert out is manager
        manager.setup.assert_called_once_with(disable_system_metrics=True)

    def test_setup_mlflow_outer_exception_returns_none(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        cfg.experiment_tracking.mlflow = SimpleNamespace(
            tracking_uri="http://localhost:5002",
            system_metrics_callback_enabled=False,
        )
        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=_mk_secrets(), stages=[])

        with patch("src.pipeline.orchestrator.MLflowManager", side_effect=RuntimeError("boom")):
            assert orch._setup_mlflow() is None


class TestRunFinallyAndStageSpecificInfoMissingLines:
    def test_run_flushes_pending_collectors_in_finally(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Invariant: finally always calls _flush_pending_collectors even if stage failed."""
        cfg = _mk_config()
        cfg.experiment_tracking.mlflow = SimpleNamespace(
            tracking_uri="http://localhost:5002",
            system_metrics_callback_enabled=False,
        )

        stage = MagicMock()
        stage.stage_name = "Dataset Validator"
        stage.run.return_value = Ok(None)
        stage_list = [stage]

        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=_mk_secrets(), stages=stage_list)

        mgr = MagicMock()
        mgr.is_active = True
        mgr._run_id = "rid"
        ctx = MagicMock()
        mgr.start_run.return_value = ctx

        orch._setup_mlflow = MagicMock(return_value=mgr)
        orch._aggregate_training_metrics = MagicMock()
        orch._generate_experiment_report = MagicMock()

        flush_calls = []
        original_flush = orch._flush_pending_collectors

        def tracking_flush():
            flush_calls.append(True)
            original_flush()

        orch._flush_pending_collectors = tracking_flush

        # Fake mlflow module where disable_system_metrics_logging raises (run() try/except)
        monkeypatch.setitem(
            __import__("sys").modules,
            "mlflow",
            SimpleNamespace(disable_system_metrics_logging=lambda: (_ for _ in ()).throw(RuntimeError("nope"))),
        )

        mock_state = _mk_pipeline_state(tmp_path)
        mock_lock = MagicMock()

        def _fake_bootstrap(**kwargs):
            orch._state_store = MagicMock()
            orch._state_store.next_attempt_dir.return_value = tmp_path / "attempt_1"
            orch.run_directory = tmp_path
            return (mock_state, "fresh", "fresh", StageNames.DATASET_VALIDATOR)

        with (
            patch.object(orch, "_bootstrap_pipeline_state", side_effect=_fake_bootstrap),
            patch("src.pipeline.orchestrator.acquire_run_lock", return_value=mock_lock),
            patch.object(orch, "_save_state"),
        ):
            res = orch.run()
        assert res.is_ok()
        # _flush_pending_collectors must always be called in finally
        assert len(flush_calls) == 1

    def test_run_does_not_call_log_summary_artifact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression: log_summary_artifact(pipeline_events.json) is no longer called."""
        cfg = _mk_config()
        cfg.experiment_tracking.mlflow = SimpleNamespace(
            tracking_uri="http://localhost:5002",
            system_metrics_callback_enabled=False,
        )

        stage = MagicMock()
        stage.stage_name = "Dataset Validator"
        stage.run.return_value = Ok(None)

        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=_mk_secrets(), stages=[stage])

        mgr = MagicMock()
        mgr.is_active = True
        mgr._run_id = "rid"
        mgr.start_run.return_value = MagicMock()

        orch._setup_mlflow = MagicMock(return_value=mgr)
        orch._aggregate_training_metrics = MagicMock()
        orch._generate_experiment_report = MagicMock()

        monkeypatch.setitem(
            __import__("sys").modules,
            "mlflow",
            SimpleNamespace(disable_system_metrics_logging=lambda: None),
        )

        mock_state = _mk_pipeline_state(tmp_path)
        mock_lock = MagicMock()

        def _fake_bootstrap(**kwargs):
            orch._state_store = MagicMock()
            orch._state_store.next_attempt_dir.return_value = tmp_path / "attempt_1"
            orch.run_directory = tmp_path
            return (mock_state, "fresh", "fresh", StageNames.DATASET_VALIDATOR)

        with (
            patch.object(orch, "_bootstrap_pipeline_state", side_effect=_fake_bootstrap),
            patch("src.pipeline.orchestrator.acquire_run_lock", return_value=mock_lock),
            patch.object(orch, "_save_state"),
        ):
            orch.run()
        # pipeline_events.json logging is removed — log_summary_artifact must NOT be called
        mgr.log_summary_artifact.assert_not_called()

    def test_log_stage_specific_info_returns_when_no_mlflow(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._mlflow_manager = None
        orch._log_stage_specific_info("any")

    def test_log_stage_specific_info_gpu_deployer_logs_upload_and_deps_events(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._mlflow_manager = MagicMock()
        orch.context["GPU Deployer"] = {
            "provider_name": "mock",
            "provider_type": "cloud",
            "gpu_type": "A100",
            "resource_id": "pod123",
            "upload_duration_seconds": 1.2,
            "deps_duration_seconds": 3.4,
        }

        orch._log_stage_specific_info("GPU Deployer")
        assert orch._mlflow_manager.log_event_info.call_count >= 2

    def test_log_stage_specific_info_dataset_validator_plugin_metrics_handles_non_numeric(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._mlflow_manager = MagicMock()
        orch.context["Dataset Validator"] = {
            "validation_mode": "plugin",
            "metrics": {"avg_length": "not-a-number"},
            "sample_count": 10,
        }
        orch._log_stage_specific_info("Dataset Validator")
        orch._mlflow_manager.log_params.assert_called()

    def test_log_stage_specific_info_dataset_validator_legacy_metrics(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._mlflow_manager = MagicMock()
        orch.context["Dataset Validator"] = {
            "validation_mode": "legacy",
            "metrics": {"avg_length": 5, "empty_ratio": 0.0, "diversity_score": 0.1},
            "sample_count": 3,
        }
        orch._log_stage_specific_info("Dataset Validator")
        orch._mlflow_manager.log_params.assert_called()


class TestPrintSummaryCleanupAndMetricsCollectionMissingLines:
    def test_print_summary_branches(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = _mk_config()
        secrets = _mk_secrets()
        orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=cfg, secrets=secrets, stages=[])

        # Patch console to avoid noisy output
        fake_console = MagicMock()
        monkeypatch.setattr("src.pipeline.orchestrator.console", fake_console)

        # Case 1: huggingface dataset with eval_id -> covers HF branch prints
        hf_ds = SimpleNamespace(
            get_source_type=lambda: "huggingface",
            source_hf=SimpleNamespace(train_id="train", eval_id="eval"),
            source_local=None,
            adapter_type=None,
        )
        cfg.get_primary_dataset.return_value = hf_ds

        # Ensure adapter config path raises ValueError (except branch)
        cfg.get_adapter_config.side_effect = ValueError("no adapter")

        orch.context["Dataset Validator"] = {"sample_count": 10, "avg_length": 123}
        orch._print_summary()

        # Case 2: dataset source not configured (no hf/local)
        cfg.get_primary_dataset.return_value = SimpleNamespace(
            get_source_type=lambda: "huggingface",
            source_hf=None,
            source_local=None,
            adapter_type=None,
        )
        orch._print_summary()

        assert fake_console.print.called

    def test_cleanup_resources_exception_is_swallowed(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )

        calls: list[str] = []

        bad = MagicMock()
        bad.stage_name = "bad"
        bad.cleanup.side_effect = lambda: (calls.append("bad"), (_ for _ in ()).throw(RuntimeError("boom")))[-1]

        good = MagicMock()
        good.stage_name = "good"
        good.cleanup.side_effect = lambda: calls.append("good")

        # Ensure reverse order cleanup executes "good" first, then "bad"
        orch.stages = [bad, good]

        orch._cleanup_resources(success=False)

        # Must continue and swallow cleanup exceptions
        assert calls == ["good", "bad"]

    def test_cleanup_resources_skips_stage_without_cleanup(self, tmp_path: Path) -> None:
        """Dependency/edge: stage object without cleanup attribute must not break cleanup loop."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )

        # A stage-like object with no cleanup() at all
        stage_without_cleanup = SimpleNamespace(stage_name="no_cleanup")
        stage_with_cleanup = MagicMock()
        stage_with_cleanup.stage_name = "has_cleanup"
        stage_with_cleanup.cleanup = MagicMock()

        orch.stages = [stage_without_cleanup, stage_with_cleanup]
        orch._cleanup_resources(success=False)

        stage_with_cleanup.cleanup.assert_called_once()


@pytest.mark.parametrize("boom_idx", [0, 1, 2])
def test_cleanup_resources_calls_all_stages_even_if_some_cleanup_raise(boom_idx: int, tmp_path: Path) -> None:
    """Combinatorial: exception position in cleanup must not stop cleanup of other stages."""
    orch = _mk_orchestrator(config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[])

    calls: list[str] = []
    stages: list[Any] = []
    for i in range(3):
        st = MagicMock()
        st.stage_name = f"s{i}"

        if i == boom_idx:
            def _boom(i=i) -> None:
                calls.append(f"s{i}")
                raise RuntimeError("boom")

            st.cleanup.side_effect = _boom
        else:
            st.cleanup.side_effect = (lambda i=i: calls.append(f"s{i}"))

        stages.append(st)

    orch.stages = stages
    orch._cleanup_resources(success=False)

    # Reverse order is invariant, and all stages must be attempted
    assert calls == ["s2", "s1", "s0"]

    def test_aggregate_training_metrics_returns_when_no_mlflow_manager(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._mlflow_manager = None
        orch._aggregate_training_metrics()

    def test_collect_descendant_metrics_edge_cases_and_bfs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )

        # No mlflow manager
        orch._mlflow_manager = None
        assert orch._collect_descendant_metrics() == []

        # Manager without run id
        mgr = MagicMock()
        mgr._run_id = None
        orch._mlflow_manager = mgr
        assert orch._collect_descendant_metrics() == []

        # BFS collection with fake MlflowClient
        mgr._run_id = "parent"
        orch._mlflow_manager = mgr

        class _Run:
            def __init__(self, run_id: str, run_name: str, metrics: dict[str, float]):
                self.info = SimpleNamespace(run_id=run_id, run_name=run_name, experiment_id="exp")
                self.data = SimpleNamespace(metrics=metrics)

        class _Client:
            def get_run(self, run_id: str):
                return _Run(run_id, "pipeline", {})

            def search_runs(self, experiment_ids: list[str], filter_string: str):
                if "parentRunId` = 'parent'" in filter_string:
                    return [
                        _Run("phase0", "phase_0_sft", {"train_loss": 1.0}),
                        _Run("child", "strategy_run", {}),
                    ]
                if "parentRunId` = 'child'" in filter_string:
                    return [
                        _Run("phase1", "phase_1_dpo", {"train_loss": 2.0}),
                    ]
                return []

        monkeypatch.setitem(__import__("sys").modules, "mlflow", SimpleNamespace())
        monkeypatch.setitem(__import__("sys").modules, "mlflow.tracking", SimpleNamespace(MlflowClient=_Client))

        metrics = orch._collect_descendant_metrics(max_depth=2)
        assert len(metrics) == 2
        assert metrics[0]["train_loss"] in (1.0, 2.0)

    def test_generate_experiment_report_no_run_id_returns(self, tmp_path: Path) -> None:
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._generate_experiment_report(run_id=None)


class TestDatasetValidatorCallbacksAndRunPipeline:
    def test_dataset_validator_callbacks_populate_accumulator(self, tmp_path: Path) -> None:
        """Positive: callbacks fill _validation_accumulator without MLflow."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        mlflow_mock = MagicMock()
        orch._mlflow_manager = mlflow_mock

        orch._validation_artifact_mgr.on_dataset_scheduled("ds", "/tmp/x", "plugin")
        # After scheduled — accumulator has entry
        assert "/tmp/x" in orch._validation_artifact_mgr._validation_accumulator
        entry = orch._validation_artifact_mgr._validation_accumulator["/tmp/x"]
        assert entry["name"] == "ds"
        assert entry["status"] == "scheduled"

        orch._validation_artifact_mgr.on_dataset_loaded("ds", "/tmp/x", 10, 0)
        assert entry["sample_count"] == 10

        orch._validation_artifact_mgr.on_plugin_start("ds", "/tmp/x", "p_main", "p", "desc")  # no-op
        orch._validation_artifact_mgr.on_plugin_complete(
            "ds",
            "/tmp/x",
            "p_main",
            "p",
            params={"x": 1},
            thresholds={},
            metrics={"m": 2},
            duration_ms=1.0,
        )
        assert len(entry["plugins"]) == 1
        assert entry["plugins"][0]["passed"] is True
        assert entry["plugins"][0]["id"] == "p_main"
        assert entry["plugins"][0]["plugin_name"] == "p"
        assert entry["plugins"][0]["description"] == "desc"

        orch._validation_artifact_mgr.on_validation_completed("ds", "/tmp/x", {"a": 1}, ["w"])
        assert entry["status"] == "passed"

        orch._validation_artifact_mgr.on_plugin_start("ds", "/tmp/x", "p_main", "p", "desc failed")
        orch._validation_artifact_mgr.on_plugin_failed(
            "ds", "/tmp/x", "p_main", "p", {"x": 1}, {}, {"m": 2}, 123.4, ["e"], ["r"]
        )
        assert len(entry["plugins"]) == 2
        assert entry["plugins"][1]["passed"] is False
        assert entry["plugins"][1]["description"] == "desc failed"
        assert entry["plugins"][1]["duration_ms"] == 123.4

        orch._validation_artifact_mgr.on_validation_failed("ds", "/tmp/x", ["e"])
        assert entry["status"] == "failed"

        # MLflow is NOT called by callbacks anymore — all goes to accumulator
        mlflow_mock.log_event_info.assert_not_called()

    def test_dataset_validator_callbacks_multiple_datasets(self, tmp_path: Path) -> None:
        """Invariant: multiple datasets accumulate independently in accumulator."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        orch._validation_artifact_mgr.on_dataset_scheduled("ds1", "/tmp/a", "plugin")
        orch._validation_artifact_mgr.on_dataset_scheduled("ds2", "/tmp/b", "plugin")
        assert len(orch._validation_artifact_mgr._validation_accumulator) == 2
        assert orch._validation_artifact_mgr._validation_accumulator["/tmp/a"]["name"] == "ds1"
        assert orch._validation_artifact_mgr._validation_accumulator["/tmp/b"]["name"] == "ds2"

    def test_dataset_validator_on_dataset_loaded_ignores_missing_path(self, tmp_path: Path) -> None:
        """Boundary: _on_dataset_loaded for unknown path does not error."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        # path not in accumulator — should be a no-op
        orch._validation_artifact_mgr.on_dataset_loaded("ds", "/nonexistent", 5, 0)  # must not raise

    def test_run_pipeline_exit_codes(self) -> None:
        with patch("src.pipeline.orchestrator.PipelineOrchestrator") as MockOrch:
            inst = MockOrch.return_value

            inst.run.return_value = Ok({})
            assert run_pipeline("config.yaml") == 0

            inst.run.return_value = Err("boom")
            assert run_pipeline("config.yaml") == 1

            inst.run.side_effect = KeyboardInterrupt()
            assert run_pipeline("config.yaml") == 130


# =============================================================================
# _fill_from_context
# =============================================================================

class TestFillFromContext:
    """Tests for PipelineOrchestrator._fill_from_context."""

    def _mk(self, tmp_path: Path):
        return _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )

    def test_fill_gpu_deployer(self, tmp_path: Path) -> None:
        """Positive: GPU Deployer reads upload/deps/provider/gpu/resource from context."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context[StageNames.GPU_DEPLOYER] = {
            "upload_duration_seconds": 12.3,
            "deps_duration_seconds": 4.5,
            "provider_name": "runpod",
            "provider_type": "cloud",
            "gpu_type": "A100",
            "resource_id": "pod123",
        }
        col = StageArtifactCollector(stage=StageNames.GPU_DEPLOYER, artifact_name="gpu.json")
        orch._fill_from_context(StageNames.GPU_DEPLOYER, col)
        assert col._data["upload_duration_seconds"] == 12.3
        assert col._data["provider_name"] == "runpod"
        assert col._data["gpu_type"] == "A100"

    def test_fill_training_monitor(self, tmp_path: Path) -> None:
        """Positive: Training Monitor reads training_duration_seconds."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context[StageNames.TRAINING_MONITOR] = {"training_duration_seconds": 3600.0}
        col = StageArtifactCollector(stage=StageNames.TRAINING_MONITOR, artifact_name="t.json")
        orch._fill_from_context(StageNames.TRAINING_MONITOR, col)
        assert col._data["training_duration_seconds"] == 3600.0

    def test_fill_model_retriever(self, tmp_path: Path) -> None:
        """Positive: Model Retriever reads model_size_mb, hf_repo_id, upload_duration_seconds."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context[StageNames.MODEL_RETRIEVER] = {
            "model_size_mb": 1500.0,
            "hf_repo_id": "org/model-adapter",
            "upload_duration_seconds": 60.0,
        }
        col = StageArtifactCollector(stage=StageNames.MODEL_RETRIEVER, artifact_name="m.json")
        orch._fill_from_context(StageNames.MODEL_RETRIEVER, col)
        assert col._data["model_size_mb"] == 1500.0
        assert col._data["hf_repo_id"] == "org/model-adapter"

    def test_fill_inference_deployer(self, tmp_path: Path) -> None:
        """Positive: Inference Deployer reads endpoint_url, model_name, provider."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context[StageNames.INFERENCE_DEPLOYER] = {
            "endpoint_url": "http://localhost:8000/v1",
            "model_name": "test-model",
            "provider": "single_node",
        }
        col = StageArtifactCollector(stage=StageNames.INFERENCE_DEPLOYER, artifact_name="i.json")
        orch._fill_from_context(StageNames.INFERENCE_DEPLOYER, col)
        assert col._data["endpoint_url"] == "http://localhost:8000/v1"
        assert col._data["provider"] == "single_node"

    def test_fill_model_evaluator_with_eval_summary(self, tmp_path: Path) -> None:
        """Positive: Model Evaluator expands eval_summary into data."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context[StageNames.MODEL_EVALUATOR] = {
            "eval_summary": {"overall_passed": True, "sample_count": 10}
        }
        col = StageArtifactCollector(stage=StageNames.MODEL_EVALUATOR, artifact_name="e.json")
        orch._fill_from_context(StageNames.MODEL_EVALUATOR, col)
        assert col._data["overall_passed"] is True
        assert col._data["sample_count"] == 10

    def test_fill_missing_stage_context_is_noop(self, tmp_path: Path) -> None:
        """Boundary: missing stage_name in context → no crash, empty data."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        col = StageArtifactCollector(stage=StageNames.GPU_DEPLOYER, artifact_name="g.json")
        orch._fill_from_context(StageNames.GPU_DEPLOYER, col)
        # GPU deployer with empty context: all fields are None
        assert col._data.get("provider_name") is None

    def test_fill_non_dict_context_is_noop(self, tmp_path: Path) -> None:
        """Boundary: context[stage_name] not dict → no crash."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context[StageNames.TRAINING_MONITOR] = "not_a_dict"
        col = StageArtifactCollector(stage=StageNames.TRAINING_MONITOR, artifact_name="t.json")
        orch._fill_from_context(StageNames.TRAINING_MONITOR, col)
        assert col._data == {}

    def test_fill_unknown_stage_name_is_noop(self, tmp_path: Path) -> None:
        """Boundary: unknown stage_name → no crash, data unchanged."""
        from src.pipeline.artifacts.base import StageArtifactCollector

        orch = self._mk(tmp_path)
        orch.context["Unknown Stage"] = {"x": 1}
        col = StageArtifactCollector(stage="Unknown Stage", artifact_name="u.json")
        orch._fill_from_context("Unknown Stage", col)
        assert col._data == {}


# =============================================================================
# _flush_pending_collectors
# =============================================================================

class TestFlushPendingCollectors:
    def _mk(self, tmp_path: Path):
        return _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )

    def test_flushes_only_started_collectors(self, tmp_path: Path) -> None:
        """Positive: only collectors with set_started_at get interrupted status."""
        orch = self._mk(tmp_path)
        # None have started_at — all should be skipped (not flushed)
        orch._flush_pending_collectors()
        assert all(not c.is_flushed for c in orch._collectors.values())

    def test_flushes_started_collectors_as_interrupted(self, tmp_path: Path) -> None:
        """Positive: collectors with started_at flush as interrupted."""

        orch = self._mk(tmp_path)
        gpu_col = orch._collectors[StageNames.GPU_DEPLOYER]
        gpu_col.set_started_at("2026-01-01T00:00:00")
        train_col = orch._collectors[StageNames.TRAINING_MONITOR]
        train_col.set_started_at("2026-01-01T00:01:00")

        orch._flush_pending_collectors()

        assert gpu_col.is_flushed
        assert train_col.is_flushed
        # Collectors without started_at remain not flushed
        for name, col in orch._collectors.items():
            if name not in (StageNames.GPU_DEPLOYER, StageNames.TRAINING_MONITOR):
                assert not col.is_flushed

    def test_not_started_collectors_produce_no_artifact(self, tmp_path: Path) -> None:
        """Invariant: stages without set_started_at do not write artifacts (empty noise)."""
        orch = self._mk(tmp_path)
        for col in orch._collectors.values():
            assert col._started_at is None
        orch._flush_pending_collectors()
        # Nothing flushed
        assert all(not c.is_flushed for c in orch._collectors.values())

    def test_skips_already_flushed_collectors(self, tmp_path: Path) -> None:
        """Invariant: already flushed collectors are not overwritten."""
        orch = self._mk(tmp_path)
        gpu_col = orch._collectors[StageNames.GPU_DEPLOYER]
        gpu_col.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert gpu_col.is_flushed

        # Even if we set started_at, already flushed collectors are skipped
        gpu_col.set_started_at("2026-01-01T00:00:00")
        orch._flush_pending_collectors()
        # Still ok (not changed to interrupted)
        assert gpu_col.is_flushed

    def test_uses_set_started_at_if_available(self, tmp_path: Path) -> None:
        """Positive: set_started_at value is used in envelope when set."""
        from src.pipeline.artifacts.base import STATUS_INTERRUPTED

        orch = self._mk(tmp_path)
        col = orch._collectors[StageNames.TRAINING_MONITOR]
        ts = "2026-01-15T10:00:00"
        col.set_started_at(ts)

        flushed_envelopes = []
        orig_flush = col.flush_interrupted

        def tracking_flush(**kwargs):
            env = orig_flush(**kwargs)
            if env:
                flushed_envelopes.append(env)
            return env

        col.flush_interrupted = tracking_flush
        orch._flush_pending_collectors()

        assert len(flushed_envelopes) == 1
        assert flushed_envelopes[0].started_at == ts
        assert flushed_envelopes[0].status == STATUS_INTERRUPTED

    def test_exception_during_flush_does_not_stop_others(self, tmp_path: Path) -> None:
        """Dependency errors: exception in one flush_interrupted does not stop others."""
        orch = self._mk(tmp_path)
        all_collectors = list(orch._collectors.values())

        # Give all collectors a started_at so they are eligible for flushing
        for col in all_collectors:
            col.set_started_at("2026-01-01T00:00:00")

        # Make the first collector raise
        first_col = all_collectors[0]

        def bad_flush(**kwargs):
            raise RuntimeError("disk full")

        first_col.flush_interrupted = bad_flush

        # Should not propagate the exception
        orch._flush_pending_collectors()

        # Other collectors still got flushed
        for col in all_collectors[1:]:
            assert col.is_flushed


# =============================================================================
# _flush_validation_artifact
# =============================================================================

class TestFlushValidationArtifact:
    def _mk(self, tmp_path: Path):
        return _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )

    def test_all_passed_flushes_ok(self, tmp_path: Path) -> None:
        """Positive: all datasets passed → collector.flush_ok."""

        orch = self._mk(tmp_path)
        orch._validation_artifact_mgr._validation_accumulator = {
            "/tmp/a": {"name": "ds1", "path": "/tmp/a", "sample_count": 10,
                       "status": "passed", "critical_failures": 0, "plugins": []},
        }
        collector = orch._collectors[StageNames.DATASET_VALIDATOR]
        orch._validation_artifact_mgr.flush_validation_artifact(started_at="t", duration_seconds=1.0)
        assert collector.is_flushed
        assert collector._is_flushed

    def test_any_failed_flushes_error(self, tmp_path: Path) -> None:
        """Positive: at least one dataset failed → collector.flush_error."""
        orch = self._mk(tmp_path)
        orch._validation_artifact_mgr._validation_accumulator = {
            "/tmp/a": {"name": "ds1", "path": "/tmp/a", "sample_count": 10,
                       "status": "passed", "critical_failures": 0, "plugins": []},
            "/tmp/b": {"name": "ds2", "path": "/tmp/b", "sample_count": 5,
                       "status": "failed", "critical_failures": 1, "plugins": []},
        }
        collector = orch._collectors[StageNames.DATASET_VALIDATOR]
        orch._validation_artifact_mgr.flush_validation_artifact(started_at="t", duration_seconds=2.0)
        assert collector.is_flushed

    def test_empty_accumulator_passes(self, tmp_path: Path) -> None:
        """Boundary: empty accumulator → flush_ok (nothing to check)."""
        orch = self._mk(tmp_path)
        orch._validation_artifact_mgr._validation_accumulator = {}
        collector = orch._collectors[StageNames.DATASET_VALIDATOR]
        orch._validation_artifact_mgr.flush_validation_artifact(started_at="t", duration_seconds=0.0)
        assert collector.is_flushed

    def test_already_flushed_collector_is_noop(self, tmp_path: Path) -> None:
        """Invariant: if collector already flushed → no-op."""
        orch = self._mk(tmp_path)
        collector = orch._collectors[StageNames.DATASET_VALIDATOR]
        collector.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert collector.is_flushed

        # Call again — should be no-op
        orch._validation_artifact_mgr.flush_validation_artifact(started_at="t", duration_seconds=1.0)
        assert collector.is_flushed  # still flushed, no error

    def test_scheduled_status_treated_as_passed(self, tmp_path: Path) -> None:
        """Boundary: status='scheduled' (not loaded) → treated as passed (not failed)."""
        orch = self._mk(tmp_path)
        orch._validation_artifact_mgr._validation_accumulator = {
            "/tmp/x": {"name": "ds", "path": "/tmp/x", "sample_count": None,
                       "status": "scheduled", "critical_failures": 0, "plugins": []},
        }
        collector = orch._collectors[StageNames.DATASET_VALIDATOR]
        orch._validation_artifact_mgr.flush_validation_artifact(started_at="t", duration_seconds=0.0)
        assert collector.is_flushed
        # 'scheduled' treated as passed → flush_ok, no error


# =============================================================================
# _init_collectors
# =============================================================================

class TestInitCollectors:
    def test_all_stage_names_have_collector(self, tmp_path: Path) -> None:
        """Invariant: each stage has a collector in _collectors."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        expected_stages = {
            StageNames.DATASET_VALIDATOR,
            StageNames.GPU_DEPLOYER,
            StageNames.TRAINING_MONITOR,
            StageNames.MODEL_RETRIEVER,
            StageNames.INFERENCE_DEPLOYER,
            StageNames.MODEL_EVALUATOR,
        }
        assert expected_stages.issubset(set(orch._collectors.keys()))

    def test_collectors_are_not_flushed_at_init(self, tmp_path: Path) -> None:
        """Invariant: on init all collectors are not flushed."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        for stage_name, col in orch._collectors.items():
            assert not col.is_flushed, f"Collector for {stage_name} should not be flushed at init"

    def test_validation_accumulator_empty_at_init(self, tmp_path: Path) -> None:
        """Invariant: _validation_accumulator is empty on init."""
        orch = _mk_orchestrator(
            config_path=tmp_path / "cfg.yaml", config=_mk_config(), secrets=_mk_secrets(), stages=[]
        )
        assert orch._validation_artifact_mgr._validation_accumulator == {}

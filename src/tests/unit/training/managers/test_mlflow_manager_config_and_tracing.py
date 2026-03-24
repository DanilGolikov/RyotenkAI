from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.training.managers.mlflow_manager import MLflowManager
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    ExperimentTrackingConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    LoraConfig,
    MLflowConfig,
    ModelConfig,
    PhaseHyperparametersConfig,
    PipelineConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype="bfloat16",
        trust_remote_code=False,
    )


def _lora_cfg() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )


def _hp_global_cfg() -> GlobalHyperparametersConfig:
    return GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )


def _inference_cfg_disabled() -> InferenceConfig:
    return InferenceConfig(
        enabled=False,
        provider="single_node",
        engine="vllm",
        engines=InferenceEnginesConfig(
            vllm=InferenceVLLMEngineConfig(
                merge_image="test/merge:latest",
                serve_image="test/vllm:latest",
            )
        ),
    )


def _ds_local(train_path: str) -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train=train_path, eval=None)),
    )


def _mk_cfg() -> PipelineConfig:
    return PipelineConfig(
        model=_model_cfg(),
        training=TrainingOnlyConfig(
            type="qlora",
            provider="runpod",
            lora=_lora_cfg(),
            hyperparams=_hp_global_cfg(),
            strategies=[
                StrategyPhaseConfig(
                    strategy_type="sft",
                    dataset="default",
                    hyperparams=PhaseHyperparametersConfig(epochs=1),
                ),
                StrategyPhaseConfig(
                    strategy_type="dpo",
                    dataset="default",
                    hyperparams=PhaseHyperparametersConfig(epochs=2),
                ),
            ],
        ),
        providers={
            "runpod": {
                "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
                "cleanup": {},
                "training": {"gpu_type": "A40", "image_name": "test/training-runtime:latest"},
                "inference": {},
            }
        },
        datasets={"default": _ds_local("data/train.jsonl")},
        inference=_inference_cfg_disabled(),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                enabled=True,
                tracking_uri="http://127.0.0.1:5002",
                experiment_name="test",
                log_artifacts=True,
                log_model=True,
            )
        ),
    )


class TestConfigLogging:
    def test_log_training_config_builds_params_and_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)

        seen_params: dict[str, Any] = {}
        seen_tags: dict[str, str] = {}

        monkeypatch.setattr(mgr, "log_params", lambda p: seen_params.update(p))
        monkeypatch.setattr(mgr, "set_tags", lambda t: seen_tags.update(t))

        mgr.log_training_config(cfg)

        assert seen_params["model_name"] == cfg.model.name
        assert seen_params["training_type"] == cfg.training.type
        assert "lora_r" in seen_params
        assert "strategy_chain" in seen_tags
        assert seen_tags["num_phases"] == "2"

    def test_log_pipeline_config_collects_provider_tags_and_hyperparams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)
        tags: dict[str, str] = {}
        params: dict[str, Any] = {}

        monkeypatch.setattr(mgr, "set_tags", lambda t: tags.update(t))
        monkeypatch.setattr(mgr, "log_params", lambda p: params.update(p))

        mgr.log_pipeline_config(cfg)

        assert tags["provider.name"] == "runpod"
        assert tags["provider.gpu_type"] == "A40"
        assert params["config.model.name"] == cfg.model.name
        assert params["config.training.type"] == cfg.training.type
        # at least one hyperparam logged
        assert any(k.startswith("training.hyperparams.") for k in params)
        # phase-specific params logged
        assert params["config.strategy.0.type"] == "sft"
        assert params["config.strategy.0.dataset"] == "default"

    def test_log_pipeline_config_handles_provider_exceptions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        # No provider -> get_active_provider_name raises -> tags fallback to unknown
        cfg.training.provider = None

        mgr = MLflowManager(cfg)
        tags: dict[str, str] = {}
        monkeypatch.setattr(mgr, "set_tags", lambda t: tags.update(t))
        monkeypatch.setattr(mgr, "log_params", lambda p: None)

        mgr.log_pipeline_config(cfg)
        assert tags["provider.name"] == "unknown"

    def test_log_dataset_config_hf_and_error_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)
        params: dict[str, Any] = {}
        tags: dict[str, str] = {}

        monkeypatch.setattr(mgr, "log_params", lambda p: params.update(p))
        monkeypatch.setattr(mgr, "set_tags", lambda t: tags.update(t))

        cfg.datasets["default"] = DatasetConfig(
            source_type="huggingface",
            source_hf={"train_id": "org/ds", "eval_id": "org/ds_eval"},
            max_samples=123,
        )

        mgr.log_dataset_config(cfg)
        assert params["dataset.default.source_type"] == "huggingface"
        assert params["dataset.default.hf.train_id"] == "org/ds"
        assert params["dataset.default.max_samples"] == "123"
        assert tags["dataset.names"] == "default"
        assert tags["dataset.count"] == "1"

        # exception path should not raise
        cfg.datasets.clear()
        mgr.log_dataset_config(cfg)


@dataclass
class _MetricPoint:
    step: int
    timestamp: int
    value: float


class TestRunQueries:
    def test_get_run_metrics_history_success_and_failure(self) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)

        class _Client:
            def get_metric_history(self, run_id: str, metric: str):
                _ = run_id, metric
                return [_MetricPoint(step=1, timestamp=2, value=3.0)]

        from unittest.mock import MagicMock
        from src.training.mlflow.run_analytics import MLflowRunAnalytics
        mock_gw = MagicMock()
        mock_gw.get_client.return_value = _Client()
        mgr._gateway = mock_gw
        mlflow_mod = object()  # mark as initialized
        mgr._mlflow = mlflow_mod
        mgr._analytics = MLflowRunAnalytics(mock_gw, mlflow_mod, experiment_name="test", event_log=mgr._event_log)

        out = mgr.get_run_metrics_history("r", "m")
        assert out == [{"step": 1, "timestamp": 2, "value": 3.0}]

        class _BadClient:
            def get_metric_history(self, run_id: str, metric: str):
                raise RuntimeError("boom")

        mock_gw.get_client.return_value = _BadClient()
        assert mgr.get_run_metrics_history("r", "m") == []

    def test_get_experiment_summary_empty_and_metrics(self) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)

        from src.training.mlflow.run_analytics import MLflowRunAnalytics
        from unittest.mock import MagicMock

        # empty df
        mlflow_empty = SimpleNamespace(search_runs=lambda **kw: pd.DataFrame())
        mgr._mlflow = mlflow_empty
        mgr._analytics = MLflowRunAnalytics(MagicMock(), mlflow_empty, experiment_name="x", event_log=mgr._event_log)
        assert mgr.get_experiment_summary(experiment_name="x") == {"total_runs": 0}

        df = pd.DataFrame(
            [
                {"metrics.eval_loss": 0.5, "metrics.train_loss": 1.0},
                {"metrics.eval_loss": 0.2, "metrics.train_loss": 2.0},
            ]
        )
        mlflow_df = SimpleNamespace(search_runs=lambda **kw: df)
        mgr._mlflow = mlflow_df
        mgr._analytics = MLflowRunAnalytics(MagicMock(), mlflow_df, experiment_name="x", event_log=mgr._event_log)
        s = mgr.get_experiment_summary(experiment_name="x")
        assert s["total_runs"] == 2
        assert s["best_eval_loss"] == 0.2
        assert s["worst_train_loss"] == 2.0


class TestAutologAndTracing:
    def test_enable_pytorch_autolog_success_and_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)

        fake_mlflow = ModuleType("mlflow")
        calls: list[dict[str, Any]] = []
        pytorch_mod = ModuleType("mlflow.pytorch")
        pytorch_mod.autolog = lambda **kw: calls.append(kw)  # type: ignore[attr-defined]

        fake_mlflow.pytorch = pytorch_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setitem(sys.modules, "mlflow.pytorch", pytorch_mod)

        mgr._mlflow = fake_mlflow
        assert mgr.enable_pytorch_autolog(log_models=True, log_every_n_epoch=2, log_every_n_step=3) is True
        assert calls and calls[-1]["log_every_n_epoch"] == 2

        # failure branch
        pytorch_mod.autolog = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
        assert mgr.enable_pytorch_autolog() is False

    def test_enable_disable_tracing_and_trace_io(self) -> None:
        cfg = _mk_cfg()
        mgr = MLflowManager(cfg)

        # tracing object present
        enabled: list[bool] = []
        tracing = SimpleNamespace(enable=lambda: enabled.append(True), disable=lambda: enabled.append(False))

        class _Span:
            def __init__(self):
                self.inputs = None
                self.outputs = None
                self.attrs = None

            def set_inputs(self, d):
                self.inputs = d

            def set_outputs(self, d):
                self.outputs = d

            def set_attributes(self, d):
                self.attrs = d

        span = _Span()

        fake_mlflow = SimpleNamespace(
            tracing=tracing,
            start_span=lambda name, attributes: nullcontext(span),  # noqa: ARG005
            get_current_active_span=lambda: span,
        )
        mgr._mlflow = fake_mlflow

        assert mgr.enable_tracing() is True
        assert mgr.disable_tracing() is True

        with mgr.trace_llm_call("x", model_name="m", attributes={"a": 1}) as s:
            assert s is span
            mgr.log_trace_io(input_data="in", output_data={"out": "x"}, input_tokens=1, output_tokens=2, latency_ms=3.0)

        assert span.inputs == {"input": "in"}
        assert span.outputs == {"out": "x"}
        assert span.attrs["input_tokens"] == 1

        # attribute missing branch
        mgr._mlflow = SimpleNamespace(tracing=SimpleNamespace())
        assert mgr.enable_tracing() is False

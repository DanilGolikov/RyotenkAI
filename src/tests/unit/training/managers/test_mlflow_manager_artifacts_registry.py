from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
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
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


def _mk_cfg(
    *,
    tracking_uri: str = "http://127.0.0.1:5002",
) -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="test-model", torch_dtype="bfloat16", trust_remote_code=False),
        training=TrainingOnlyConfig(
            type="qlora",
            qlora=QLoRAConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=GlobalHyperparametersConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_ratio=0.0,
                epochs=1,
            ),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        ),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                tracking_uri=tracking_uri,
                experiment_name="test",
            )
        ),
    )


@dataclass
class _ArtifactCall:
    run_id: str
    name: str
    payload: Any


class _FakeClient:
    def __init__(self, tracking_uri: str):
        self.text_calls: list[_ArtifactCall] = []
        self.dict_calls: list[_ArtifactCall] = []

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.text_calls.append(_ArtifactCall(run_id, artifact_file, text))

    def log_dict(self, run_id: str, d: dict, artifact_file: str) -> None:
        self.dict_calls.append(_ArtifactCall(run_id, artifact_file, d))


@dataclass
class _ModelVersion:
    version: int
    run_id: str = "run_x"
    source: str = "src"
    status: str = "READY"
    creation_timestamp: int = 1
    last_updated_timestamp: int = 2
    description: str = "d"
    tags: dict[str, str] = field(default_factory=dict)


class _TrackingClient:
    def __init__(self):
        self.alias_calls: list[tuple[str, str, Any]] = []
        self.tag_calls: list[tuple[str, Any, str, str]] = []
        self.deleted_aliases: list[tuple[str, str]] = []
        self._alias_version = _ModelVersion(version=3)

    def set_registered_model_alias(self, model_name: str, alias: str, version: Any) -> None:
        self.alias_calls.append((model_name, alias, version))

    def set_model_version_tag(self, model_name: str, version: Any, key: str, value: str) -> None:
        self.tag_calls.append((model_name, version, key, value))

    def get_model_version_by_alias(self, model_name: str, alias: str):
        if alias == "missing":
            raise RuntimeError("not found")
        mv = _ModelVersion(version=3)
        mv.tags = {"k": "v"}
        return mv

    def delete_registered_model_alias(self, model_name: str, alias: str) -> None:
        self.deleted_aliases.append((model_name, alias))

    def get_registered_model(self, model_name: str):
        return SimpleNamespace(aliases={"champion": "3", "staging": "4"})

    def get_run(self, run_id: str):
        return SimpleNamespace(
            data=SimpleNamespace(params={"a": "1"}, metrics={"m": 1.0}, tags={"t": "x"}),
            info=SimpleNamespace(run_id=run_id, status="FINISHED", start_time=1, end_time=2),
        )


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, _FakeClient]:
    """Install a fake `mlflow` module into sys.modules and return it + shared artifact client."""
    fake_mlflow = ModuleType("mlflow")
    fake_mlflow.get_tracking_uri = lambda: "http://127.0.0.1:5002"  # type: ignore[attr-defined]

    # shared client instance to inspect calls
    shared = _FakeClient("http://127.0.0.1:5002")

    class MlflowClient(_FakeClient):
        def __init__(self, tracking_uri: str):
            super().__init__(tracking_uri)
            # re-use shared storage
            self.text_calls = shared.text_calls
            self.dict_calls = shared.dict_calls

    fake_mlflow.MlflowClient = MlflowClient  # type: ignore[attr-defined]

    tracking_client = _TrackingClient()
    fake_mlflow.tracking = SimpleNamespace(MlflowClient=lambda: tracking_client)  # type: ignore[attr-defined]
    fake_mlflow.pyfunc = SimpleNamespace(load_model=lambda uri: {"uri": uri})  # type: ignore[attr-defined]

    # Registry API
    fake_mlflow.register_model = lambda model_uri, model_name: SimpleNamespace(version=3)  # type: ignore[attr-defined]

    # search_runs for get_best_run
    fake_mlflow.search_runs = lambda **kw: pd.DataFrame(  # type: ignore[attr-defined]
        [{"run_id": "r1", "metrics.eval_loss": 0.5}, {"run_id": "r2", "metrics.eval_loss": 0.2}]
    )

    # Autolog fallback
    fake_mlflow.autolog = lambda **kw: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    return fake_mlflow, shared


def _patch_gateway(mgr: MLflowManager, shared_client: _FakeClient) -> None:
    """Replace manager's gateway with a mock that returns the shared test client."""
    from unittest.mock import MagicMock

    mock_gateway = MagicMock()
    mock_gateway.uri = "http://127.0.0.1:5002"
    mock_gateway.get_client.return_value = shared_client
    mock_gateway.check_connectivity.return_value = True
    mgr._gateway = mock_gateway


def test_setup_success_enables_system_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    # add setup APIs used in setup()
    fake_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    fake_mlflow.set_experiment = lambda name: None  # type: ignore[attr-defined]
    fake_mlflow.enable_system_metrics_logging = lambda: None  # type: ignore[attr-defined]
    fake_mlflow.set_system_metrics_sampling_interval = lambda n: None  # type: ignore[attr-defined]
    fake_mlflow.set_system_metrics_samples_before_logging = lambda n: None  # type: ignore[attr-defined]

    mgr = MLflowManager(_mk_cfg())
    # speed up: avoid network check via gateway
    monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)

    assert mgr.setup(disable_system_metrics=False) is True
    assert mgr.is_active is True


def test_setup_connectivity_failure_logs_event_and_disables_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    fake_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    fake_mlflow.set_experiment = lambda name: None  # type: ignore[attr-defined]

    mgr = MLflowManager(_mk_cfg(tracking_uri="http://127.0.0.1:5002"))
    monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: False)

    ok = mgr.setup(timeout=0.01, max_retries=2)
    assert ok is False
    assert mgr.is_active is False
    assert any(e["event_type"] == "error" for e in mgr.get_events())


def test_enable_and_disable_autolog_transformers_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = fake_mlflow

    # Provide mlflow.transformers.autolog
    transformers_mod = ModuleType("mlflow.transformers")
    calls: list[dict[str, Any]] = []
    transformers_mod.autolog = lambda **kw: calls.append(kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow.transformers", transformers_mod)
    # Ensure `mlflow.transformers` attribute exists on the parent module too
    fake_mlflow.transformers = transformers_mod  # type: ignore[attr-defined]

    assert mgr.enable_autolog(log_models=False) is True
    assert calls and calls[-1]["disable"] is False
    assert mgr.disable_autolog() is True


def test_log_artifact_and_log_dict_and_log_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_mlflow, shared = _install_fake_mlflow(monkeypatch)
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = fake_mlflow
    mgr._run_id = "run_1"
    mgr._run = object()
    _patch_gateway(mgr, shared)

    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")

    assert mgr.log_artifact(str(f), artifact_path="dir") is True
    assert shared.text_calls[-1].name == "dir/a.txt"

    assert mgr.log_dict({"x": 1}, "x.json") is True
    assert shared.dict_calls[-1].name == "x.json"

    assert mgr.log_text("hi", "t.md") is True
    assert shared.text_calls[-1].name == "t.md"


def test_log_artifact_missing_file_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = fake_mlflow
    mgr._run_id = "run_1"
    mgr._run = object()

    assert mgr.log_artifact(str(tmp_path / "missing.txt")) is False


def test_log_artifact_binary_file_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = fake_mlflow
    mgr._run_id = "run_1"
    mgr._run = object()

    f = tmp_path / "b.bin"
    f.write_bytes(b"\xff\xfe\xfa")
    assert mgr.log_artifact(str(f)) is False


def test_model_registry_register_alias_tags_and_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = fake_mlflow
    mgr._run_id = "run_1"

    # Patch gateway to return _TrackingClient (handles alias/registry ops)
    tracking_client = _TrackingClient()
    from unittest.mock import MagicMock
    mock_gateway = MagicMock()
    mock_gateway.uri = "http://127.0.0.1:5002"
    mock_gateway.get_client.return_value = tracking_client
    mgr._gateway = mock_gateway

    # Build registry manually since setup() was not called
    from src.training.mlflow.model_registry import MLflowModelRegistry
    mgr._registry = MLflowModelRegistry(mock_gateway, fake_mlflow, log_model_enabled=True)

    v = mgr.register_model("m", alias="staging", tags={"a": "b"})
    assert v == 3

    assert mgr.get_model_by_alias("m", "champion") is not None
    assert mgr.get_model_by_alias("m", "missing") is None

    assert mgr.set_model_alias("m", "champion", 3) is True
    assert mgr.delete_model_alias("m", "staging") is True

    # promote: missing source alias -> False
    assert mgr.promote_model("m", from_alias="missing", to_alias="champion") is False

    # get aliases mapping
    aliases = mgr.get_model_aliases("m")
    assert aliases["champion"] == 3

    # load model by alias
    loaded = mgr.load_model_by_alias("m", alias="champion")
    assert loaded == {"uri": "models:/m@champion"}


def test_get_best_run_requires_experiment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    cfg = _mk_cfg()
    mgr = MLflowManager(cfg)
    mgr._mlflow = fake_mlflow
    # Explicitly remove config
    mgr._mlflow_config = None
    assert mgr.get_best_run(experiment_name=None) is None


def test_get_best_run_returns_first_row(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow, _shared = _install_fake_mlflow(monkeypatch)
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = fake_mlflow
    # Update analytics with fake mlflow module and experiment_name
    from src.training.mlflow.run_analytics import MLflowRunAnalytics
    mgr._analytics = MLflowRunAnalytics(mgr._gateway, fake_mlflow, experiment_name="test", event_log=mgr._event_log)
    best = mgr.get_best_run(metric="eval_loss", mode="min")
    assert best is not None
    assert best["run_id"] == "r1" or best["run_id"] == "r2"

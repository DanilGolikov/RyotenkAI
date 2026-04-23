from __future__ import annotations

import contextlib
import sys
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
    MLflowConfig,
    ModelConfig,
    QLoRAConfig,
    PipelineConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        torch_dtype="bfloat16",
        trust_remote_code=False,
    )


def _lora_cfg() -> QLoRAConfig:
    return QLoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )


def _hp_cfg() -> GlobalHyperparametersConfig:
    return GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )


def _inference_cfg_disabled() -> InferenceConfig:
    # Even when inference.enabled=False, schema requires engine images to exist in config.
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


def _mk_cfg(
    *,
    tracking_uri: str = "http://127.0.0.1:5002",
    local_tracking_uri: str | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        model=_model_cfg(),
        training=TrainingOnlyConfig(
            type="qlora",
            qlora=_lora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
            )
        },
        inference=_inference_cfg_disabled(),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                tracking_uri=tracking_uri,
                local_tracking_uri=local_tracking_uri,
                experiment_name="test",
            )
        ),
    )


def test_is_active_reflects_runtime_setup() -> None:
    mgr = MLflowManager(_mk_cfg())
    assert mgr.is_active is False

    mgr._mlflow = object()
    assert mgr.is_active is True


def test_resolve_runtime_tracking_uri_control_plane_prefers_local_tracking_uri() -> None:
    from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris

    cfg = _mk_cfg(tracking_uri="https://public.example", local_tracking_uri="http://localhost:5002")
    resolved = resolve_mlflow_uris(cfg.experiment_tracking.mlflow, runtime_role="control_plane")
    assert resolved.effective_local_tracking_uri == "http://localhost:5002"
    assert resolved.effective_remote_tracking_uri == "https://public.example"
    assert resolved.runtime_tracking_uri == "http://localhost:5002"


def test_resolve_runtime_tracking_uri_training_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://env.example")
    cfg = _mk_cfg(tracking_uri="https://public.example", local_tracking_uri="http://localhost:5002")
    resolved = resolve_mlflow_uris(cfg.experiment_tracking.mlflow, runtime_role="training")
    assert resolved.effective_local_tracking_uri == "http://localhost:5002"
    assert resolved.effective_remote_tracking_uri == "https://public.example"
    assert resolved.runtime_tracking_uri == "https://env.example"


def test_check_connectivity_http_error_4xx_counts_as_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request

    from src.infrastructure.mlflow.gateway import MLflowGateway

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(url="x", code=404, msg="not found", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    gw = MLflowGateway("http://x")
    assert gw.check_connectivity(timeout=0.01) is True


def test_check_connectivity_exception_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from src.infrastructure.mlflow.gateway import MLflowGateway

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    gw = MLflowGateway("http://x")
    assert gw.check_connectivity(timeout=0.01) is False
    assert gw.last_connectivity_error is not None
    assert gw.last_connectivity_error.code == "MLFLOW_PREFLIGHT_CONNECTION_FAILED"


def test_client_property_uses_tracking_uri_from_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """client property on MLflowManager delegates to gateway.get_client()."""
    from src.infrastructure.mlflow.gateway import MLflowGateway

    created: dict[str, Any] = {}

    class FakeMlflowClient:
        def __init__(self, tracking_uri: str):
            created["tracking_uri"] = tracking_uri

    # Patch mlflow.MlflowClient inside the gateway module
    monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.get_client",
                        lambda self: FakeMlflowClient(self._uri))

    mgr = MLflowManager(_mk_cfg(tracking_uri="http://127.0.0.1:5002"))
    mgr._mlflow = object()  # mark as initialized
    # _gateway is NullMLflowGateway initially; replace with a real one
    mgr._gateway = MLflowGateway("http://127.0.0.1:5002")

    _ = mgr.client
    assert created["tracking_uri"] == "http://127.0.0.1:5002"


@dataclass
class FakeRun:
    run_id: str

    @property
    def info(self):
        return SimpleNamespace(run_id=self.run_id)


class FakeMLflow:
    def __init__(self):
        self.tags_calls: list[dict[str, str]] = []
        self.params_calls: list[dict[str, str]] = []
        self.metric_calls: list[tuple[str, float, int | None]] = []
        self.started: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def start_run(self, **kwargs):
        self.started.append(kwargs)
        yield FakeRun(run_id=kwargs.get("run_name", "run"))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags_calls.append(tags)

    def log_params(self, params: dict[str, str]) -> None:
        self.params_calls.append(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metric_calls.append((key, value, step))

    def search_runs(self, **kwargs):
        return SimpleNamespace(empty=True)


def test_start_run_without_setup_yields_none() -> None:
    mgr = MLflowManager(_mk_cfg())
    with mgr.start_run(run_name="x") as run:
        assert run is None


def test_start_run_sets_parent_and_run_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = FakeMLflow()

    # avoid filesystem in _load_description_file
    monkeypatch.setattr(mgr, "_load_description_file", lambda: None)

    with mgr.start_run(run_name="parent") as run:
        assert run is not None
        assert mgr.parent_run_id == "parent"
        assert mgr.run_id == "parent"

    # parent_run_id is cleared on exit
    assert mgr.parent_run_id is None


def test_start_nested_run_without_parent_falls_back_to_start_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = FakeMLflow()
    monkeypatch.setattr(mgr, "_load_description_file", lambda: None)

    with mgr.start_nested_run("child", tags={"k": "v"}) as run:
        assert run is not None

    ml: FakeMLflow = mgr._mlflow
    assert ml.started and ml.started[0]["run_name"] == "child"
    assert ml.tags_calls and ml.tags_calls[0] == {"k": "v"}


def test_start_nested_run_pushes_and_restores_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())
    ml = FakeMLflow()
    mgr._mlflow = ml
    monkeypatch.setattr(mgr, "_load_description_file", lambda: None)

    with mgr.start_run(run_name="parent"):
        assert mgr.parent_run_id == "parent"
        with mgr.start_nested_run("child", tags={"t": "1"}):
            assert mgr.run_id == "child"
            assert mgr.is_nested is True
        assert mgr.run_id == "parent"
        assert mgr.is_nested is False

    # auto-tags include parentRunId and depth
    assert any("mlflow.parentRunId" in call for call in ml.tags_calls)


def test_log_params_filters_and_stringifies(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = FakeMLflow()
    mgr._run = object()

    mgr.log_params({"a": 1, "b": None})
    ml: FakeMLflow = mgr._mlflow
    assert ml.params_calls[-1] == {"a": "1", "b": "None"}


def test_log_metrics_calls_log_metric_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = FakeMLflow()
    mgr._run = object()

    mgr.log_metrics({"loss": 0.5, "maybe": None}, step=10)  # type: ignore[arg-type]
    ml: FakeMLflow = mgr._mlflow
    assert ("loss", 0.5, 10) in ml.metric_calls


def test_get_child_runs_returns_empty_when_search_empty() -> None:
    mgr = MLflowManager(_mk_cfg())
    mgr._mlflow = FakeMLflow()
    mgr._parent_run_id = "parent"
    assert mgr.get_child_runs() == []


def test_delete_run_tree_deletes_descendants_before_root() -> None:
    mgr = MLflowManager(_mk_cfg())
    deleted: list[str] = []

    class _Run:
        def __init__(self, run_id: str, experiment_id: str = "exp_1") -> None:
            self.info = SimpleNamespace(run_id=run_id, experiment_id=experiment_id)

    class _Client:
        def get_run(self, run_id: str):
            return _Run(run_id)

        def search_runs(self, *, experiment_ids, filter_string):
            _ = experiment_ids
            if filter_string.endswith("'root'"):
                return [_Run("child_a"), _Run("child_b")]
            if filter_string.endswith("'child_a'"):
                return [_Run("grandchild")]
            return []

        def delete_run(self, run_id: str) -> None:
            deleted.append(run_id)

    mgr._mlflow = object()
    mgr._gateway = SimpleNamespace(get_client=lambda: _Client())

    deleted_ids = mgr.delete_run_tree("root")

    assert deleted_ids == ["grandchild", "child_b", "child_a", "root"]
    assert deleted == ["grandchild", "child_b", "child_a", "root"]


def test_log_dataset_info_builds_params_and_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())
    captured = {"params": None, "tags": None, "dict": None}

    monkeypatch.setattr(mgr, "log_params", lambda p: captured.__setitem__("params", p))
    monkeypatch.setattr(mgr, "set_tags", lambda t: captured.__setitem__("tags", t))
    monkeypatch.setattr(mgr, "log_dict", lambda d, name, run_id=None: captured.__setitem__("dict", (d, name, run_id)))

    # Build dataset_logger manually using the facade as primitives provider
    from src.training.mlflow.dataset_logger import MLflowDatasetLogger
    mgr._dataset_logger = MLflowDatasetLogger(
        mlflow_module=None,
        primitives=mgr,  # type: ignore[arg-type]
        has_active_run=lambda: True,
    )

    mgr.log_dataset_info(
        name="ds",
        path="/data",
        source="local",
        version="v1",
        num_rows=10,
        num_features=2,
        context="training",
        extra_info={"x": 1},
        extra_tags={"a": "b"},
    )

    assert captured["params"]["dataset_training_name"] == "ds"
    assert captured["params"]["dataset_training_samples"] == 10
    assert captured["params"]["dataset_training_path"] == "/data"
    assert captured["params"]["dataset_training_source"] == "local"
    assert captured["params"]["dataset_training_version"] == "v1"
    assert captured["params"]["dataset_training_features"] == 2
    assert captured["tags"] == {"a": "b"}
    assert captured["dict"][1] == "dataset_training_info.json"


def test_create_mlflow_dataset_from_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = MLflowManager(_mk_cfg())

    created: dict[str, Any] = {}

    fake_mlflow = SimpleNamespace(
        data=SimpleNamespace(
            from_pandas=lambda df, source, name, targets=None: created.update(
                {"rows": len(df), "source": source, "name": name, "targets": targets}
            )
            or {"ok": True}
        )
    )
    mgr._mlflow = fake_mlflow

    # Build dataset_logger manually using the fake mlflow module
    from src.training.mlflow.dataset_logger import MLflowDatasetLogger
    mgr._dataset_logger = MLflowDatasetLogger(
        mlflow_module=fake_mlflow,
        primitives=mgr,  # type: ignore[arg-type]
        has_active_run=lambda: True,
    )

    df = pd.DataFrame([{"a": 1}, {"a": 2}])
    ds = mgr.create_mlflow_dataset(df, name="ds", source="local://x", targets="y")
    assert ds == {"ok": True}
    assert created == {"rows": 2, "source": "local://x", "name": "ds", "targets": "y"}


def test_event_logging_sets_has_errors_and_severity() -> None:
    mgr = MLflowManager(_mk_cfg())
    ev = mgr.log_event_error("boom", category="system", source="MLflowManager", code=1)
    assert ev["event_type"] == "error"
    assert ev["severity"] == "ERROR"
    assert mgr.get_events(category="system")
    assert any(e["event_type"] == "error" for e in mgr.get_events())


def test_log_dataset_config_single_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_dataset_config with single dataset."""
    config = PipelineConfig(
        model=_model_cfg(),
        training=TrainingOnlyConfig(
            type="qlora",
            qlora=_lora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft", dataset="default")],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(
                    local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None),
                ),
            )
        },
        inference=_inference_cfg_disabled(),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5002",
                experiment_name="test",
            )
        ),
    )

    mgr = MLflowManager(config)
    captured = {"params": None, "tags": None}

    monkeypatch.setattr(mgr, "log_params", lambda p: captured.__setitem__("params", p))
    monkeypatch.setattr(mgr, "set_tags", lambda t: captured.__setitem__("tags", t))

    mgr.log_dataset_config(config)

    # Check params
    assert captured["params"] is not None
    assert captured["params"]["dataset.default.name"] == "train"
    assert captured["params"]["dataset.default.source_type"] == "local"
    assert captured["params"]["dataset.default.adapter_type"] == "auto"
    assert captured["params"]["dataset.default.local.train_path"] == "data/train.jsonl"

    # Check tags
    assert captured["tags"] is not None
    assert captured["tags"]["dataset.names"] == "default"
    assert captured["tags"]["dataset.count"] == "1"


def test_log_dataset_config_multiple_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_dataset_config with multiple datasets (CPT + SFT)."""
    config = PipelineConfig(
        model=_model_cfg(),
        training=TrainingOnlyConfig(
            type="qlora",
            qlora=_lora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[
                StrategyPhaseConfig(strategy_type="cpt", dataset="corpus"),
                StrategyPhaseConfig(strategy_type="sft", dataset="sft_data"),
                StrategyPhaseConfig(strategy_type="cot", dataset="cot_data"),
            ],
        ),
        datasets={
            "sft_data": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(
                    local_paths=DatasetLocalPaths(train="data/sft.jsonl", eval=None),
                ),
            ),
            "cot_data": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(
                    local_paths=DatasetLocalPaths(train="data/cot.jsonl", eval=None),
                ),
            ),
            "corpus": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(
                    local_paths=DatasetLocalPaths(train="data/corpus.jsonl", eval=None),
                ),
                max_samples=10000,
            ),
        },
        inference=_inference_cfg_disabled(),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5002",
                experiment_name="test",
            )
        ),
    )

    mgr = MLflowManager(config)
    captured = {"params": None, "tags": None}

    monkeypatch.setattr(mgr, "log_params", lambda p: captured.__setitem__("params", p))
    monkeypatch.setattr(mgr, "set_tags", lambda t: captured.__setitem__("tags", t))

    mgr.log_dataset_config(config)

    # Check params for both datasets
    assert captured["params"] is not None

    # corpus dataset
    assert captured["params"]["dataset.corpus.name"] == "corpus"
    assert captured["params"]["dataset.corpus.source_type"] == "local"
    assert captured["params"]["dataset.corpus.local.train_path"] == "data/corpus.jsonl"
    assert captured["params"]["dataset.corpus.max_samples"] == "10000"

    # sft_data dataset (display_name from file stem: sft.jsonl → sft)
    assert captured["params"]["dataset.sft_data.name"] == "sft"
    assert captured["params"]["dataset.sft_data.source_type"] == "local"
    assert captured["params"]["dataset.sft_data.local.train_path"] == "data/sft.jsonl"

    # cot_data dataset (display_name from file stem: cot.jsonl → cot)
    assert captured["params"]["dataset.cot_data.name"] == "cot"
    assert captured["params"]["dataset.cot_data.source_type"] == "local"
    assert captured["params"]["dataset.cot_data.local.train_path"] == "data/cot.jsonl"

    # Check tags (sorted alphabetically: corpus, cot_data, sft_data)
    assert captured["tags"] is not None
    assert captured["tags"]["dataset.names"] == "corpus,cot_data,sft_data"
    assert captured["tags"]["dataset.count"] == "3"

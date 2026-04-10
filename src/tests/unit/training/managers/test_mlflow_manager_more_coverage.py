from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.training.managers.mlflow_manager import MLflowManager
from src.training.mlflow.resilient_transport import ResilientMLflowTransport
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
    PipelineConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)

pytestmark = pytest.mark.unit


def _mk_cfg(
    *,
    tracking_uri: str = "http://127.0.0.1:5002",
    local_tracking_uri: str | None = None,
    system_metrics_callback_enabled: bool = False,
    run_description_file: str | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="test-model", torch_dtype="bfloat16", trust_remote_code=False),
        training=TrainingOnlyConfig(
            type="qlora",
            qlora=LoraConfig(
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
                local_tracking_uri=local_tracking_uri,
                experiment_name="test",
                system_metrics_callback_enabled=system_metrics_callback_enabled,
                run_description_file=run_description_file,
            )
        ),
    )


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake = ModuleType("mlflow")
    fake.get_tracking_uri = lambda: "http://127.0.0.1:5002"  # type: ignore[attr-defined]
    fake.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined, ARG005]
    fake.set_experiment = lambda name: None  # type: ignore[attr-defined, ARG005]
    fake.enable_system_metrics_logging = lambda: None  # type: ignore[attr-defined]
    fake.set_system_metrics_sampling_interval = lambda n: None  # type: ignore[attr-defined, ARG005]
    fake.set_system_metrics_samples_before_logging = lambda n: None  # type: ignore[attr-defined, ARG005]
    fake.autolog = lambda **kw: None  # type: ignore[attr-defined, ARG005]
    fake.end_run = lambda **kw: None  # type: ignore[attr-defined, ARG005]
    fake.log_metric = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.log_metrics = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.log_param = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.log_params = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.set_tag = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.set_tags = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.log_dict = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]
    fake.log_text = lambda *a, **k: None  # type: ignore[attr-defined, ARG005]

    class FakeMlflowClient:
        def log_batch(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def log_metric(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def log_param(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def log_params(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def set_tag(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def set_tags(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def log_dict(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

        def log_text(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

    fake.MlflowClient = FakeMlflowClient  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


class TestSmallBranchesAndSetup:
    def test_client_property_returns_none_when_mlflow_not_initialized(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        assert mgr.client is None

    def test_get_active_run_id_prefers_explicit(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._run_id = "rid"
        assert mgr._get_active_run_id("explicit") == "explicit"

    def test_setup_import_error_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = MLflowManager(_mk_cfg())

        orig_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        assert mgr.setup() is False

    def test_setup_set_experiment_failure_disables_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg())
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005

        # Simulate the real-world failure when the experiment is soft-deleted in MLflow backend.
        fake.set_experiment = lambda name: (_ for _ in ()).throw(  # type: ignore[attr-defined]
            RuntimeError("Cannot set a deleted experiment 'x' as the active experiment.")
        )

        assert mgr.setup() is False
        assert mgr.is_active is False

    def test_setup_system_metrics_enabled_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg(system_metrics_callback_enabled=True))
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005
        assert mgr.setup(disable_system_metrics=False) is True

    def test_setup_system_metrics_enable_failure_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)

        # force enable_system_metrics_logging to fail to hit exception branch
        fake.enable_system_metrics_logging = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]

        mgr = MLflowManager(_mk_cfg(system_metrics_callback_enabled=True))
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005
        assert mgr.setup(disable_system_metrics=False) is True

    def test_setup_sets_requests_ca_bundle_env_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        cfg = _mk_cfg()
        cfg.experiment_tracking.mlflow.ca_bundle_path = "certs/mlflow-ca.pem"
        mgr = MLflowManager(cfg)
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        assert mgr.setup(disable_system_metrics=True) is True
        assert mgr.get_runtime_tracking_uri() == "http://127.0.0.1:5002"
        assert mgr.get_effective_remote_tracking_uri() == "http://127.0.0.1:5002"
        assert os.environ["REQUESTS_CA_BUNDLE"] == "certs/mlflow-ca.pem"
        assert os.environ["SSL_CERT_FILE"] == "certs/mlflow-ca.pem"


class TestResilientTransportShim:
    def test_setup_installs_shim_only_for_training_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        original = fake.log_metrics
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005

        control_plane_mgr = MLflowManager(_mk_cfg(), runtime_role="control_plane")
        assert control_plane_mgr.setup(disable_system_metrics=True) is True
        assert fake.log_metrics is original
        control_plane_mgr.cleanup()

        training_mgr = MLflowManager(_mk_cfg(), runtime_role="training")
        training_mgr._resilient_transport = ResilientMLflowTransport(warning_interval_s=0.0)
        assert training_mgr.setup(disable_system_metrics=True) is True
        assert fake.log_metrics is not original
        training_mgr.cleanup()
        assert fake.log_metrics is original

    def test_transport_failures_open_breaker_and_skip_repeated_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005

        calls = {"count": 0}

        def failing_log_metrics(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
            calls["count"] += 1
            raise RuntimeError(
                "API request to https://mlflow.example/api/2.0/mlflow/runs/log-batch "
                "failed with exception HTTPSConnectionPool(host='mlflow.example', port=443): "
                "Max retries exceeded with url: /api/2.0/mlflow/runs/log-batch "
                "(Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred "
                "in violation of protocol (_ssl.c:1010)')))"
            )

        fake.log_metrics = failing_log_metrics  # type: ignore[attr-defined]

        mgr = MLflowManager(_mk_cfg(), runtime_role="training")
        mgr._resilient_transport = ResilientMLflowTransport(
            failure_threshold=2,
            recovery_cooldown_s=999.0,
            warning_interval_s=0.0,
        )
        assert mgr.setup(disable_system_metrics=True) is True

        fake.log_metrics({"loss": 1.0}, step=1)  # type: ignore[attr-defined]
        fake.log_metrics({"loss": 1.0}, step=2)  # type: ignore[attr-defined]
        fake.log_metrics({"loss": 1.0}, step=3)  # type: ignore[attr-defined]

        assert calls["count"] == 2
        assert mgr._resilient_transport.breaker_state == "open"
        mgr.cleanup()

    def test_half_open_recovery_closes_breaker_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005

        calls = {"count": 0}

        def flaky_log_metric(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError(
                    "API request to https://mlflow.example/api/2.0/mlflow/runs/log-metric "
                    "failed with exception HTTPSConnectionPool(host='mlflow.example', port=443): "
                    "Max retries exceeded with url: /api/2.0/mlflow/runs/log-metric"
                )

        fake.log_metric = flaky_log_metric  # type: ignore[attr-defined]

        mgr = MLflowManager(_mk_cfg(), runtime_role="training")
        mgr._resilient_transport = ResilientMLflowTransport(
            failure_threshold=1,
            recovery_cooldown_s=0.0,
            warning_interval_s=0.0,
        )
        assert mgr.setup(disable_system_metrics=True) is True

        fake.log_metric("loss", 1.0, step=1)  # type: ignore[attr-defined]
        assert mgr._resilient_transport.breaker_state == "open"

        fake.log_metric("loss", 0.5, step=2)  # type: ignore[attr-defined]
        assert calls["count"] == 2
        assert mgr._resilient_transport.breaker_state == "closed"
        mgr.cleanup()

    def test_non_transport_exception_is_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setattr("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", lambda self, timeout: True)  # noqa: ARG005

        def bad_log_metric(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
            raise ValueError("bad metric payload")

        fake.log_metric = bad_log_metric  # type: ignore[attr-defined]

        mgr = MLflowManager(_mk_cfg(), runtime_role="training")
        mgr._resilient_transport = ResilientMLflowTransport(warning_interval_s=0.0)
        assert mgr.setup(disable_system_metrics=True) is True

        with pytest.raises(ValueError, match="bad metric payload"):
            fake.log_metric("loss", 1.0, step=1)  # type: ignore[attr-defined]
        mgr.cleanup()


class TestAutologFallbackAndDisable:
    def test_enable_autolog_fallback_to_generic_and_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = fake

        called: list[dict[str, Any]] = []
        fake.autolog = lambda **kw: called.append(kw)  # type: ignore[attr-defined]

        # No mlflow.transformers module -> fallback to generic autolog
        assert mgr.enable_autolog(log_models=True) is True
        assert called and called[-1]["log_models"] is True

        # Failure branch
        fake.autolog = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
        assert mgr.enable_autolog() is False

    def test_disable_autolog_fallback_and_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = fake

        called: list[dict[str, Any]] = []
        fake.autolog = lambda **kw: called.append(kw)  # type: ignore[attr-defined]
        assert mgr.disable_autolog() is True
        assert called and called[-1]["disable"] is True

        fake.autolog = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
        assert mgr.disable_autolog() is False

    def test_disable_autolog_returns_false_when_mlflow_none(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        assert mgr.disable_autolog() is False


class TestTracingDecoratorAndUrls:
    def test_trace_llm_call_yields_none_when_unavailable(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None

        with mgr.trace_llm_call("x") as span:
            assert span is None

    def test_trace_llm_call_attribute_error_is_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = SimpleNamespace()

        with mgr.trace_llm_call("x") as span:
            assert span is None

    def test_trace_llm_call_exception_is_handled(self) -> None:
        mgr = MLflowManager(_mk_cfg())

        class _BadMLflow:
            def start_span(self, **kwargs):
                raise RuntimeError("boom")

        mgr._mlflow = _BadMLflow()
        with mgr.trace_llm_call("x") as span:
            assert span is None

    def test_create_trace_decorator_noop_and_attribute_error(self) -> None:
        mgr = MLflowManager(_mk_cfg())

        mgr._mlflow = None
        deco = mgr.create_trace_decorator(name="x")

        def f():
            return 1

        assert deco(f) is f

        # AttributeError branch: no .trace on mlflow object
        mgr._mlflow = SimpleNamespace()
        deco = mgr.create_trace_decorator(name="x")
        assert deco(f) is f

    def test_get_trace_url_branches(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        assert mgr.get_trace_url("t") is None  # no mlflow set up

        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: SimpleNamespace(trace_id="tid"))
        assert mgr.get_trace_url(None) == "http://127.0.0.1:5002/#/traces/tid"

        # no tracking URI -> None
        mgr = MLflowManager(_mk_cfg())
        mgr._gateway = SimpleNamespace(uri="")
        mgr.get_runtime_tracking_uri = lambda: ""  # type: ignore[method-assign]
        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: None)
        assert mgr.get_trace_url(None) is None

        # no current span -> None
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: None)
        assert mgr.get_trace_url(None) is None

        # exception -> None
        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert mgr.get_trace_url(None) is None

    def test_log_trace_io_branches(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        mgr.log_trace_io(input_data="x")  # early return

        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: None)
        mgr.log_trace_io(input_data="x")  # span None -> return

        class _Span:
            def __init__(self):
                self.attrs = None

            def set_inputs(self, d):
                _ = d

            def set_outputs(self, d):
                _ = d

            def set_attributes(self, d):
                self.attrs = d

        span = _Span()
        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: span)
        mgr.log_trace_io(input_data="x", output_data="y", metadata={"k": "v"})
        assert span.attrs and span.attrs["k"] == "v"

        # AttributeError branch: span lacks set_inputs
        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: SimpleNamespace())
        mgr.log_trace_io(input_data="x")

        # Exception branch: span.set_inputs raises
        class _BadSpan:
            def set_inputs(self, d):
                raise RuntimeError("boom")

        mgr._mlflow = SimpleNamespace(get_current_active_span=lambda: _BadSpan())
        mgr.log_trace_io(input_data="x")


class TestDescriptionAndConnectivityAndRuns:
    def test_load_description_file_custom_and_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        desc = tmp_path / "desc.md"
        desc.write_text("hi", encoding="utf-8")

        mgr = MLflowManager(_mk_cfg(run_description_file=str(desc)))
        assert mgr._load_description_file() == "hi"

        # Force both custom and default template to be treated as missing -> return None branch
        mgr = MLflowManager(_mk_cfg(run_description_file=str(tmp_path / "missing.md")))
        orig_exists = Path.exists

        def exists(p: Path) -> bool:
            # hide both the custom and the default template
            if str(p).endswith("missing.md") or str(p).endswith("experiment_description.md"):
                return False
            return orig_exists(p)

        monkeypatch.setattr("pathlib.Path.exists", exists)
        assert mgr._load_description_file() is None

    def test_resolve_runtime_tracking_uri_remote_falls_back_to_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        resolved = resolve_mlflow_uris(
            _mk_cfg(tracking_uri="", local_tracking_uri="http://localhost:5002").experiment_tracking.mlflow,
            runtime_role="training",
        )
        assert resolved.effective_local_tracking_uri == "http://localhost:5002"
        assert resolved.effective_remote_tracking_uri == "http://localhost:5002"
        assert resolved.runtime_tracking_uri == "http://localhost:5002"

    def test_check_connectivity_success_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        from src.infrastructure.mlflow.gateway import MLflowGateway

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())

        gw = MLflowGateway("http://x")
        assert gw.check_connectivity(timeout=0.01) is True

    def test_check_connectivity_https_uses_custom_ca_bundle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ssl
        import urllib.request

        from src.infrastructure.mlflow.gateway import MLflowGateway

        captured: dict[str, Any] = {}

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_create_default_context(*, cafile=None, **kwargs):
            captured["cafile"] = cafile
            return object()

        def fake_urlopen(req, timeout, context):
            captured["context"] = context
            return _Resp()

        monkeypatch.setattr(ssl, "create_default_context", fake_create_default_context)
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        gw = MLflowGateway("https://mlflow.example", ca_bundle_path="certs/mlflow-ca.pem")
        assert gw.check_connectivity(timeout=0.01) is True
        assert captured["cafile"] == "certs/mlflow-ca.pem"
        assert captured["context"] is not None
        assert gw.last_connectivity_error is None

    def test_check_connectivity_ca_bundle_missing_sets_structured_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ssl

        from src.infrastructure.mlflow.gateway import MLflowGateway

        def fake_create_default_context(*, cafile=None, **kwargs):
            raise FileNotFoundError(cafile)

        monkeypatch.setattr(ssl, "create_default_context", fake_create_default_context)

        gw = MLflowGateway("https://mlflow.example", ca_bundle_path="missing-ca.pem")
        assert gw.check_connectivity(timeout=0.01) is False
        assert gw.last_connectivity_error is not None
        assert gw.last_connectivity_error.code == "MLFLOW_TLS_CA_BUNDLE_INVALID"

    def test_start_run_exception_path_yields_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = fake

        # Make start_run raise
        fake.start_run = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
        monkeypatch.setattr(mgr, "_load_description_file", lambda: None)

        with mgr.start_run(run_name="x") as run:
            assert run is None

    def test_end_run_no_mlflow_and_exception(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr.end_run(status="FAILED")  # should not raise when _mlflow is None

        mgr._mlflow = SimpleNamespace(end_run=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        mgr.end_run(status="FAILED")

    def test_start_nested_run_yields_none_when_not_setup_and_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = MLflowManager(_mk_cfg())
        with mgr.start_nested_run("x") as run:
            assert run is None

        fake = _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = fake
        mgr._parent_run_id = "parent"

        fake.start_run = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
        with mgr.start_nested_run("child") as run:
            assert run is None

    def test_get_child_runs_and_log_params_errors(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        assert mgr.get_child_runs() == []

        mgr._mlflow = SimpleNamespace(search_runs=lambda **kw: [])
        mgr._parent_run_id = None
        assert mgr.get_child_runs() == []

        mgr._parent_run_id = "p"
        mgr._mlflow = SimpleNamespace(search_runs=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        assert mgr.get_child_runs() == []

        # log_params exception branch
        mgr._run = object()
        mgr._mlflow = SimpleNamespace(log_params=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        mgr.log_params({"a": 1})

        # log_metrics early return (no run)
        mgr._run = None
        mgr.log_metrics({"m": 1.0})


class TestDatasetsAndTags:
    def test_log_dataset_dict_unsupported_and_exception(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._run = object()

        class _DataAPI:
            def __init__(self):
                self.from_pandas = MagicMock(return_value="ds")

        mlflow_obj = SimpleNamespace(
            data=_DataAPI(),
            log_input=MagicMock(),
        )
        mgr._mlflow = mlflow_obj

        assert mgr.log_dataset(data={"a": [1]}, name="n", source="s") is True
        assert mgr.log_dataset(data=123, name="n", source="s") is False

        mlflow_obj.log_input.side_effect = RuntimeError("boom")
        assert mgr.log_dataset(data={"a": [1]}, name="n", source="s") is False

    def test_log_dataset_from_file_branches(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = SimpleNamespace()
        mgr._run = object()

        # missing file
        assert mgr.log_dataset_from_file(str(tmp_path / "missing.jsonl")) is False

        # json branch (not jsonl)
        f_json = tmp_path / "d.json"
        f_json.write_text("{}", encoding="utf-8")

        import pandas as pd

        monkeypatch.setattr(pd, "read_json", lambda *a, **k: pd.DataFrame({"x": [1]}))
        mgr._dataset_logger.log_dataset = MagicMock(return_value=True)  # type: ignore[method-assign]
        assert mgr.log_dataset_from_file(str(f_json)) is True

        # parquet branch (patched to avoid pyarrow dependency)
        f_parquet = tmp_path / "d.parquet"
        f_parquet.write_text("x", encoding="utf-8")
        monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame({"x": [1]}))
        assert mgr.log_dataset_from_file(str(f_parquet)) is True

        # unsupported suffix
        f_txt = tmp_path / "d.txt"
        f_txt.write_text("x", encoding="utf-8")
        assert mgr.log_dataset_from_file(str(f_txt)) is False

        # exception branch
        monkeypatch.setattr(pd, "read_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert mgr.log_dataset_from_file(str(f_json)) is False

        # early return when run missing
        mgr._run = None
        assert mgr.log_dataset_from_file(str(f_json)) is False

    def test_create_mlflow_dataset_and_log_dataset_input_and_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = MLflowManager(_mk_cfg())

        # create_mlflow_dataset returns None when mlflow missing
        assert mgr.create_mlflow_dataset(data=[{"x": 1}], name="n", source="s") is None

        import pandas as pd

        class _DataAPI:
            def __init__(self):
                self.from_pandas = MagicMock(return_value="ds")

        mlflow_obj = SimpleNamespace(data=_DataAPI(), log_input=MagicMock(), set_tags=MagicMock())
        mgr._mlflow = mlflow_obj
        mgr._run = object()

        # to_pandas branch
        class _HF:
            def to_pandas(self):
                return pd.DataFrame({"x": [1]})

        assert mgr.create_mlflow_dataset(data=_HF(), name="n", source="s") == "ds"

        # DataFrame constructor branch
        assert mgr.create_mlflow_dataset(data=[{"x": 1}], name="n", source="s") == "ds"

        # exception branch
        mlflow_obj.data.from_pandas.side_effect = RuntimeError("boom")
        assert mgr.create_mlflow_dataset(data=[{"x": 1}], name="n", source="s") is None

        # log_dataset_input
        mlflow_obj.data.from_pandas.side_effect = None
        assert mgr.log_dataset_input(dataset=None) is False

        mlflow_obj.log_input.side_effect = RuntimeError("boom")
        assert mgr.log_dataset_input(dataset="ds") is False

        # set_tags early return + exception
        mgr2 = MLflowManager(_mk_cfg())
        mgr2.set_tags({"a": "1"})  # no-op

        mgr._run = object()
        mlflow_obj.set_tags.side_effect = RuntimeError("boom")
        mgr.set_tags({"a": "1"})
        mgr.set_tag("k", "v")


class TestLlmEvaluation:
    def test_log_llm_evaluation_no_mlflow_is_noop(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        mgr.log_llm_evaluation(prompt="p", response="r")

    def test_log_llm_evaluation_logs_metric_and_handles_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = fake
        mgr._run_id = "run_12345678"

        mgr.log_dict = MagicMock(return_value=True)  # type: ignore[method-assign]
        fake.log_metric = MagicMock()  # type: ignore[attr-defined]

        mgr.log_llm_evaluation(prompt="p", response="r", expected="e", feedback="fb", score=0.5, evaluator="human")
        fake.log_metric.assert_called_once()

        # exception path: log_dict raises -> should be swallowed
        mgr.log_dict = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        mgr.log_llm_evaluation(prompt="p", response="r", score=0.5, evaluator="human")


class TestFinalSmallCoveragePush:
    def test_tracing_enable_disable_branches(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        assert mgr.enable_pytorch_autolog() is False
        assert mgr.enable_tracing() is False
        assert mgr.disable_tracing() is False

        # enable_tracing exception path
        mgr._mlflow = SimpleNamespace(
            tracing=SimpleNamespace(enable=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        )
        assert mgr.enable_tracing() is False

        # disable_tracing AttributeError + Exception paths
        mgr._mlflow = SimpleNamespace()
        assert mgr.disable_tracing() is False
        mgr._mlflow = SimpleNamespace(
            tracing=SimpleNamespace(disable=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        )
        assert mgr.disable_tracing() is False

    def test_get_child_runs_non_dataframe_returns_empty(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = SimpleNamespace(search_runs=lambda **kw: object())
        mgr._parent_run_id = "p"
        assert mgr.get_child_runs() == []

    def test_log_artifact_and_log_dict_early_returns_and_exceptions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # log_artifact: missing file -> False
        mgr = MLflowManager(_mk_cfg())
        assert mgr.log_artifact("x.txt") is False

        # log_artifact: no run_id / no client -> False
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        mgr._run_id = None
        assert mgr.log_artifact("x.txt") is False

        # Install fake mlflow with MlflowClient that raises on log_text/log_dict
        fake_mlflow = ModuleType("mlflow")
        fake_mlflow.get_tracking_uri = lambda: "http://127.0.0.1:5002"  # type: ignore[attr-defined]

        class _Client:
            def __init__(self, tracking_uri: str):
                pass

            def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
                raise RuntimeError("boom")

            def log_dict(self, run_id: str, d: dict, artifact_file: str) -> None:
                raise RuntimeError("boom")

        fake_mlflow.MlflowClient = _Client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = fake_mlflow
        mgr._run_id = "run_1"

        f = tmp_path / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert mgr.log_artifact(str(f)) is False

        # log_dict: no active run -> False
        mgr2 = MLflowManager(_mk_cfg())
        mgr2._mlflow = fake_mlflow
        mgr2._run_id = None
        assert mgr2.log_dict({"x": 1}, "x.json") is False

        # log_dict: exception -> False
        assert mgr.log_dict({"x": 1}, "x.json") is False

    def test_log_dataset_early_return_and_log_dataset_input_success(self) -> None:
        mgr = MLflowManager(_mk_cfg())
        mgr._mlflow = None
        mgr._run = None
        assert mgr.log_dataset(data={"a": [1]}, name="n", source="s") is False

        mgr._mlflow = SimpleNamespace(log_input=MagicMock())
        mgr._run = object()
        assert mgr.log_dataset_input(dataset="ds") is True


class TestBuildSubcomponentsAndSetupWiring:
    """Verify _build_subcomponents() wires all mlflow-dependent attributes after setup()."""

    def test_subcomponents_wired_after_successful_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        monkeypatch.setattr(
            "src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity",
            lambda self, timeout: True,
        )
        mgr = MLflowManager(_mk_cfg())
        assert mgr.setup(disable_system_metrics=True) is True

        # Autolog wired
        assert mgr._autolog._mlflow is fake
        assert mgr._autolog._tracking_uri is not None

        # Analytics wired
        assert mgr._analytics._mlflow is fake
        assert mgr._analytics._experiment_name == "test"
        assert mgr._analytics._gateway is mgr._gateway

        # Registry created
        from src.training.mlflow.model_registry import MLflowModelRegistry
        assert isinstance(mgr._registry, MLflowModelRegistry)

        # Dataset logger recreated with mlflow module
        from src.training.mlflow.dataset_logger import MLflowDatasetLogger
        assert isinstance(mgr._dataset_logger, MLflowDatasetLogger)

        # Domain logger recreated
        from src.training.mlflow.domain_logger import MLflowDomainLogger
        assert isinstance(mgr._domain_logger, MLflowDomainLogger)

    def test_resilient_transport_skipped_for_control_plane(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        original_log_metrics = fake.log_metrics
        monkeypatch.setattr(
            "src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity",
            lambda self, timeout: True,
        )
        mgr = MLflowManager(_mk_cfg(), runtime_role="control_plane")
        assert mgr.setup(disable_system_metrics=True) is True
        # Transport should NOT be installed — methods remain original
        assert fake.log_metrics is original_log_metrics
        mgr.cleanup()

    def test_resilient_transport_skipped_for_local_tracking_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_mlflow(monkeypatch)
        original_log_metrics = fake.log_metrics
        # Use file-based tracking URI (no http prefix)
        mgr = MLflowManager(_mk_cfg(tracking_uri="file:///tmp/mlruns"), runtime_role="training")
        monkeypatch.setattr(
            "src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity",
            lambda self, timeout: True,
        )
        # For file:// URIs, setup doesn't do connectivity check, so it should succeed
        assert mgr.setup(disable_system_metrics=True) is True
        # Transport not installed because URI doesn't start with "http"
        assert fake.log_metrics is original_log_metrics
        mgr.cleanup()

    def test_connectivity_retry_exhaustion_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        monkeypatch.setattr(
            "src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity",
            lambda self, timeout: False,
        )
        mgr = MLflowManager(_mk_cfg())
        assert mgr.setup(max_retries=2, disable_system_metrics=True) is False
        assert mgr.is_active is False

    def test_connectivity_retry_succeeds_on_second_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_mlflow(monkeypatch)
        attempts = {"count": 0}

        def flaky_connectivity(self, timeout):
            attempts["count"] += 1
            return attempts["count"] >= 2  # Fails first, succeeds second

        monkeypatch.setattr(
            "src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity",
            flaky_connectivity,
        )
        mgr = MLflowManager(_mk_cfg())
        assert mgr.setup(max_retries=3, disable_system_metrics=True) is True
        assert attempts["count"] == 2

    def test_uri_resolution_lazy_init(self) -> None:
        mgr = MLflowManager(_mk_cfg(tracking_uri="http://127.0.0.1:5002"))
        assert mgr._resolved_uris is None
        uri = mgr.get_runtime_tracking_uri()
        assert uri == "http://127.0.0.1:5002"
        assert mgr._resolved_uris is not None

    def test_get_effective_uris(self) -> None:
        mgr = MLflowManager(_mk_cfg(tracking_uri="http://127.0.0.1:5002"))
        local = mgr.get_effective_local_tracking_uri()
        remote = mgr.get_effective_remote_tracking_uri()
        assert local  # non-empty
        assert remote  # non-empty

    def test_get_raw_uris(self) -> None:
        mgr = MLflowManager(_mk_cfg(
            tracking_uri="http://mlflow.example:5002",
            local_tracking_uri="http://localhost:5002",
        ))
        assert mgr.get_raw_tracking_uri() == "http://mlflow.example:5002"
        assert mgr.get_raw_local_tracking_uri() == "http://localhost:5002"

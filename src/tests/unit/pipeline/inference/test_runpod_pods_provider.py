"""
Unit tests for RunPodPodInferenceProvider (src/providers/runpod/inference/pods/provider.py).

Coverage:
- deploy(): local path, missing API key, missing SSH key, no volume success,
            with volume success, keep_running=True, volume ensure failure,
            pod ensure failure, stop failure
- undeploy(): no api/pod_id (noop), success, failure
- health_check(): not deployed, pod found (ok), pod get failure
- activate_for_eval(): no api, no pod_id, empty adapter_ref, success, session failure
- deactivate_after_eval(): no api (noop), no session+no pod (noop),
                           no session+pod (delete success/fail),
                           session deactivate success/fail
- _ensure_pod(): existing stopped pod, existing running pod,
                 no pods → create, create returns pod on retry, create fails all
- _stop_pod_if_running(): running → stop, not running → skip, get failure, stop failure
- provider_name / provider_type properties
- get_pipeline_readiness_mode
- collect_startup_logs
- get_capabilities
- get_endpoint_info
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.config.inference.schema import InferenceConfig
from src.config.providers.runpod import RunPodProviderConfig
from src.constants import PROVIDER_RUNPOD
from src.providers.inference.interfaces import EndpointInfo, InferenceArtifactsContext, PipelineReadinessMode
from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider
from src.utils.result import Err, InferenceError, Ok, ProviderError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub API
# ---------------------------------------------------------------------------


@dataclass
class StubApi:
    """Thin stub for RunPodPodsRESTClient, driven by queued results."""

    list_pods_results: list[Any] = field(default_factory=list)
    get_pod_results: list[Any] = field(default_factory=list)
    create_pod_results: list[Any] = field(default_factory=list)
    stop_pod_results: list[Any] = field(default_factory=list)
    start_pod_results: list[Any] = field(default_factory=list)
    delete_pod_results: list[Any] = field(default_factory=list)
    list_network_volumes_results: list[Any] = field(default_factory=list)
    get_network_volume_results: list[Any] = field(default_factory=list)
    create_network_volume_results: list[Any] = field(default_factory=list)

    created_pod_payloads: list[dict[str, Any]] = field(default_factory=list)

    def list_pods(self, *, params: dict[str, Any] | None = None) -> Any:
        if self.list_pods_results:
            return self.list_pods_results.pop(0)
        return Ok([])  # type: ignore[call-arg]

    def get_pod(self, *, pod_id: str) -> Any:
        if self.get_pod_results:
            return self.get_pod_results.pop(0)
        return Ok({"id": pod_id, "desiredStatus": "EXITED"})  # type: ignore[call-arg]

    def create_pod(self, *, payload: dict[str, Any]) -> Any:
        self.created_pod_payloads.append(payload)
        if self.create_pod_results:
            return self.create_pod_results.pop(0)
        return Ok({"id": "pod_created"})  # type: ignore[call-arg]

    def stop_pod(self, *, pod_id: str) -> Any:
        if self.stop_pod_results:
            return self.stop_pod_results.pop(0)
        return Ok(None)  # type: ignore[call-arg]

    def start_pod(self, *, pod_id: str) -> Any:
        if self.start_pod_results:
            return self.start_pod_results.pop(0)
        return Ok(None)  # type: ignore[call-arg]

    def delete_pod(self, *, pod_id: str) -> Any:
        if self.delete_pod_results:
            return self.delete_pod_results.pop(0)
        return Ok(None)  # type: ignore[call-arg]

    def list_network_volumes(self) -> Any:
        if self.list_network_volumes_results:
            return self.list_network_volumes_results.pop(0)
        return Ok([])  # type: ignore[call-arg]

    def get_network_volume(self, *, network_volume_id: str) -> Any:
        if self.get_network_volume_results:
            return self.get_network_volume_results.pop(0)
        return Ok({"id": network_volume_id, "name": "vol"})  # type: ignore[call-arg]

    def create_network_volume(self, *, payload: dict[str, Any]) -> Any:
        if self.create_network_volume_results:
            return self.create_network_volume_results.pop(0)
        return Ok({"id": "nv_created", "name": payload.get("name")})  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Provider factory helpers
# ---------------------------------------------------------------------------


def _mk_provider(
    *,
    api: StubApi | None = None,
    api_key: str = "rk",
    key_path: str = "/tmp/test_key",
    with_volume: bool = False,
    pod_id: str | None = None,
    pod_name: str | None = None,
    adapter_ref: str = "hf-org/my-model",
    eval_session: Any | None = None,
) -> RunPodPodInferenceProvider:
    """Build a RunPodPodInferenceProvider via __new__, bypassing __init__."""
    volume_cfg_raw: dict[str, Any] | None = None
    if with_volume:
        volume_cfg_raw = {"id": "vol-123", "name": "my-vol", "size_gb": 50, "data_center_id": "US-KS-2"}

    prov_cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": key_path}},
        cleanup={},
        training={"image_name": "img", "gpu_type": "A40"},
        inference={
            "pod": {"image_name": "inference-img", "gpu_type_ids": ["NVIDIA A40"]},
            **({"volume": volume_cfg_raw} if volume_cfg_raw else {}),
        },
    )
    inf_cfg = InferenceConfig(provider="runpod")

    p = RunPodPodInferenceProvider.__new__(RunPodPodInferenceProvider)
    p._cfg = SimpleNamespace(
        model=SimpleNamespace(name="meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=False)
    )
    p._secrets = SimpleNamespace(runpod_api_key=api_key, hf_token="hf_test")
    p._inf_cfg = inf_cfg
    p._engine_cfg = inf_cfg.engines.vllm
    p._provider_cfg = prov_cfg
    p._volume_cfg = prov_cfg.inference.volume
    p._pod_cfg = prov_cfg.inference.pod
    p._serve_cfg = prov_cfg.inference.serve
    p._api = api
    p._pod_control = api
    p._endpoint_info = None
    p._network_volume_id = None
    p._network_volume_meta = None
    p._pod_id = pod_id
    p._pod_name = pod_name
    p._event_logger = None
    p._eval_session = eval_session
    p._adapter_ref = adapter_ref
    return p


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_provider_name_is_runpod() -> None:
    p = _mk_provider()
    assert p.provider_name == PROVIDER_RUNPOD


def test_provider_type_is_runpod() -> None:
    p = _mk_provider()
    assert p.provider_type == PROVIDER_RUNPOD


def test_get_pipeline_readiness_mode_is_skip() -> None:
    p = _mk_provider()
    assert p.get_pipeline_readiness_mode() == PipelineReadinessMode.SKIP


def test_collect_startup_logs_is_noop(tmp_path: Path) -> None:
    p = _mk_provider()
    p.collect_startup_logs(local_path=tmp_path)  # must not raise


def test_get_capabilities() -> None:
    p = _mk_provider()
    caps = p.get_capabilities()
    assert caps.provider_type == PROVIDER_RUNPOD
    assert "vllm" in caps.supported_engines
    assert caps.supports_lora is True


def test_get_endpoint_info_none_before_deploy() -> None:
    p = _mk_provider()
    assert p.get_endpoint_info() is None


def test_set_event_logger_stores_logger() -> None:
    p = _mk_provider()
    fake_logger = object()
    p.set_event_logger(fake_logger)  # type: ignore[arg-type]
    assert p._event_logger is fake_logger


# ---------------------------------------------------------------------------
# deploy() — early error paths
# ---------------------------------------------------------------------------


def test_deploy_local_adapter_path_returns_err() -> None:
    p = _mk_provider()
    for bad_path in ["/local/path", "~/adapter", "./adapter", "file:///path"]:
        res = p.deploy(bad_path, run_id="r1", base_model_id="llama")
        assert res.is_failure(), f"expected failure for: {bad_path}"
        assert res.unwrap_err().code == "RUNPOD_ADAPTER_LOCAL_PATH_NOT_SUPPORTED"


def test_deploy_missing_api_key_returns_err() -> None:
    p = _mk_provider(api_key="")
    res = p.deploy("hf-org/model", run_id="r1", base_model_id="llama")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_API_KEY_MISSING"


def test_deploy_none_api_key_returns_err() -> None:
    p = _mk_provider()
    p._secrets = SimpleNamespace(runpod_api_key=None, hf_token=None)
    res = p.deploy("hf-org/model", run_id="r1", base_model_id="llama")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_API_KEY_MISSING"


def test_deploy_missing_ssh_key_file_returns_err(tmp_path: Path) -> None:
    missing_key = str(tmp_path / "id_ed25519")
    p = _mk_provider(key_path=missing_key)
    res = p.deploy("hf-org/model", run_id="r1", base_model_id="llama")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SSH_KEY_NOT_FOUND"


# ---------------------------------------------------------------------------
# deploy() — full flow (patching RunPodPodsRESTClient + internals)
# ---------------------------------------------------------------------------


def test_deploy_success_no_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: Ok(("pod-123", "pod-name")))
    monkeypatch.setattr(p, "_stop_pod_if_running", lambda **kw: Ok(None))

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert res.is_success()
    info = res.unwrap()
    assert info.resource_id == "pod-123"
    assert info.endpoint_url == "http://127.0.0.1:8000/v1"
    assert info.health_url == "http://127.0.0.1:8000/v1/models"
    assert p._pod_id == "pod-123"


def test_deploy_keep_running_skips_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))
    stop_called: list[bool] = []

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: Ok(("pod-456", "pod-name")))
    monkeypatch.setattr(p, "_stop_pod_if_running", lambda **kw: (stop_called.append(True), Ok(None))[1])

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama", keep_running=True)
    assert res.is_success()
    assert not stop_called  # stop must NOT be called when keep_running=True


def test_deploy_with_volume_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file), with_volume=True)

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_network_volume", lambda: Ok("vol-123"))
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: Ok(("pod-789", "pod-name")))
    monkeypatch.setattr(p, "_stop_pod_if_running", lambda **kw: Ok(None))

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert res.is_success()
    assert p._network_volume_id == "vol-123"


def test_deploy_volume_ensure_failure_returns_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file), with_volume=True)

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(
        p,
        "_ensure_network_volume",
        lambda: Err(InferenceError(message="vol fail", code="RUNPOD_VOLUME_LIST_FAILED")),
    )

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert res.is_failure()
    assert "RUNPOD_VOLUME_LIST_FAILED" in res.unwrap_err().code


def test_deploy_pod_ensure_failure_returns_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(
        p,
        "_ensure_pod",
        lambda **kw: Err(InferenceError(message="pod fail", code="RUNPOD_POD_CREATE_FAILED")),
    )

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert res.is_failure()
    assert "RUNPOD_POD_CREATE_FAILED" in res.unwrap_err().code


def test_deploy_stop_failure_returns_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: Ok(("pod-123", "pod-name")))
    monkeypatch.setattr(
        p,
        "_stop_pod_if_running",
        lambda **kw: Err(InferenceError(message="stop failed", code="RUNPOD_POD_STOP_FAILED")),
    )

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert res.is_failure()
    assert "RUNPOD_POD_STOP_FAILED" in res.unwrap_err().code


def test_deploy_provider_error_from_volume_wrapped_as_inference_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file), with_volume=True)

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(
        p,
        "_ensure_network_volume",
        lambda: Err(ProviderError(message="provider err", code="PROV_ERR")),
    )

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert res.is_failure()
    assert isinstance(res.unwrap_err(), InferenceError)
    assert res.unwrap_err().code == "RUNPOD_VOLUME_ENSURE_FAILED"


# ---------------------------------------------------------------------------
# undeploy()
# ---------------------------------------------------------------------------


def test_undeploy_no_api_returns_ok() -> None:
    p = _mk_provider()
    assert p._api is None
    res = p.undeploy()
    assert res.is_success()


def test_undeploy_no_pod_id_returns_ok() -> None:
    p = _mk_provider(api=StubApi())
    assert p._pod_id is None
    res = p.undeploy()
    assert res.is_success()


def test_undeploy_success_stops_pod() -> None:
    api = StubApi()
    p = _mk_provider(api=api, pod_id="pod-1")
    res = p.undeploy()
    assert res.is_success()


def test_undeploy_stop_failure_returns_err() -> None:
    api = StubApi(
        stop_pod_results=[Err(ProviderError(message="stop failed", code="RUNPOD_STOP"))]
    )
    p = _mk_provider(api=api, pod_id="pod-1")
    res = p.undeploy()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_STOP_FAILED"


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


def test_health_check_not_deployed_returns_err() -> None:
    p = _mk_provider()
    res = p.health_check()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_NOT_DEPLOYED"


def test_health_check_no_pod_id_returns_err() -> None:
    p = _mk_provider(api=StubApi())
    res = p.health_check()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_NOT_DEPLOYED"


def test_health_check_pod_found_returns_true() -> None:
    api = StubApi(get_pod_results=[Ok({"id": "pod-1", "desiredStatus": "EXITED"})])  # type: ignore[call-arg]
    p = _mk_provider(api=api, pod_id="pod-1")
    res = p.health_check()
    assert res.is_success()
    assert res.unwrap() is True


def test_health_check_get_pod_failure_returns_err() -> None:
    api = StubApi(
        get_pod_results=[Err(ProviderError(message="not found", code="RUNPOD_NOT_FOUND"))]
    )
    p = _mk_provider(api=api, pod_id="pod-1")
    res = p.health_check()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_GET_FAILED"


# ---------------------------------------------------------------------------
# activate_for_eval()
# ---------------------------------------------------------------------------


def test_activate_for_eval_no_api_returns_err() -> None:
    p = _mk_provider(pod_id="pod-1")
    p._api = None
    res = p.activate_for_eval()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_NOT_DEPLOYED"


def test_activate_for_eval_no_pod_id_returns_err() -> None:
    p = _mk_provider(api=StubApi())
    p._pod_id = None
    res = p.activate_for_eval()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_NOT_DEPLOYED"


def test_activate_for_eval_empty_adapter_ref_returns_err() -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="")
    res = p.activate_for_eval()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_ADAPTER_REF_MISSING"


def test_activate_for_eval_whitespace_adapter_ref_returns_err() -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="   ")
    res = p.activate_for_eval()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_ADAPTER_REF_MISSING"


def test_activate_for_eval_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="hf-org/model")

    fake_session = SimpleNamespace(endpoint_url="http://127.0.0.1:8000/v1")

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.pod_session.activate",
        lambda **kw: Ok(fake_session),  # type: ignore[call-arg]
    )

    res = p.activate_for_eval()
    assert res.is_success()
    assert res.unwrap() == "http://127.0.0.1:8000/v1"
    assert p._eval_session is fake_session


def test_activate_for_eval_session_failure_returns_inference_err(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="hf-org/model")

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.pod_session.activate",
        lambda **kw: Err(ProviderError(message="ssh fail", code="SSH_FAIL")),
    )

    res = p.activate_for_eval()
    assert res.is_failure()
    err = res.unwrap_err()
    assert isinstance(err, InferenceError)
    assert err.code == "RUNPOD_EVAL_ACTIVATE_FAILED"


def test_activate_for_eval_session_inference_error_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="hf-org/model")

    from src.utils.result import ProviderError

    orig_err = ProviderError(message="vllm error", code="VLLM_FAIL")
    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.pod_session.activate",
        lambda **kw: Err(orig_err),
    )

    res = p.activate_for_eval()
    assert res.is_failure()
    err = res.unwrap_err()
    # pod_session errors are always wrapped into InferenceError at provider boundary
    assert err.code == "RUNPOD_EVAL_ACTIVATE_FAILED"
    assert "vllm error" in err.message


# ---------------------------------------------------------------------------
# deactivate_after_eval()
# ---------------------------------------------------------------------------


def test_deactivate_after_eval_no_api_returns_ok() -> None:
    p = _mk_provider()
    p._api = None
    res = p.deactivate_after_eval()
    assert res.is_success()


def test_deactivate_after_eval_no_session_no_pod_returns_ok() -> None:
    p = _mk_provider(api=StubApi())
    p._eval_session = None
    p._pod_id = None
    res = p.deactivate_after_eval()
    assert res.is_success()


def test_deactivate_after_eval_no_session_with_pod_deletes_pod() -> None:
    api = StubApi()
    p = _mk_provider(api=api, pod_id="pod-1")
    p._eval_session = None

    res = p.deactivate_after_eval()
    assert res.is_success()
    assert p._pod_id is None


def test_deactivate_after_eval_no_session_delete_failure_returns_err() -> None:
    api = StubApi(
        delete_pod_results=[Err(ProviderError(message="delete fail", code="DELETE_FAIL"))]
    )
    p = _mk_provider(api=api, pod_id="pod-1")
    p._eval_session = None

    res = p.deactivate_after_eval()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_DELETE_FAILED"


def test_deactivate_after_eval_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = SimpleNamespace(endpoint_url="http://127.0.0.1:8000/v1")
    p = _mk_provider(api=StubApi(), pod_id="pod-1", eval_session=fake_session)

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.pod_session.deactivate",
        lambda **kw: Ok(None),  # type: ignore[call-arg]
    )

    res = p.deactivate_after_eval()
    assert res.is_success()
    assert p._eval_session is None


def test_deactivate_after_eval_session_failure_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = SimpleNamespace(endpoint_url="http://127.0.0.1:8000/v1")
    p = _mk_provider(api=StubApi(), pod_id="pod-1", eval_session=fake_session)

    monkeypatch.setattr(
        "src.providers.runpod.inference.pods.pod_session.deactivate",
        lambda **kw: Err(ProviderError(message="deactivate fail", code="DEACT_FAIL")),
    )

    res = p.deactivate_after_eval()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_EVAL_DEACTIVATE_FAILED"


# ---------------------------------------------------------------------------
# _ensure_pod()
# ---------------------------------------------------------------------------


def test_ensure_pod_reuses_existing_stopped_pod() -> None:
    api = StubApi(
        list_pods_results=[Ok([{"id": "pod-stopped", "desiredStatus": "EXITED"}])]  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    assert res.is_success()
    pod_id, _ = res.unwrap()
    assert pod_id == "pod-stopped"


def test_ensure_pod_prefers_stopped_over_running() -> None:
    api = StubApi(
        list_pods_results=[
            Ok(  # type: ignore[call-arg]
                [
                    {"id": "pod-running", "desiredStatus": "RUNNING"},
                    {"id": "pod-stopped", "desiredStatus": "EXITED"},
                ]
            )
        ]
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    assert res.is_success()
    pod_id, _ = res.unwrap()
    assert pod_id == "pod-stopped"


def test_ensure_pod_falls_back_to_running_pod() -> None:
    api = StubApi(
        list_pods_results=[Ok([{"id": "pod-running", "desiredStatus": "RUNNING"}])]  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    assert res.is_success()
    pod_id, _ = res.unwrap()
    assert pod_id == "pod-running"


def test_ensure_pod_creates_new_when_none_exist(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[
            Ok([]),  # type: ignore[call-arg]  # initial list → empty
            Ok([]),  # type: ignore[call-arg]  # post-create re-check → empty (create succeeded directly)
        ],
        create_pod_results=[Ok({"id": "pod-new"})],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    assert res.is_success()
    pod_id, _ = res.unwrap()
    assert pod_id == "pod-new"


def test_ensure_pod_create_fails_but_pod_appears_on_relist(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[
            Ok([]),  # type: ignore[call-arg]  # initial list
            Ok([{"id": "pod-appeared", "desiredStatus": "RUNNING"}]),  # type: ignore[call-arg]  # post-create relist
        ],
        create_pod_results=[Err(ProviderError(message="timeout", code="TIMEOUT"))],
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    assert res.is_success()
    pod_id, _ = res.unwrap()
    assert pod_id == "pod-appeared"


def test_ensure_pod_list_failure_returns_err() -> None:
    api = StubApi(
        list_pods_results=[Err(ProviderError(message="list failed", code="LIST_FAIL"))]
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_LIST_FAILED"


def test_ensure_pod_all_create_attempts_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # All create and list attempts return failure / empty
    call_count = [0]

    def _list_pods(**kw: Any) -> Any:
        call_count[0] += 1
        return Ok([])  # type: ignore[call-arg]

    api = StubApi(
        create_pod_results=[Err(ProviderError(message="fail", code="FAIL"))] * 8,
    )
    api.list_pods = _list_pods  # type: ignore[method-assign]

    p = _mk_provider(api=api)
    monkeypatch.setattr("time.sleep", lambda s: None)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    assert res.is_failure()
    assert "RUNPOD_POD_CREATE_FAILED" in res.unwrap_err().code


def test_ensure_pod_with_network_volume_includes_volume_id(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[Ok([])],  # type: ignore[call-arg]
        create_pod_results=[Ok({"id": "pod-vol-new"})],  # type: ignore[call-arg]
    )
    api.list_pods_results.append(Ok([]))  # post-create relist  # type: ignore[call-arg]
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id="vol-abc", key_path=tmp_path / "key")
    assert res.is_success()
    assert api.created_pod_payloads[0]["networkVolumeId"] == "vol-abc"


def test_ensure_pod_uses_deterministic_name_for_volume(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[Ok([]), Ok([])],  # type: ignore[call-arg]
        create_pod_results=[Ok({"id": "pod-vol-new"})],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)

    res = p._ensure_pod(network_volume_id="vol-abc", key_path=tmp_path / "key")

    assert res.is_success()
    pod_id, pod_name = res.unwrap()
    payload = api.created_pod_payloads[0]
    assert pod_id == "pod-vol-new"
    assert payload["name"] == pod_name
    assert payload["name"].startswith(f"{p._pod_cfg.name_prefix}-")
    assert "ephemeral" not in payload["name"]


def test_ensure_pod_without_volume_uses_volume_in_gb(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[Ok([])],  # type: ignore[call-arg]
        create_pod_results=[Ok({"id": "pod-ephemeral"})],  # type: ignore[call-arg]
    )
    api.list_pods_results.append(Ok([]))  # type: ignore[call-arg]
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    assert res.is_success()
    payload = api.created_pod_payloads[0]
    assert "networkVolumeId" not in payload
    assert "volumeInGb" in payload


def test_ensure_pod_without_volume_uses_ephemeral_name(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[Ok([]), Ok([])],  # type: ignore[call-arg]
        create_pod_results=[Ok({"id": "pod-ephemeral"})],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)

    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")

    assert res.is_success()
    _, pod_name = res.unwrap()
    assert pod_name.endswith("-ephemeral")
    assert api.created_pod_payloads[0]["name"] == pod_name


def test_ensure_pod_pod_without_id_returns_err() -> None:
    api = StubApi(
        list_pods_results=[Ok([{"desiredStatus": "EXITED"}])]  # type: ignore[call-arg]  # no "id" field
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_NO_ID"


def test_ensure_pod_create_success_without_id_but_relist_finds_pod(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_results=[
            Ok([]),  # type: ignore[call-arg]
            Ok([{"id": "pod-after-relist", "desiredStatus": "RUNNING"}]),  # type: ignore[call-arg]
        ],
        create_pod_results=[Ok({"desiredStatus": "RUNNING"})],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)

    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")

    assert res.is_success()
    assert res.unwrap()[0] == "pod-after-relist"


# ---------------------------------------------------------------------------
# _stop_pod_if_running()
# ---------------------------------------------------------------------------


def test_stop_pod_if_running_running_pod_calls_stop() -> None:
    api = StubApi(
        get_pod_results=[Ok({"id": "pod-1", "desiredStatus": "RUNNING"})],  # type: ignore[call-arg]
        stop_pod_results=[Ok(None)],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    res = p._stop_pod_if_running(pod_id="pod-1")
    assert res.is_success()


def test_stop_pod_if_running_stopped_pod_skips_stop() -> None:
    api = StubApi(
        get_pod_results=[Ok({"id": "pod-1", "desiredStatus": "EXITED"})],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    res = p._stop_pod_if_running(pod_id="pod-1")
    assert res.is_success()
    # No stop_pod call was consumed from the queue
    assert len(api.stop_pod_results) == 0


def test_stop_pod_if_running_get_failure_returns_err() -> None:
    api = StubApi(
        get_pod_results=[Err(ProviderError(message="get failed", code="GET_FAIL"))]
    )
    p = _mk_provider(api=api)
    res = p._stop_pod_if_running(pod_id="pod-1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_GET_FAILED"


def test_stop_pod_if_running_stop_failure_returns_err() -> None:
    api = StubApi(
        get_pod_results=[Ok({"id": "pod-1", "desiredStatus": "RUNNING"})],  # type: ignore[call-arg]
        stop_pod_results=[Err(ProviderError(message="stop failed", code="STOP_FAIL"))],
    )
    p = _mk_provider(api=api)
    res = p._stop_pod_if_running(pod_id="pod-1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_POD_STOP_FAILED"


# ---------------------------------------------------------------------------
# _ensure_network_volume() — via id lookup
# ---------------------------------------------------------------------------


def test_ensure_network_volume_by_id_success() -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        get_network_volume_results=[
            Ok({"id": "vol-123", "name": "my-vol"})  # type: ignore[call-arg]
        ]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id="vol-123", name="my-vol", size_gb=50)
    res = p._ensure_network_volume()
    assert res.is_success()
    assert res.unwrap() == "vol-123"


def test_ensure_network_volume_by_id_not_found_returns_err() -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        get_network_volume_results=[
            Err(ProviderError(message="not found", code="NOT_FOUND"))
        ]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id="bad-id", name="my-vol", size_gb=50)
    res = p._ensure_network_volume()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_VOLUME_NOT_FOUND"


def test_ensure_network_volume_list_failure_returns_err() -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_results=[Err(ProviderError(message="list fail", code="LIST_FAIL"))]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")
    res = p._ensure_network_volume()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_VOLUME_LIST_FAILED"


def test_ensure_network_volume_reuses_existing_match_by_name_and_dc() -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_results=[
            Ok(  # type: ignore[call-arg]
                [
                    {"id": "vol-123", "name": "my-vol", "dataCenterId": "US-KS-2"},
                    {"id": "vol-999", "name": "my-vol", "dataCenterId": "EU-RO-1"},
                ]
            )
        ]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    res = p._ensure_network_volume()

    assert res.is_success()
    assert res.unwrap() == "vol-123"
    assert p._network_volume_meta == {"id": "vol-123", "name": "my-vol", "dataCenterId": "US-KS-2"}


def test_ensure_network_volume_returns_ambiguous_when_multiple_name_matches() -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_results=[
            Ok(  # type: ignore[call-arg]
                [
                    {"id": "vol-1", "name": "my-vol", "dataCenterId": "US-KS-2"},
                    {"id": "vol-2", "name": "my-vol", "dataCenterId": "US-KS-2"},
                ]
            )
        ]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    res = p._ensure_network_volume()

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_VOLUME_AMBIGUOUS"


def test_ensure_network_volume_missing_data_center_returns_err_before_create() -> None:
    api = StubApi(list_network_volumes_results=[Ok([])])  # type: ignore[call-arg]
    p = _mk_provider(api=api)
    p._volume_cfg = SimpleNamespace(id=None, name="my-vol", size_gb=50, data_center_id=None)

    res = p._ensure_network_volume()

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_VOLUME_DATA_CENTER_MISSING"


def test_ensure_network_volume_existing_match_without_id_returns_volume_no_id_err() -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_results=[Ok([{"name": "my-vol", "dataCenterId": "US-KS-2"}])]  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    res = p._ensure_network_volume()

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_VOLUME_NO_ID"


def test_ensure_network_volume_create_success_without_id_but_relist_finds_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    monkeypatch.setattr("time.sleep", lambda s: None)
    api = StubApi(
        list_network_volumes_results=[
            Ok([]),  # type: ignore[call-arg]
            Ok([{"id": "vol-after", "name": "my-vol", "dataCenterId": "US-KS-2"}]),  # type: ignore[call-arg]
        ],
        create_network_volume_results=[Ok({"name": "my-vol"})],  # type: ignore[call-arg]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    res = p._ensure_network_volume()

    assert res.is_success()
    assert res.unwrap() == "vol-after"
